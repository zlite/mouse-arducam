import argparse
import math
import threading
import time

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index, fit_to_tile, get_formats, resize_to_height, rotate_frame


WINDOW_NAME = "Ten DirectShow Cameras"


def parse_args():
    parser = argparse.ArgumentParser(description="Display ten DirectShow USB cameras.")
    parser.add_argument("--cameras", type=int, nargs="+", default=None, help="Defaults to the first ten USB Camera devices.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--cols", type=int, default=0, help="Uniform grid columns. Overrides --row-layout when > 0.")
    parser.add_argument("--row-layout", default="3,3,3,1", help="Comma-separated cameras per row, e.g. 3,3,3,1.")
    parser.add_argument("--display-height", type=int, default=180)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--startup-timeout", type=float, default=4.0)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--list-formats", action="store_true")
    return parser.parse_args()


def list_devices():
    return FilterGraph().get_input_devices()


def print_formats(device_indices):
    devices = list_devices()
    for index in device_indices:
        device_name = devices[index] if index < len(devices) else "<missing>"
        print(f"\nDevice {index}: {device_name}", flush=True)
        for fmt in sorted(get_formats(index), key=lambda item: (item["media_type_str"], item["width"], item["height"])):
            print(
                f"{fmt['index']:3d} {fmt['media_type_str']:4s} "
                f"{fmt['width']}x{fmt['height']} "
                f"fps={fmt['max_framerate']:.1f}-{fmt['min_framerate']:.1f}",
                flush=True,
            )


class DShowCamera:
    def __init__(self, camera_id, format_index):
        self.camera_id = camera_id
        self.format_index = format_index
        self.graph = FilterGraph()
        self.latest_frame = None
        self.frame_count = 0
        self.started_at = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        self.graph.add_video_input_device(self.camera_id)
        self.graph.get_input_device().set_format(self.format_index)
        self.graph.add_sample_grabber(self._on_frame)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.started_at = time.perf_counter()
        self.running = True
        self.thread = threading.Thread(target=self._request_loop, name=f"dshow-ten-grid-{self.camera_id}", daemon=True)
        self.thread.start()

    def _on_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            self.frame_count += 1

    def _request_loop(self):
        while self.running:
            self.graph.grab_frame()
            time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        return self.frame_count / max(time.perf_counter() - self.started_at, 1e-6)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()


def draw_overlay(frame, camera_id, fps):
    frame = frame.copy()
    text = f"cam {camera_id}  {fps:4.1f}"
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, 116), 22), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def parse_row_layout(layout_text, camera_count):
    if not layout_text:
        return []
    rows = [int(part.strip()) for part in layout_text.split(",") if part.strip()]
    if any(row <= 0 for row in rows):
        raise ValueError("--row-layout values must be positive integers")
    if sum(rows) < camera_count:
        rows.append(camera_count - sum(rows))
    return rows


def make_row_layout(camera_count, cols, row_layout):
    if cols > 0:
        return [cols for _start in range(0, camera_count, cols)]
    rows = parse_row_layout(row_layout, camera_count)
    if not rows:
        cols = max(1, math.ceil(math.sqrt(camera_count)))
        return [cols for _start in range(0, camera_count, cols)]
    return rows


def make_grid(cameras, row_counts, display_height, rotation_steps):
    entries = []
    for camera in cameras:
        frame = camera.get_frame()
        if frame is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(frame, f"cam {camera.camera_id} waiting", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        else:
            frame = rotate_frame(frame, rotation_steps)
            frame = resize_to_height(frame, display_height)
        entries.append(draw_overlay(frame, camera.camera_id, camera.average_fps()))

    if not entries:
        frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
        cv2.putText(frame, "no cameras selected", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return frame

    tile_height = display_height
    tile_width = int(round(display_height * 16 / 9))
    rows = []
    start = 0
    max_cols = max(row_counts)
    for row_count in row_counts:
        row_entries = entries[start : start + row_count]
        start += row_count
        if not row_entries:
            break
        row_tiles = [fit_to_tile(frame, tile_width, tile_height) for frame in row_entries]
        while len(row_tiles) < max_cols:
            row_tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def main():
    args = parse_args()
    devices = list_devices()
    if args.scan:
        for index, device in enumerate(devices):
            print(f"{index}: {device}")
        return

    if args.cameras is None:
        exact_usb = [index for index, device in enumerate(devices) if device == "USB Camera"]
        likely_usb = [
            index
            for index, device in enumerate(devices)
            if "usb" in device.lower() or "arducam" in device.lower()
        ]
        args.cameras = (exact_usb or likely_usb)[:10]
    if args.list_formats:
        print_formats(args.cameras)
        return

    if not args.cameras:
        print("No cameras were auto-selected.", flush=True)
        print("Run this to see DirectShow devices:", flush=True)
        print("  python ten_dshow_camera_grid.py --scan", flush=True)
        print("Then pass the ArduCam indices explicitly, for example:", flush=True)
        print("  python ten_dshow_camera_grid.py --cameras 0 1 2 3 4 5 6 7 8 9", flush=True)
        return

    if len(args.cameras) != 10:
        print(f"Warning: displaying {len(args.cameras)} cameras, not exactly 10.", flush=True)
    print("selected devices:", flush=True)
    for camera_id in args.cameras:
        device_name = devices[camera_id] if camera_id < len(devices) else "<missing>"
        print(f"  {camera_id}: {device_name}", flush=True)

    row_counts = make_row_layout(len(args.cameras), args.cols, args.row_layout)
    print(f"row layout: {','.join(str(count) for count in row_counts)}", flush=True)
    cameras = []
    window_sized = False
    try:
        for camera_id in args.cameras:
            device_name = devices[camera_id] if camera_id < len(devices) else "<missing>"
            print(f"starting camera {camera_id}: {device_name}...", flush=True)
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.get_frame() is not None for camera in cameras):
            time.sleep(0.05)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        print("q/Esc: quit", flush=True)
        while True:
            grid = make_grid(cameras, row_counts, args.display_height, args.rotation // 90)
            if not window_sized:
                cv2.resizeWindow(WINDOW_NAME, grid.shape[1], grid.shape[0])
                window_sized = True
            cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
