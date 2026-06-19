import argparse
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display Arducam UVC cameras using exact DirectShow format selection."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2], help="DirectShow camera indices.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=800, help="Requested camera height.")
    parser.add_argument("--format", default="MJPG", help="Requested DirectShow media subtype, such as MJPG or YUY2.")
    parser.add_argument("--cols", type=int, default=0, help="Grid columns. Defaults to a square-ish grid.")
    parser.add_argument("--display-height", type=int, default=400, help="Displayed height for each camera tile.")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many seconds. 0 runs until q/Esc.")
    parser.add_argument("--no-display", action="store_true", help="Benchmark without opening a window.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Write one grid frame to this image path.")
    parser.add_argument("--scan", action="store_true", help="List DirectShow camera devices and exit.")
    parser.add_argument("--list-formats", action="store_true", help="List advertised DirectShow formats and exit.")
    parser.add_argument("--startup-timeout", type=float, default=3.0, help="Seconds to wait for first frames.")
    return parser.parse_args()


def list_devices():
    return FilterGraph().get_input_devices()


def get_formats(device_index):
    graph = FilterGraph()
    graph.add_video_input_device(device_index)
    try:
        return graph.get_input_device().get_formats()
    finally:
        graph.remove_filters()


def find_format_index(device_index, subtype, width, height):
    subtype = subtype.upper()
    matches = [
        fmt
        for fmt in get_formats(device_index)
        if fmt["media_type_str"].upper() == subtype
        and fmt["width"] == width
        and fmt["height"] == height
    ]
    if not matches:
        raise RuntimeError(f"Camera {device_index} has no {subtype} {width}x{height} DirectShow format.")
    return matches[0]["index"]


class DShowCamera:
    def __init__(self, device_index, format_index):
        self.device_index = device_index
        self.format_index = format_index
        self.graph = FilterGraph()
        self.latest_frame = None
        self.frame_count = 0
        self.started_at = None
        self.last_frame_at = None
        self.running = False
        self.thread = None

    def start(self):
        self.graph.add_video_input_device(self.device_index)
        self.graph.get_input_device().set_format(self.format_index)
        self.graph.add_sample_grabber(self._on_frame)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.started_at = time.perf_counter()
        self.running = True
        self.thread = threading.Thread(target=self._request_loop, name=f"dshow-camera-{self.device_index}", daemon=True)
        self.thread.start()

    def _on_frame(self, frame):
        self.latest_frame = frame
        self.frame_count += 1
        self.last_frame_at = time.perf_counter()

    def _request_loop(self):
        while self.running:
            self.request_frame()
            time.sleep(0.001)

    def request_frame(self):
        self.graph.grab_frame()

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return self.frame_count / elapsed

    def has_frame(self):
        return self.latest_frame is not None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()


BUTTON_WIDTH = 110
BUTTON_HEIGHT = 34
BUTTON_MARGIN = 10


def rotate_frame(frame, steps):
    steps %= 4
    if steps == 1:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if steps == 2:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if steps == 3:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_overlay(frame, label, fps):
    frame = frame.copy()
    height, width = frame.shape[:2]
    text = f"{label} | {width}x{height} | {fps:5.1f} FPS"
    cv2.rectangle(frame, (8, 8), (min(width - 8, 430), 44), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_panel_overlay(frame, text, rotation_degrees):
    frame = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (8, 8), (min(width - 8, 560), 44), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    x1 = max(BUTTON_MARGIN, width - BUTTON_WIDTH - BUTTON_MARGIN)
    y1 = BUTTON_MARGIN
    x2 = min(width - BUTTON_MARGIN, x1 + BUTTON_WIDTH)
    y2 = min(height - BUTTON_MARGIN, y1 + BUTTON_HEIGHT)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 35), thickness=-1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), thickness=1)
    cv2.putText(
        frame,
        f"ROT {rotation_degrees}",
        (x1 + 10, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame, (x1, y1, x2, y2)


def resize_to_height(frame, target_height):
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    target_width = max(1, int(width * (target_height / height)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def fit_to_tile(frame, tile_width, tile_height):
    height, width = frame.shape[:2]
    scale = min(tile_width / width, tile_height / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    x = (tile_width - resized_width) // 2
    y = (tile_height - resized_height) // 2
    tile[y : y + resized_height, x : x + resized_width] = resized
    return tile


def make_grid(cameras, cols, display_height, rotations=None):
    rotations = rotations or {}
    entries = []
    for camera in cameras:
        rotation_steps = rotations.get(camera.device_index, 0)
        rotation_degrees = (rotation_steps % 4) * 90
        if camera.latest_frame is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            text = f"Camera {camera.device_index} | waiting"
        else:
            frame = rotate_frame(camera.latest_frame, rotation_steps)
            height, width = frame.shape[:2]
            text = f"Camera {camera.device_index} | {width}x{height} | {camera.average_fps():5.1f} FPS"
            frame = resize_to_height(frame, display_height)
        entries.append(
            {
                "camera_id": camera.device_index,
                "frame": frame,
                "text": text,
                "rotation_degrees": rotation_degrees,
            }
        )

    tile_height = max(entry["frame"].shape[0] for entry in entries)
    tile_width = max(entry["frame"].shape[1] for entry in entries)
    rows = []
    button_rects = []
    for start in range(0, len(entries), cols):
        row_tiles = []
        row_index = start // cols
        for col_index, entry in enumerate(entries[start : start + cols]):
            fitted = fit_to_tile(entry["frame"], tile_width, tile_height)
            fitted, rect = draw_panel_overlay(fitted, entry["text"], entry["rotation_degrees"])
            x1, y1, x2, y2 = rect
            global_x = col_index * tile_width
            global_y = row_index * tile_height
            button_rects.append(
                {
                    "camera_id": entry["camera_id"],
                    "rect": (global_x + x1, global_y + y1, global_x + x2, global_y + y2),
                }
            )
            row_tiles.append(fitted)
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows), button_rects


def print_formats(device_indices):
    devices = list_devices()
    for index in device_indices:
        print(f"\nDevice {index}: {devices[index]}")
        formats = get_formats(index)
        for fmt in sorted(formats, key=lambda item: (item["media_type_str"], item["width"], item["height"])):
            print(
                f"{fmt['index']:3d} {fmt['media_type_str']:4s} "
                f"{fmt['width']}x{fmt['height']} "
                f"fps={fmt['max_framerate']:.1f}-{fmt['min_framerate']:.1f}"
            )


def main():
    args = parse_args()
    if args.scan:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return
    if args.list_formats:
        print_formats(args.cameras)
        return

    cols = args.cols or math.ceil(math.sqrt(len(args.cameras)))
    cameras = []
    rotations = {camera_id: 0 for camera_id in args.cameras}
    button_rects = []

    def handle_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in button_rects:
            x1, y1, x2, y2 = button["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                camera_id = button["camera_id"]
                rotations[camera_id] = (rotations.get(camera_id, 0) + 1) % 4
                print(f"Camera {camera_id} rotation: {rotations[camera_id] * 90} degrees", flush=True)
                break

    try:
        for device_index in args.cameras:
            format_index = find_format_index(device_index, args.format, args.width, args.height)
            print(f"Camera {device_index}: using {args.format.upper()} {args.width}x{args.height}, format index {format_index}")
            camera = DShowCamera(device_index, format_index)
            camera.start()
            cameras.append(camera)

        startup_deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < startup_deadline and not all(camera.has_frame() for camera in cameras):
            time.sleep(0.02)
        for camera in cameras:
            if not camera.has_frame():
                print(f"Warning: camera {camera.device_index} has not delivered an initial frame yet.", flush=True)

        stop_at = time.perf_counter() + args.duration if args.duration else None
        if not args.no_display:
            cv2.namedWindow("DirectShow Arducam Viewer", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("DirectShow Arducam Viewer", handle_mouse)

        while True:
            if args.snapshot and not all(camera.latest_frame is not None for camera in cameras):
                continue

            if not args.no_display or args.snapshot:
                grid, button_rects = make_grid(cameras, cols, args.display_height, rotations)
                if args.snapshot:
                    cv2.imwrite(str(args.snapshot), grid)
                    print(f"Wrote snapshot: {args.snapshot}")
                    args.snapshot = None
                if not args.no_display:
                    cv2.imshow("DirectShow Arducam Viewer", grid)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

            if stop_at is not None and time.perf_counter() >= stop_at:
                break

        for camera in cameras:
            print(f"Camera {camera.device_index} average FPS: {camera.average_fps():.1f}")
            if camera.frame_count == 0:
                print(
                    f"Warning: camera {camera.device_index} started but delivered no frames. "
                    "Try a different resolution such as 1280x720.",
                    flush=True,
                )
    finally:
        for camera in cameras:
            camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
