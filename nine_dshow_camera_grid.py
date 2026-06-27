import argparse
import math
import threading
import time

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index, fit_to_tile, rotate_frame


WINDOW_NAME = "Nine DirectShow Cameras"


def parse_args():
    parser = argparse.ArgumentParser(description="Display a DirectShow grid of USB cameras.")
    parser.add_argument("--cameras", type=int, nargs="+", default=None, help="Defaults to every DirectShow device named USB Camera.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--display-height", type=int, default=240)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. 0 runs until q/Esc.")
    return parser.parse_args()


def list_devices():
    return FilterGraph().get_input_devices()


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
        self.thread = threading.Thread(target=self._request_loop, name=f"dshow-grid-{self.camera_id}", daemon=True)
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


def resize_to_height(frame, target_height):
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    target_width = max(1, int(width * (target_height / height)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_overlay(frame, camera_id, fps):
    frame = frame.copy()
    text = f"cam {camera_id} | {fps:4.1f} FPS"
    cv2.rectangle(frame, (6, 6), (min(frame.shape[1] - 6, 250), 36), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def make_grid(cameras, cols, display_height, rotation):
    entries = []
    for camera in cameras:
        frame = camera.get_frame()
        if frame is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(frame, f"cam {camera.camera_id} waiting", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        else:
            frame = rotate_frame(frame, rotation // 90)
            frame = resize_to_height(frame, display_height)
            frame = draw_overlay(frame, camera.camera_id, camera.average_fps())
        entries.append(frame)

    tile_height = max(frame.shape[0] for frame in entries)
    tile_width = max(frame.shape[1] for frame in entries)
    rows = []
    for start in range(0, len(entries), cols):
        row_tiles = [fit_to_tile(frame, tile_width, tile_height) for frame in entries[start : start + cols]]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def main():
    args = parse_args()
    if args.scan:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return

    if args.cameras is None:
        args.cameras = [index for index, device in enumerate(list_devices()) if device == "USB Camera"]

    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.cameras))))
    cameras = []
    try:
        for camera_id in args.cameras:
            print(f"starting camera {camera_id}...", flush=True)
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            try:
                camera.start()
            except Exception as exc:
                print(f"failed to start camera {camera_id}: {exc}", flush=True)
                raise
            cameras.append(camera)
            print(f"started camera {camera_id}", flush=True)

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.get_frame() is not None for camera in cameras):
            time.sleep(0.05)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        started_at = time.perf_counter()
        while True:
            grid = make_grid(cameras, cols, args.display_height, args.rotation)
            cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if args.duration > 0 and time.perf_counter() - started_at >= args.duration:
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
