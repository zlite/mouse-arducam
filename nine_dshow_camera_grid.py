import argparse
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index, fit_to_tile, rotate_frame


WINDOW_NAME = "Nine DirectShow Cameras"
POSITION_SLOTS = [
    ("front_1", "Front 1"),
    ("front_2", "Front 2"),
    ("back_1", "Back 1"),
    ("back_2", "Back 2"),
    ("side_1", "Side 1"),
    ("side_2", "Side 2"),
    ("top_1", "Top 1"),
    ("top_2", "Top 2"),
    ("top_3", "Top 3"),
]
POSITION_LABELS = dict(POSITION_SLOTS)
POSITION_ORDER = {position: index for index, (position, _) in enumerate(POSITION_SLOTS)}
POSITION_PREFIXES = {"f": "front", "b": "back", "s": "side", "t": "top"}
POSITION_LIMITS = {"front": 2, "back": 2, "side": 2, "top": 3}


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
    parser.add_argument("--positions-config", type=Path, default=Path("nine_dshow_camera_grid_positions.json"))
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


def load_position_assignments(path):
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not load {path}: {exc}", flush=True)
        return {}

    raw_assignments = data.get("assignments", data)
    assignments = {}
    for camera_id, position in raw_assignments.items():
        if position in POSITION_LABELS:
            assignments[int(camera_id)] = position
    return assignments


def save_position_assignments(path, assignments):
    data = {
        "positions": [{"key": key, "label": label} for key, label in POSITION_SLOTS],
        "assignments": {str(camera_id): position for camera_id, position in sorted(assignments.items())},
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def arrange_cameras(cameras, assignments):
    return sorted(
        cameras,
        key=lambda camera: (
            POSITION_ORDER.get(assignments.get(camera.camera_id), len(POSITION_ORDER)),
            camera.camera_id,
        ),
    )


def draw_overlay(frame, camera_id, fps, position, tile_number, selected):
    frame = frame.copy()
    label = POSITION_LABELS.get(position, "Unassigned")
    text = f"{tile_number}: cam {camera_id} | {label} | {fps:4.1f} FPS"
    cv2.rectangle(frame, (6, 6), (frame.shape[1] - 6, 38), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    if selected:
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), (0, 255, 255), thickness=4)
    return frame


def make_grid(cameras, assignments, selected_camera_id, cols, display_height, rotation):
    entries = []
    display_cameras = arrange_cameras(cameras, assignments)
    for index, camera in enumerate(display_cameras, start=1):
        frame = camera.get_frame()
        position = assignments.get(camera.camera_id)
        selected = camera.camera_id == selected_camera_id
        if frame is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(frame, f"cam {camera.camera_id} waiting", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            frame = draw_overlay(frame, camera.camera_id, camera.average_fps(), position, index, selected)
        else:
            frame = rotate_frame(frame, rotation // 90)
            frame = resize_to_height(frame, display_height)
            frame = draw_overlay(frame, camera.camera_id, camera.average_fps(), position, index, selected)
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


def assign_position(assignments, camera_id, prefix, number):
    position = f"{prefix}_{number}"
    for assigned_camera_id, assigned_position in list(assignments.items()):
        if assigned_camera_id != camera_id and assigned_position == position:
            del assignments[assigned_camera_id]
    assignments[camera_id] = position
    return position


def print_controls():
    print("controls:", flush=True)
    print("  1-9: select the visible tile/camera", flush=True)
    print("  f/b/s/t then 1-3: assign Front/Back/Side/Top position", flush=True)
    print("  u: unassign selected camera", flush=True)
    print("  r: clear all assignments", flush=True)
    print("  q/Esc: quit", flush=True)


def handle_key(key, display_cameras, selected_camera_id, pending_prefix, assignments, config_path):
    if key in (27, ord("q")):
        return selected_camera_id, pending_prefix, True

    if ord("1") <= key <= ord("9"):
        number = key - ord("0")
        if pending_prefix is not None and selected_camera_id is not None:
            limit = POSITION_LIMITS[pending_prefix]
            if number <= limit:
                position = assign_position(assignments, selected_camera_id, pending_prefix, number)
                save_position_assignments(config_path, assignments)
                print(f"assigned camera {selected_camera_id} to {POSITION_LABELS[position]}", flush=True)
            else:
                print(f"{pending_prefix.title()} only has positions 1-{limit}", flush=True)
            return selected_camera_id, None, False

        if number <= len(display_cameras):
            selected_camera_id = display_cameras[number - 1].camera_id
            print(f"selected camera {selected_camera_id}", flush=True)
        return selected_camera_id, None, False

    char = chr(key).lower() if 0 <= key < 256 else ""
    if char in POSITION_PREFIXES:
        if selected_camera_id is None:
            print("select a camera tile with 1-9 first", flush=True)
            return selected_camera_id, None, False
        pending_prefix = POSITION_PREFIXES[char]
        print(f"assign camera {selected_camera_id} to {pending_prefix.title()} position: press 1-{POSITION_LIMITS[pending_prefix]}", flush=True)
        return selected_camera_id, pending_prefix, False

    if char == "u" and selected_camera_id is not None:
        if selected_camera_id in assignments:
            del assignments[selected_camera_id]
            save_position_assignments(config_path, assignments)
            print(f"unassigned camera {selected_camera_id}", flush=True)
        return selected_camera_id, None, False

    if char == "r":
        assignments.clear()
        save_position_assignments(config_path, assignments)
        print("cleared all camera position assignments", flush=True)
        return selected_camera_id, None, False

    return selected_camera_id, pending_prefix, False


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
    assignments = load_position_assignments(args.positions_config)
    selected_camera_id = None
    pending_prefix = None
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
        print_controls()
        started_at = time.perf_counter()
        while True:
            display_cameras = arrange_cameras(cameras, assignments)
            grid = make_grid(cameras, assignments, selected_camera_id, cols, args.display_height, args.rotation)
            cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                selected_camera_id, pending_prefix, should_quit = handle_key(
                    key,
                    display_cameras,
                    selected_camera_id,
                    pending_prefix,
                    assignments,
                    args.positions_config,
                )
                if should_quit:
                    break
            if args.duration > 0 and time.perf_counter() - started_at >= args.duration:
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
