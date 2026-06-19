import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

if sys.platform.startswith("win"):
    import msvcrt


POSITION_ARGS = (
    ("left_1", "Left 1", "L1"),
    ("left_2", "Left 2", "L2"),
    ("right_1", "Right 1", "R1"),
    ("right_2", "Right 2", "R2"),
)
GRID_POSITION_ORDER = ("left_1", "right_1", "left_2", "right_2")
POSITION_LABELS = {key: label for key, label, _short_label in POSITION_ARGS}
CONTROL_BAR_HEIGHT = 58
CONTROL_BUTTON_HEIGHT = 30
CONTROL_MARGIN = 8
CONTROL_GAP = 6
WINDOW_NAME = "Arducam Grid Viewer"
CONFIG_PATH = Path(__file__).with_name("dual_arducam_viewer_config.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display high-resolution Arducam UVC cameras using exact DirectShow format selection."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4], help="DirectShow camera indices.")
    parser.add_argument("--left", type=int, default=None, help="Legacy two-camera left index.")
    parser.add_argument("--right", type=int, default=None, help="Legacy two-camera right index.")
    parser.add_argument("--left-1", dest="left_1", type=int, default=None, help="Initial camera index for left 1.")
    parser.add_argument("--left-2", dest="left_2", type=int, default=None, help="Initial camera index for left 2.")
    parser.add_argument("--right-1", dest="right_1", type=int, default=None, help="Initial camera index for right 1.")
    parser.add_argument("--right-2", dest="right_2", type=int, default=None, help="Initial camera index for right 2.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=800, help="Requested camera height.")
    parser.add_argument("--format", default="MJPG", help="Requested DirectShow media subtype, such as MJPG or YUY2.")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180, help="Clockwise rotation applied to all camera frames.")
    parser.add_argument("--cols", type=int, default=2, help="Grid columns.")
    parser.add_argument("--display-height", type=int, default=400, help="Displayed height for each camera tile.")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many seconds. 0 runs until q/Esc.")
    parser.add_argument("--frames", type=int, default=0, help="Stop after this many displayed frames. 0 runs until q/Esc.")
    parser.add_argument("--no-display", action="store_true", help="Capture frames without opening a window.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Write one grid frame to this image path.")
    parser.add_argument("--scan", action="store_true", help="List DirectShow camera devices and exit.")
    parser.add_argument("--list-formats", action="store_true", help="List advertised DirectShow formats and exit.")
    parser.add_argument("--startup-timeout", type=float, default=3.0, help="Seconds to wait for first frames.")
    parser.add_argument("--fast", action="store_true", help="Use a lower-bandwidth 320x240 display/capture preset.")
    return parser.parse_args()


class SingleInstanceLock:
    def __init__(self, path):
        self.path = path
        self.file = None

    def acquire(self):
        self.file = open(self.path, "a+", encoding="utf-8")
        self.file.seek(0)
        try:
            if sys.platform.startswith("win"):
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                return True
        except OSError:
            return False

        self.file.seek(0)
        self.file.truncate()
        self.file.write(str(os.getpid()))
        self.file.flush()
        return True

    def release(self):
        if not self.file:
            return
        try:
            self.file.seek(0)
            if sys.platform.startswith("win"):
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.file.close()


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
        self.lock = threading.Lock()

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
        with self.lock:
            self.latest_frame = frame.copy()
            self.frame_count += 1
            self.last_frame_at = time.perf_counter()

    def _request_loop(self):
        while self.running:
            self.graph.grab_frame()
            time.sleep(0.001)

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None, self.average_fps()
            return True, self.latest_frame.copy(), self.average_fps()

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return self.frame_count / elapsed

    def has_frame(self):
        with self.lock:
            return self.latest_frame is not None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()


def initial_position_by_camera(args):
    position_by_camera = {}
    for position_key, _label, _short_label in POSITION_ARGS:
        index = getattr(args, position_key)
        if index is None:
            continue
        if index in position_by_camera:
            raise RuntimeError(f"Camera index {index} is assigned to more than one position.")
        position_by_camera[index] = position_key
    return position_by_camera


def label_for_camera(index, position_by_camera):
    position_key = position_by_camera.get(index)
    return POSITION_LABELS[position_key] if position_key else f"Camera {index}"


def camera_selector_order(cameras):
    return sorted(cameras, key=lambda camera: camera.device_index)


def grid_camera_order(cameras, position_by_camera):
    by_index = {camera.device_index: camera for camera in cameras}
    ordered = []
    used = set()
    for position_key in GRID_POSITION_ORDER:
        for index, assigned_position in position_by_camera.items():
            if assigned_position == position_key and index in by_index:
                ordered.append(by_index[index])
                used.add(index)
                break
    for camera in camera_selector_order(cameras):
        if camera.device_index not in used:
            ordered.append(camera)
    return ordered


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


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


def draw_overlay(frame, label, source_shape, fps):
    frame = frame.copy()
    source_height, source_width = source_shape[:2]
    text = f"{label} | {source_width}x{source_height} | {fps:4.1f} FPS"
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    box_width = min(frame.shape[1] - 8, text_width + 18)
    box_height = text_height + 14
    cv2.rectangle(frame, (6, 6), (6 + box_width, 6 + box_height), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (14, 6 + text_height + 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def make_error_frame(label, message, display_height):
    frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
    cv2.putText(frame, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, message, (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 180, 255), 1, cv2.LINE_AA)
    return frame


def draw_control_button(frame, x, y, width, text, active=False):
    x2 = min(frame.shape[1] - CONTROL_MARGIN, x + width)
    y2 = min(frame.shape[0] - CONTROL_MARGIN, y + CONTROL_BUTTON_HEIGHT)
    fill = (50, 105, 55) if active else (35, 35, 35)
    border = (230, 255, 230) if active else (220, 220, 220)
    cv2.rectangle(frame, (x, y), (x2, y2), fill, thickness=-1)
    cv2.rectangle(frame, (x, y), (x2, y2), border, thickness=1)
    cv2.putText(frame, text, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return (x, y, x2, y2)


def make_control_bar(width, cameras, position_by_camera, selected_camera_index):
    bar = np.zeros((CONTROL_BAR_HEIGHT, width, 3), dtype=np.uint8)
    rects = []
    x = CONTROL_MARGIN
    y = CONTROL_MARGIN
    cv2.putText(bar, "Select camera:", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    x += 112
    for camera in cameras:
        camera_index = camera.device_index
        rect = draw_control_button(bar, x, y, 64, f"Cam {camera_index}", active=camera_index == selected_camera_index)
        rects.append({"action": "select_camera", "camera_index": camera_index, "rect": rect})
        x += 64 + CONTROL_GAP
    x += CONTROL_GAP * 2
    cv2.putText(bar, "Assign:", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    x += 64
    selected_position = position_by_camera.get(selected_camera_index)
    for position_key, _label, short_label in POSITION_ARGS:
        rect = draw_control_button(bar, x, y, 46, short_label, active=selected_position == position_key)
        rects.append({"action": "assign_position", "position_key": position_key, "rect": rect})
        x += 46 + CONTROL_GAP

    x += CONTROL_GAP * 2
    rect = draw_control_button(bar, x, y, 58, "Save")
    rects.append({"action": "save_config", "rect": rect})
    return bar, rects


def make_grid(cameras, cols, display_height, rotation_degrees, position_by_camera):
    tiles = []
    for camera in grid_camera_order(cameras, position_by_camera):
        ok, frame, fps = camera.read_latest()
        label = label_for_camera(camera.device_index, position_by_camera)
        if ok:
            frame = rotate_frame(frame, rotation_degrees)
            source_shape = frame.shape
            frame = resize_to_height(frame, display_height)
            frame = draw_overlay(frame, f"{label} (cam {camera.device_index})", source_shape, fps)
        else:
            frame = make_error_frame(f"{label} (cam {camera.device_index})", "Waiting for frame", display_height)
        tiles.append(frame)

    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), cols):
        row_tiles = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + cols]]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def make_display_frame(cameras, cols, display_height, rotation_degrees, position_by_camera, selected_camera_index):
    grid = make_grid(cameras, cols, display_height, rotation_degrees, position_by_camera)
    selector_cameras = camera_selector_order(cameras)
    control_bar, button_rects = make_control_bar(grid.shape[1], selector_cameras, position_by_camera, selected_camera_index)
    return np.vstack([control_bar, grid]), button_rects


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


def print_position_assignments(position_by_camera):
    assigned = []
    for position_key, label, _short_label in POSITION_ARGS:
        for index, assigned_position in position_by_camera.items():
            if assigned_position == position_key:
                assigned.append(f"{label}=cam {index}")
                break
    print("Camera positions: " + (", ".join(assigned) if assigned else "none assigned"), flush=True)


def save_configuration(args, position_by_camera):
    positions = {}
    for position_key, label, _short_label in POSITION_ARGS:
        for index, assigned_position in position_by_camera.items():
            if assigned_position == position_key:
                positions[position_key] = index
                break

    config = {
        "cameras": list(args.cameras),
        "positions": positions,
        "width": args.width,
        "height": args.height,
        "format": args.format,
        "rotation": args.rotation,
        "cols": args.cols,
        "display_height": args.display_height,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved configuration: {CONFIG_PATH}", flush=True)


def main():
    args = parse_args()
    if args.fast:
        args.width = 320
        args.height = 240
        args.display_height = min(args.display_height, 240)
    if args.left is not None or args.right is not None:
        if args.left is None or args.right is None:
            raise RuntimeError("Use both --left and --right together, or use --cameras.")
        args.cameras = [args.left, args.right]

    if args.scan:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return
    if args.list_formats:
        print_formats(args.cameras)
        return

    lock = SingleInstanceLock(Path(__file__).with_suffix(".lock"))
    if not lock.acquire():
        raise RuntimeError("Another dual_arducam_viewer.py instance is already running. Close it first.")

    cameras = []
    position_by_camera = initial_position_by_camera(args)
    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.cameras))))
    selected_camera = {"index": args.cameras[0] if args.cameras else None}
    button_rects = []

    def handle_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in button_rects:
            x1, y1, x2, y2 = button["rect"]
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if button["action"] == "select_camera":
                selected_camera["index"] = button["camera_index"]
                print(f"Selected camera {selected_camera['index']}", flush=True)
                break
            if button["action"] == "assign_position" and selected_camera["index"] is not None:
                position_key = button["position_key"]
                for assigned_index, assigned_position in list(position_by_camera.items()):
                    if assigned_position == position_key:
                        del position_by_camera[assigned_index]
                position_by_camera[selected_camera["index"]] = position_key
                print_position_assignments(position_by_camera)
                break
            if button["action"] == "save_config":
                save_configuration(args, position_by_camera)
                break

    try:
        for device_index in args.cameras:
            format_index = find_format_index(device_index, args.format, args.width, args.height)
            print(f"Camera {device_index}: using {args.format.upper()} {args.width}x{args.height}, format index {format_index}", flush=True)
            camera = DShowCamera(device_index, format_index)
            camera.start()
            cameras.append(camera)

        startup_deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < startup_deadline and not all(camera.has_frame() for camera in cameras):
            time.sleep(0.02)
        for camera in cameras:
            if not camera.has_frame():
                print(f"Warning: camera {camera.device_index} has not delivered an initial frame yet.", flush=True)

        print("Press q or Esc to quit. Use the top bar: select a camera, then click L1/L2/R1/R2.", flush=True)
        print(f"Rotation: {args.rotation} degrees clockwise", flush=True)
        print_position_assignments(position_by_camera)

        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

        stop_at = time.perf_counter() + args.duration if args.duration else None
        while True:
            if args.snapshot and not all(camera.has_frame() for camera in cameras):
                time.sleep(0.02)
                continue

            combined, button_rects = make_display_frame(
                cameras,
                cols,
                args.display_height,
                args.rotation,
                position_by_camera,
                selected_camera["index"],
            )

            if args.snapshot:
                cv2.imwrite(str(args.snapshot), combined)
                print(f"Wrote snapshot: {args.snapshot}", flush=True)
                if args.no_display or args.frames == 1:
                    break
                args.snapshot = None

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.frames:
                args.frames -= 1
                if args.frames <= 0:
                    break
            if stop_at is not None and time.perf_counter() >= stop_at:
                break

        for camera in cameras:
            print(f"Camera {camera.device_index} average FPS: {camera.average_fps():.1f}", flush=True)
    finally:
        for camera in cameras:
            camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        lock.release()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print()
        print("Tip: run `python dual_arducam_viewer.py --scan` to see DirectShow devices.", flush=True)
        if sys.stdin.isatty():
            try:
                input("Press Enter to close...")
            except EOFError:
                pass
        sys.exit(1)
