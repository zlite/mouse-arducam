#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

cv2 = None
np = None


WINDOW_NAME = "Ten V4L2 USB Cameras"
CONFIG_PATH = Path(__file__).with_name("v4l2_camera_positions.json")
POSITION_GROUPS = {
    "front": 2,
    "back": 2,
    "right": 2,
    "left": 2,
    "top": 4,
}
POSITION_KEYS = [
    *(f"front_{index}" for index in range(1, 3)),
    *(f"back_{index}" for index in range(1, 3)),
    *(f"right_{index}" for index in range(1, 3)),
    *(f"left_{index}" for index in range(1, 3)),
    *(f"top_{index}" for index in range(1, 5)),
]
POSITION_ORDER = {position: index for index, position in enumerate(POSITION_KEYS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display ten Ubuntu/V4L2 USB cameras in a low-latency grid."
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Camera devices. Defaults to the first ten /dev/video* devices that open.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=800, help="Requested capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS.")
    parser.add_argument(
        "--display-fps",
        type=float,
        default=20.0,
        help="Preview grid refresh FPS. Capture still runs at --fps. Use 0 for uncapped.",
    )
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="Requested V4L2 pixel format. MJPG is usually best for ten USB cameras.",
    )
    parser.add_argument("--cols", type=int, default=5, help="Grid columns.")
    parser.add_argument("--display-height", type=int, default=200, help="Displayed height per tile.")
    parser.add_argument(
        "--camera-count",
        type=int,
        default=10,
        help="Number of cameras to auto-select when --devices is omitted.",
    )
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after seconds. 0 runs until q/Esc.")
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV worker threads. 1 often lowers CPU contention with many cameras. Use 0 for OpenCV default.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Hide camera/FPS overlays.")
    parser.add_argument(
        "--positions-config",
        type=Path,
        default=CONFIG_PATH,
        help="JSON file for persistent camera position assignments.",
    )
    parser.add_argument("--scan", action="store_true", help="List V4L2 video devices and exit.")
    return parser.parse_args()


def natural_video_key(path):
    stem = Path(path).name.replace("video", "")
    return int(stem) if stem.isdigit() else 10_000


def video_devices():
    return sorted((str(path) for path in Path("/dev").glob("video*")), key=natural_video_key)


def device_name(device):
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Card type"):
            return stripped.split(":", 1)[1].strip()
    return ""


def scan_devices():
    for device in video_devices():
        name = device_name(device)
        stable_id = stable_device_id(device)
        suffix = f"  {name}" if name else ""
        if stable_id != device:
            suffix += f"  [{stable_id}]"
        print(f"{device}{suffix}", flush=True)


def symlink_id_for(device, directory):
    directory = Path(directory)
    if not directory.exists():
        return None
    target_name = Path(device).name
    matches = []
    for link in directory.iterdir():
        if not link.is_symlink():
            continue
        try:
            if link.resolve().name == target_name:
                matches.append(str(link))
        except OSError:
            continue
    if not matches:
        return None

    index0_matches = [match for match in matches if "video-index0" in Path(match).name]
    candidates = index0_matches or matches
    non_usbv2 = [match for match in candidates if "-usbv2-" not in Path(match).name]
    return sorted(non_usbv2 or candidates)[0]


def stable_device_id(device):
    by_path = symlink_id_for(device, "/dev/v4l/by-path")
    if by_path is not None:
        return by_path
    by_id = symlink_id_for(device, "/dev/v4l/by-id")
    if by_id is not None:
        return by_id
    return device


def format_position(position):
    if not position:
        return "unassigned"
    group, number = position.split("_", 1)
    return f"{group} {number}"


def load_position_assignments(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not load {path}: {exc}", flush=True)
        return {}
    assignments = data.get("assignments", data)
    return {
        str(stable_id): position
        for stable_id, position in assignments.items()
        if position in POSITION_ORDER
    }


def save_position_assignments(path, assignments):
    data = {
        "positions": POSITION_KEYS,
        "assignments": dict(sorted(assignments.items())),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"saved camera positions to {path}", flush=True)


def assign_position(assignments, stable_id, position):
    for other_stable_id, other_position in list(assignments.items()):
        if other_stable_id != stable_id and other_position == position:
            del assignments[other_stable_id]
    assignments[stable_id] = position


def arranged_cameras(cameras, assignments):
    return sorted(
        cameras,
        key=lambda camera: (
            POSITION_ORDER.get(assignments.get(camera.stable_id), len(POSITION_ORDER)),
            camera.stable_id,
            camera.device,
        ),
    )


def open_capture(device, width, height, fps, fourcc, buffer_size):
    capture_device = device
    device_path = Path(device)
    try:
        resolved_name = device_path.resolve().name
    except OSError:
        resolved_name = device_path.name
    if resolved_name.startswith("video") and resolved_name[5:].isdigit():
        capture_device = int(resolved_name[5:])

    cap = cv2.VideoCapture(capture_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def require_opencv():
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "This viewer needs OpenCV and NumPy. On Ubuntu, install them with: "
            "sudo apt install python3-opencv python3-numpy"
        ) from exc
    cv2 = cv2_module
    np = np_module


def auto_select_devices(width, height, fps, fourcc, count):
    selected = []
    for device in video_devices():
        cap = open_capture(device, width, height, fps, fourcc, 1)
        if cap is None:
            continue
        ok, _frame = cap.read()
        cap.release()
        if ok:
            selected.append(device)
            print(f"auto-selected {device}", flush=True)
        if len(selected) == count:
            break
    return selected


class V4L2Camera:
    def __init__(self, device, width, height, fps, fourcc, buffer_size=1):
        self.device = device
        self.stable_id = stable_device_id(device)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.buffer_size = buffer_size
        self.cap = None
        self.latest_frame = None
        self.latest_shape = None
        self.latest_frame_index = None
        self.latest_perf_time = None
        self.latest_unix_time = None
        self.frame_count = 0
        self.started_at = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame_times = deque(maxlen=90)

    def start(self):
        self.cap = open_capture(self.device, self.width, self.height, self.fps, self.fourcc, self.buffer_size)
        if self.cap is None:
            raise RuntimeError(f"Could not open {self.device}")

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(
            f"{self.device}: requested {self.fourcc.upper()} {self.width}x{self.height}@{self.fps:g}, "
            f"opened {actual_width}x{actual_height}@{actual_fps:.1f}",
            flush=True,
        )

        self.started_at = time.perf_counter()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, name=f"v4l2-grid-{self.device}", daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
            except cv2.error as exc:
                print(f"{self.device}: skipped bad frame: {exc}", flush=True)
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            now_unix = time.time()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame
                self.latest_shape = frame.shape
                self.latest_frame_index = self.frame_count
                self.latest_perf_time = now
                self.latest_unix_time = now_unix
                self.frame_count += 1
                self.frame_times.append(now)

    def read_latest(self, copy_frame=True):
        with self.lock:
            if self.latest_frame is None:
                return False, None, None, self.current_fps()
            frame = self.latest_frame.copy() if copy_frame else self.latest_frame
            return True, frame, self.latest_shape, self.current_fps()

    def read_latest_packet(self, copy_frame=True):
        with self.lock:
            if self.latest_frame is None:
                return False, None, None, self.current_fps(), None, None, None
            frame = self.latest_frame.copy() if copy_frame else self.latest_frame
            return (
                True,
                frame,
                self.latest_shape,
                self.current_fps(),
                self.latest_frame_index,
                self.latest_perf_time,
                self.latest_unix_time,
            )

    def current_fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / max(elapsed, 1e-9)

    def has_frame(self):
        with self.lock:
            return self.latest_frame is not None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


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
    target_width = max(1, int(round(width * target_height / height)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def fit_to_tile(frame, tile_width, tile_height):
    height, width = frame.shape[:2]
    scale = min(tile_width / width, tile_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    x = (tile_width - resized_width) // 2
    y = (tile_height - resized_height) // 2
    tile[y : y + resized_height, x : x + resized_width] = resized
    return tile


def draw_overlay(frame, label, source_shape, fps, position=None, tile_number=None):
    source_height, source_width = source_shape[:2] if source_shape else frame.shape[:2]
    tile_prefix = f"{tile_number}: " if tile_number is not None else ""
    text = f"{tile_prefix}{label} | {format_position(position)} | {source_width}x{source_height} | {fps:4.1f} FPS"
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    box_width = min(frame.shape[1] - 8, text_width + 18)
    box_height = text_height + baseline + 12
    cv2.rectangle(frame, (6, 6), (6 + box_width, 6 + box_height), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (14, 6 + text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def grid_shape(item_count, cols):
    rows = max(1, math.ceil(item_count / cols))
    return rows, cols


def tile_width_for(display_height, source_aspect, rotation):
    aspect = source_aspect
    if rotation in (90, 270):
        aspect = 1.0 / max(aspect, 1e-9)
    return max(1, int(round(display_height * aspect)))


def paste_letterboxed(canvas, frame, x, y, tile_width, tile_height):
    source_height, source_width = frame.shape[:2]
    scale = min(tile_width / source_width, tile_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    x_offset = x + (tile_width - resized_width) // 2
    y_offset = y + (tile_height - resized_height) // 2
    target = canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width]
    cv2.resize(frame, (resized_width, resized_height), dst=target, interpolation=cv2.INTER_AREA)
    return target


def make_waiting_frame(label, display_height, aspect_ratio):
    width = max(1, int(round(display_height * aspect_ratio)))
    frame = np.zeros((display_height, width, 3), dtype=np.uint8)
    cv2.putText(frame, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "waiting", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 180, 255), 1, cv2.LINE_AA)
    return frame


def draw_status_bar(canvas, selected_camera, selected_position, pending_group):
    bar_height = 28
    status = "selected: "
    if selected_camera is None:
        status += "none"
    else:
        status += f"{selected_camera.device} [{format_position(selected_position)}]"
    if pending_group is not None:
        status += f" | assign {pending_group}: press slot number"
    else:
        status += " | click tile or press 1-9/0/-/=; f/b/r/l/t then slot"

    bar = np.zeros((bar_height, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def make_grid(
    cameras,
    cols,
    display_height,
    rotation,
    no_overlay,
    source_aspect,
    assignments=None,
    selected_stable_id=None,
    pending_group=None,
    return_layout=False,
):
    assignments = assignments or {}
    display_cameras = arranged_cameras(cameras, assignments)
    tile_height = display_height
    tile_width = tile_width_for(display_height, source_aspect, rotation)
    rows, cols = grid_shape(len(display_cameras), cols)
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)
    tile_rects = []
    selected_camera = None
    selected_position = None

    for index, camera in enumerate(display_cameras):
        row = index // cols
        col = index % cols
        x = col * tile_width
        y = row * tile_height
        tile_rects.append((x, y, x + tile_width, y + tile_height, camera))
        ok, frame, source_shape, fps = camera.read_latest(copy_frame=False)
        position = assignments.get(camera.stable_id)
        if ok:
            frame = rotate_frame(frame, rotation)
            displayed = paste_letterboxed(canvas, frame, x, y, tile_width, tile_height)
            if not no_overlay:
                draw_overlay(displayed, camera.device, source_shape, fps, position, index + 1)
        else:
            frame = make_waiting_frame(camera.device, display_height, source_aspect)
            canvas[y : y + tile_height, x : x + tile_width] = fit_to_tile(frame, tile_width, tile_height)
            if not no_overlay:
                draw_overlay(canvas[y : y + tile_height, x : x + tile_width], camera.device, None, fps, position, index + 1)
        if selected_stable_id == camera.stable_id:
            selected_camera = camera
            selected_position = position
            cv2.rectangle(canvas, (x + 2, y + 2), (x + tile_width - 3, y + tile_height - 3), (0, 255, 255), 3)
    if not no_overlay:
        canvas = draw_status_bar(canvas, selected_camera, selected_position, pending_group)
    if return_layout:
        return canvas, tile_rects, display_cameras
    return canvas


def map_window_to_image_point(window_name, x, y, image_width, image_height):
    try:
        image_x, image_y, display_width, display_height = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return x, y
    if display_width <= 0 or display_height <= 0:
        return x, y
    mapped_x = int(round((x - image_x) * image_width / display_width))
    mapped_y = int(round((y - image_y) * image_height / display_height))
    return mapped_x, mapped_y


def print_position_controls():
    print("position controls:", flush=True)
    print("  click a tile or press 1-9/0/- to select a camera", flush=True)
    print("  f/b/r/l/t then 1-4 assigns front/back/right/left/top position", flush=True)
    print("  u unassigns selected camera; c clears all assignments", flush=True)
    print("  q/Esc quits", flush=True)


def handle_position_key(key, display_cameras, selected_stable_id, pending_group, assignments, config_path):
    if key in (27, ord("q")):
        return selected_stable_id, pending_group, True
    if key in (ord("f"), ord("b"), ord("r"), ord("l"), ord("t")):
        pending_group = {
            ord("f"): "front",
            ord("b"): "back",
            ord("r"): "right",
            ord("l"): "left",
            ord("t"): "top",
        }[key]
        print(f"pending assignment group: {pending_group}", flush=True)
        return selected_stable_id, pending_group, False
    if key == ord("u") and selected_stable_id is not None:
        assignments.pop(selected_stable_id, None)
        save_position_assignments(config_path, assignments)
        return selected_stable_id, pending_group, False
    if key == ord("c"):
        assignments.clear()
        save_position_assignments(config_path, assignments)
        return selected_stable_id, pending_group, False

    tile_keys = "1234567890-="
    if key in (ord(char) for char in tile_keys):
        tile_number = tile_keys.index(chr(key)) + 1
        if pending_group is not None and selected_stable_id is not None:
            limit = POSITION_GROUPS[pending_group]
            if tile_number <= limit:
                position = f"{pending_group}_{tile_number}"
                assign_position(assignments, selected_stable_id, position)
                save_position_assignments(config_path, assignments)
                print(f"assigned selected camera to {format_position(position)}", flush=True)
                pending_group = None
            return selected_stable_id, pending_group, False
        if tile_number <= len(display_cameras):
            selected_camera = display_cameras[tile_number - 1]
            selected_stable_id = selected_camera.stable_id
            print(f"selected tile {tile_number}: {selected_camera.device} [{selected_camera.stable_id}]", flush=True)
    return selected_stable_id, pending_group, False


def main():
    args = parse_args()
    if args.scan:
        scan_devices()
        return

    require_opencv()
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    if args.devices is None:
        args.devices = auto_select_devices(args.width, args.height, args.fps, args.fourcc, args.camera_count)

    if not args.devices:
        print("No cameras selected.", flush=True)
        print("Run: python ten_v4l2_camera_grid.py --scan", flush=True)
        print("Then pass devices explicitly, for example:", flush=True)
        print("  python ten_v4l2_camera_grid.py --devices /dev/video0 /dev/video2 ...", flush=True)
        return

    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.devices))))
    source_aspect = args.width / max(args.height, 1)
    cameras = [
        V4L2Camera(device, args.width, args.height, args.fps, args.fourcc)
        for device in args.devices
    ]
    assignments = load_position_assignments(args.positions_config)
    selected_stable_id = cameras[0].stable_id if cameras else None
    pending_group = None
    tile_rects = []
    display_cameras = cameras
    window_sized = False
    stop_at = time.perf_counter() + args.duration if args.duration else None
    display_period = 1.0 / args.display_fps if args.display_fps > 0 else 0.0

    try:
        for camera in cameras:
            camera.start()

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.has_frame() for camera in cameras):
            time.sleep(0.02)

        for camera in cameras:
            if not camera.has_frame():
                print(f"Warning: {camera.device} has not delivered an initial frame yet.", flush=True)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        displayed_image_shape = [0, 0]

        def handle_mouse(event, x, y, _flags, _userdata):
            nonlocal selected_stable_id
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            candidate_points = [(x, y)]
            image_height, image_width = displayed_image_shape
            if image_width > 0 and image_height > 0:
                candidate_points.append(map_window_to_image_point(WINDOW_NAME, x, y, image_width, image_height))
            for point_x, point_y in candidate_points:
                for x1, y1, x2, y2, camera in tile_rects:
                    if x1 <= point_x <= x2 and y1 <= point_y <= y2:
                        selected_stable_id = camera.stable_id
                        print(f"selected {camera.device} [{camera.stable_id}]", flush=True)
                        return
            print(f"click missed tile at ({x}, {y})", flush=True)

        cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
        if args.display_fps > 0:
            print(f"preview capped at {args.display_fps:g} FPS; capture requested at {args.fps:g} FPS", flush=True)
        else:
            print("preview uncapped", flush=True)
        print_position_controls()
        while True:
            loop_started_at = time.perf_counter()
            grid, tile_rects, display_cameras = make_grid(
                cameras,
                cols,
                args.display_height,
                args.rotation,
                args.no_overlay,
                source_aspect,
                assignments,
                selected_stable_id,
                pending_group,
                return_layout=True,
            )
            displayed_image_shape[:] = [grid.shape[0], grid.shape[1]]
            if not window_sized:
                cv2.resizeWindow(WINDOW_NAME, grid.shape[1], grid.shape[0])
                window_sized = True
            cv2.imshow(WINDOW_NAME, grid)
            elapsed = time.perf_counter() - loop_started_at
            wait_ms = 1
            if display_period > 0:
                wait_ms = max(1, int(round((display_period - elapsed) * 1000)))
            key = cv2.waitKey(wait_ms) & 0xFF
            selected_stable_id, pending_group, should_quit = handle_position_key(
                key,
                display_cameras,
                selected_stable_id,
                pending_group,
                assignments,
                args.positions_config,
            )
            if should_quit:
                break
            if stop_at is not None and time.perf_counter() >= stop_at:
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
