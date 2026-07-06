#!/usr/bin/env python3
import argparse
import math
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

cv2 = None
np = None


WINDOW_NAME = "Ten V4L2 USB Cameras"


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
        suffix = f"  {name}" if name else ""
        print(f"{device}{suffix}", flush=True)


def open_capture(device, width, height, fps, fourcc, buffer_size):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
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
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.buffer_size = buffer_size
        self.cap = None
        self.latest_frame = None
        self.latest_shape = None
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
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame
                self.latest_shape = frame.shape
                self.frame_count += 1
                self.frame_times.append(now)

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None, None, self.current_fps()
            return True, self.latest_frame.copy(), self.latest_shape, self.current_fps()

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


def draw_overlay(frame, label, source_shape, fps):
    frame = frame.copy()
    source_height, source_width = source_shape[:2] if source_shape else frame.shape[:2]
    text = f"{label} | {source_width}x{source_height} | {fps:4.1f} FPS"
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


def make_waiting_frame(label, display_height, aspect_ratio):
    width = max(1, int(round(display_height * aspect_ratio)))
    frame = np.zeros((display_height, width, 3), dtype=np.uint8)
    cv2.putText(frame, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "waiting", (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 180, 255), 1, cv2.LINE_AA)
    return frame


def make_grid(cameras, cols, display_height, rotation, no_overlay, source_aspect):
    tiles = []
    for camera in cameras:
        ok, frame, source_shape, fps = camera.read_latest()
        if ok:
            frame = rotate_frame(frame, rotation)
            source_shape = frame.shape
            frame = resize_to_height(frame, display_height)
        else:
            frame = make_waiting_frame(camera.device, display_height, source_aspect)

        if not no_overlay:
            frame = draw_overlay(frame, camera.device, source_shape, fps)
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


def main():
    args = parse_args()
    if args.scan:
        scan_devices()
        return

    require_opencv()
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    if args.devices is None:
        args.devices = auto_select_devices(args.width, args.height, args.fps, args.fourcc, 10)

    if not args.devices:
        print("No cameras selected.", flush=True)
        print("Run: python ten_v4l2_camera_grid.py --scan", flush=True)
        print("Then pass devices explicitly, for example:", flush=True)
        print("  python ten_v4l2_camera_grid.py --devices /dev/video0 /dev/video2 ...", flush=True)
        return

    if len(args.devices) != 10:
        print(f"Warning: displaying {len(args.devices)} cameras, not exactly 10.", flush=True)

    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.devices))))
    source_aspect = args.width / max(args.height, 1)
    cameras = [
        V4L2Camera(device, args.width, args.height, args.fps, args.fourcc)
        for device in args.devices
    ]
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
        if args.display_fps > 0:
            print(f"preview capped at {args.display_fps:g} FPS; capture requested at {args.fps:g} FPS", flush=True)
        else:
            print("preview uncapped", flush=True)
        print("q/Esc: quit", flush=True)
        while True:
            loop_started_at = time.perf_counter()
            grid = make_grid(cameras, cols, args.display_height, args.rotation, args.no_overlay, source_aspect)
            if not window_sized:
                cv2.resizeWindow(WINDOW_NAME, grid.shape[1], grid.shape[0])
                window_sized = True
            cv2.imshow(WINDOW_NAME, grid)
            elapsed = time.perf_counter() - loop_started_at
            wait_ms = 1
            if display_period > 0:
                wait_ms = max(1, int(round((display_period - elapsed) * 1000)))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if stop_at is not None and time.perf_counter() >= stop_at:
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
