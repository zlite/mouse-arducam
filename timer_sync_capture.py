import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index, fit_to_tile, get_formats, resize_to_height, rotate_frame


WINDOW_NAME = "Timer Sync Capture"
DEFAULT_TIMER_CAMERAS = [1, 4, 6, 7, 9]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display ten DirectShow cameras and RAM-capture a short synchronized timer test on r."
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help=f"Defaults to timer-facing cameras: {' '.join(str(camera) for camera in DEFAULT_TIMER_CAMERAS)}.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to hold frames in RAM after r is pressed.")
    parser.add_argument(
        "--review-time",
        type=float,
        default=None,
        help="Elapsed seconds to review after capture. Defaults to the midpoint of the common captured time range.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("timer_sync_captures"))
    parser.add_argument("--image-format", choices=("png", "jpg", "bmp"), default="png")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--cols", type=int, default=0, help="Uniform grid columns. Overrides --row-layout when > 0.")
    parser.add_argument("--row-layout", default="3,3,3,1")
    parser.add_argument("--display-height", type=int, default=180)
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


def session_label():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_row_layout(layout_text, camera_count):
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
    if rows:
        return rows
    cols = max(1, math.ceil(math.sqrt(camera_count)))
    return [cols for _start in range(0, camera_count, cols)]


def draw_overlay(frame, camera_id, fps, state, remaining=0.0):
    frame = frame.copy()
    text = f"cam {camera_id}  {fps:4.1f}"
    if state == "recording":
        text += f"  REC {remaining:3.1f}s"
    elif state == "writing":
        text += "  writing"
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, 180), 22), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if state == "recording":
        cv2.circle(frame, (frame.shape[1] - 16, 16), 6, (0, 0, 255), -1, cv2.LINE_AA)
    return frame


class CaptureSession:
    def __init__(self, session_dir, camera_ids, args):
        self.session_id = session_dir.name
        self.session_dir = session_dir
        self.camera_ids = list(camera_ids)
        self.duration_s = float(args.duration)
        self.start_perf_ns = time.perf_counter_ns()
        self.start_unix_ns = time.time_ns()
        self.stop_perf_ns = self.start_perf_ns + int(self.duration_s * 1_000_000_000)
        self.width = args.width
        self.height = args.height
        self.format = args.format
        self.rotation = args.rotation
        self.image_format = args.image_format
        self.jpg_quality = args.jpg_quality
        self.buffers = {camera_id: [] for camera_id in camera_ids}
        self.lock = threading.Lock()

    def add_copied_frame(self, camera_id, frame_index, frame, capture_perf_ns, unix_ns):
        if capture_perf_ns > self.stop_perf_ns:
            return
        with self.lock:
            self.buffers[camera_id].append(
                {
                    "frame_index": frame_index,
                    "capture_perf_ns": capture_perf_ns,
                    "elapsed_ns": capture_perf_ns - self.start_perf_ns,
                    "unix_ns": unix_ns,
                    "frame": frame,
                }
            )

    def snapshot_buffers(self):
        with self.lock:
            return {camera_id: list(records) for camera_id, records in self.buffers.items()}


class DShowCaptureCamera:
    def __init__(self, camera_id, format_index, session_getter):
        self.camera_id = camera_id
        self.format_index = format_index
        self.session_getter = session_getter
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
        self.thread = threading.Thread(target=self._request_loop, name=f"timer-sync-cam-{self.camera_id}", daemon=True)
        self.thread.start()

    def _on_frame(self, frame):
        capture_perf_ns = time.perf_counter_ns()
        unix_ns = time.time_ns()
        self.frame_count += 1
        frame_index = self.frame_count

        session = self.session_getter()
        should_capture = session is not None and capture_perf_ns <= session.stop_perf_ns
        frame_copy = frame.copy()
        with self.lock:
            self.latest_frame = frame_copy

        if should_capture:
            session.add_copied_frame(self.camera_id, frame_index, frame_copy, capture_perf_ns, unix_ns)

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


def write_session(session, args):
    buffers = session.snapshot_buffers()
    session.session_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "session_id": session.session_id,
        "duration_s": session.duration_s,
        "start_perf_ns": session.start_perf_ns,
        "start_unix_ns": session.start_unix_ns,
        "stop_perf_ns": session.stop_perf_ns,
        "source": {
            "format": session.format,
            "width": session.width,
            "height": session.height,
            "rotation": session.rotation,
        },
        "image_format": session.image_format,
        "cameras": {},
    }

    encode_params = []
    if session.image_format == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(session.jpg_quality)]

    for camera_id, records in buffers.items():
        camera_dir = session.session_dir / f"cam_{camera_id}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        csv_path = session.session_dir / f"cam_{camera_id}_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "written_index",
                    "source_frame_index",
                    "capture_perf_ns",
                    "elapsed_ns",
                    "elapsed_s",
                    "unix_ns",
                    "filename",
                ],
            )
            writer.writeheader()
            for written_index, record in enumerate(records):
                frame = rotate_frame(record["frame"], args.rotation // 90)
                filename = f"cam_{camera_id}_frame_{written_index:05d}.{session.image_format}"
                cv2.imwrite(str(camera_dir / filename), frame, encode_params)
                writer.writerow(
                    {
                        "written_index": written_index,
                        "source_frame_index": record["frame_index"],
                        "capture_perf_ns": record["capture_perf_ns"],
                        "elapsed_ns": record["elapsed_ns"],
                        "elapsed_s": f"{record['elapsed_ns'] / 1_000_000_000:.9f}",
                        "unix_ns": record["unix_ns"],
                        "filename": str(Path(f"cam_{camera_id}") / filename),
                    }
                )

        elapsed_values = [record["elapsed_ns"] / 1_000_000_000 for record in records]
        metadata["cameras"][str(camera_id)] = {
            "frames": len(records),
            "csv": csv_path.name,
            "folder": camera_dir.name,
            "first_elapsed_s": min(elapsed_values) if elapsed_values else None,
            "last_elapsed_s": max(elapsed_values) if elapsed_values else None,
        }
        print(f"wrote cam {camera_id}: {len(records)} frames", flush=True)

    (session.session_dir / "session_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"finished writing session -> {session.session_dir}", flush=True)


def build_review_frames(session, args):
    buffers = session.snapshot_buffers()
    nonempty = {camera_id: records for camera_id, records in buffers.items() if records}
    if not nonempty:
        return None

    first_elapsed = max(records[0]["elapsed_ns"] for records in nonempty.values())
    last_elapsed = min(records[-1]["elapsed_ns"] for records in nonempty.values())
    if args.review_time is not None:
        target_elapsed_ns = int(args.review_time * 1_000_000_000)
    elif first_elapsed <= last_elapsed:
        target_elapsed_ns = (first_elapsed + last_elapsed) // 2
    else:
        target_elapsed_ns = int(session.duration_s * 500_000_000)

    frames = []
    for camera_id in session.camera_ids:
        records = buffers.get(camera_id, [])
        if not records:
            frames.append(
                {
                    "camera_id": camera_id,
                    "frame": None,
                    "elapsed_ns": None,
                    "delta_ns": None,
                    "source_frame_index": None,
                }
            )
            continue
        closest = min(records, key=lambda record: abs(record["elapsed_ns"] - target_elapsed_ns))
        frames.append(
            {
                "camera_id": camera_id,
                "frame": rotate_frame(closest["frame"], args.rotation // 90),
                "elapsed_ns": closest["elapsed_ns"],
                "delta_ns": closest["elapsed_ns"] - target_elapsed_ns,
                "source_frame_index": closest["frame_index"],
            }
        )

    return {
        "session_id": session.session_id,
        "session_dir": str(session.session_dir),
        "target_elapsed_ns": target_elapsed_ns,
        "target_elapsed_s": target_elapsed_ns / 1_000_000_000,
        "frames": frames,
    }


def draw_review_overlay(frame, entry, target_elapsed_s):
    frame = frame.copy()
    if entry["elapsed_ns"] is None:
        text = f"cam {entry['camera_id']}  no frame"
    else:
        elapsed_s = entry["elapsed_ns"] / 1_000_000_000
        delta_ms = entry["delta_ns"] / 1_000_000
        text = f"cam {entry['camera_id']} f{entry['source_frame_index']} t={elapsed_s:.6f}s d={delta_ms:+.2f}ms"
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, 300), 24), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (4, frame.shape[0] - 24), (min(frame.shape[1] - 4, 190), frame.shape[0] - 4), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"target {target_elapsed_s:.6f}s",
        (8, frame.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def make_review_grid(review, row_counts, display_height):
    entries = []
    for entry in review["frames"]:
        if entry["frame"] is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"cam {entry['camera_id']} no frame",
                (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
            )
        else:
            frame = resize_to_height(entry["frame"], display_height)
        entries.append(draw_review_overlay(frame, entry, review["target_elapsed_s"]))

    tile_height = display_height
    tile_width = int(round(display_height * 16 / 9))
    max_cols = max(row_counts)
    rows = []
    start = 0
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


def make_grid(cameras, row_counts, display_height, rotation_steps, state, remaining):
    entries = []
    for camera in cameras:
        frame = camera.get_frame()
        if frame is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(frame, f"cam {camera.camera_id} waiting", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        else:
            frame = rotate_frame(frame, rotation_steps)
            frame = resize_to_height(frame, display_height)
        entries.append(draw_overlay(frame, camera.camera_id, camera.average_fps(), state, remaining))

    tile_height = display_height
    tile_width = int(round(display_height * 16 / 9))
    max_cols = max(row_counts)
    rows = []
    start = 0
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


def select_default_cameras(devices):
    if all(camera_id < len(devices) for camera_id in DEFAULT_TIMER_CAMERAS):
        return list(DEFAULT_TIMER_CAMERAS)
    arducams = [index for index, device in enumerate(devices) if "arducam" in device.lower()]
    exact_usb = [index for index, device in enumerate(devices) if device == "USB Camera"]
    likely_usb = [
        index
        for index, device in enumerate(devices)
        if "usb" in device.lower() or "arducam" in device.lower()
    ]
    return (arducams or exact_usb or likely_usb)[:5]


def main():
    args = parse_args()
    devices = list_devices()
    if args.scan:
        for index, device in enumerate(devices):
            print(f"{index}: {device}", flush=True)
        return

    if args.cameras is None:
        args.cameras = select_default_cameras(devices)
    if args.list_formats:
        print_formats(args.cameras)
        return
    if not args.cameras:
        print("No cameras selected. Run with --scan, then pass --cameras.", flush=True)
        return

    row_counts = make_row_layout(len(args.cameras), args.cols, args.row_layout)
    print("selected devices:", flush=True)
    for camera_id in args.cameras:
        device_name = devices[camera_id] if camera_id < len(devices) else "<missing>"
        print(f"  {camera_id}: {device_name}", flush=True)
    print(f"row layout: {','.join(str(count) for count in row_counts)}", flush=True)

    session_lock = threading.Lock()
    review_lock = threading.Lock()
    active_session = None
    latest_review = None
    writing_threads = []

    def get_active_session():
        with session_lock:
            return active_session

    def write_then_review(session):
        nonlocal latest_review
        write_session(session, args)
        review = build_review_frames(session, args)
        with review_lock:
            latest_review = review
        if review is not None:
            print(
                f"reviewing nearest frames to t={review['target_elapsed_s']:.6f}s "
                f"from {review['session_dir']}",
                flush=True,
            )

    cameras = []
    window_sized = False
    try:
        for camera_id in args.cameras:
            device_name = devices[camera_id] if camera_id < len(devices) else "<missing>"
            print(f"starting camera {camera_id}: {device_name}...", flush=True)
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCaptureCamera(camera_id, format_index, get_active_session)
            camera.start()
            cameras.append(camera)
            print(f"started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.get_frame() is not None for camera in cameras):
            time.sleep(0.05)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        print("controls: r = capture one second to RAM, q/Esc = quit", flush=True)
        while True:
            now_ns = time.perf_counter_ns()
            with session_lock:
                session = active_session
                if session is not None and now_ns >= session.stop_perf_ns:
                    active_session = None
                    session_to_write = session
                else:
                    session_to_write = None

            if session_to_write is not None:
                print(f"capture complete; writing {session_to_write.session_dir}...", flush=True)
                thread = threading.Thread(target=write_then_review, args=(session_to_write,), daemon=True)
                thread.start()
                writing_threads.append(thread)

            with session_lock:
                session = active_session
            if session is not None:
                state = "recording"
                remaining = max(0.0, (session.stop_perf_ns - time.perf_counter_ns()) / 1_000_000_000)
            elif any(thread.is_alive() for thread in writing_threads):
                state = "writing"
                remaining = 0.0
            else:
                state = "ready"
                remaining = 0.0

            with review_lock:
                review = latest_review
            if state == "ready" and review is not None:
                grid = make_review_grid(review, row_counts, args.display_height)
            else:
                grid = make_grid(cameras, row_counts, args.display_height, args.rotation // 90, state, remaining)
            if not window_sized:
                cv2.resizeWindow(WINDOW_NAME, grid.shape[1], grid.shape[0])
                window_sized = True
            cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                with session_lock:
                    can_start = active_session is None
                    writing = any(thread.is_alive() for thread in writing_threads)
                    if can_start and not writing:
                        with review_lock:
                            latest_review = None
                        session_dir = args.output_dir / session_label()
                        active_session = CaptureSession(session_dir, [camera.camera_id for camera in cameras], args)
                        print(f"capturing {args.duration:.3f}s to RAM -> {session_dir}", flush=True)
                    elif active_session is not None:
                        print("already capturing; wait for this capture to finish", flush=True)
                    else:
                        print("still writing previous capture; wait before starting another", flush=True)
    finally:
        with session_lock:
            unfinished_session = active_session
            active_session = None
        if unfinished_session is not None:
            print(f"writing interrupted capture -> {unfinished_session.session_dir}", flush=True)
            write_session(unfinished_session, args)
        for thread in writing_threads:
            thread.join()
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
