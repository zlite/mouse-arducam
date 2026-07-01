import argparse
import csv
import json
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index, fit_to_tile, resize_to_height, rotate_frame


WINDOW_NAME = "Ten DirectShow Cameras"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display ten DirectShow USB cameras and record synchronized timestamped MP4 clips on S."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=None, help="Defaults to the first ten USB Camera devices.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--record-fps", type=float, default=30.0, help="Nominal MP4 playback FPS.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to record after S is pressed.")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--display-height", type=int, default=180)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("timer_sync_recordings"))
    parser.add_argument("--startup-timeout", type=float, default=4.0)
    parser.add_argument("--queue-size", type=int, default=600, help="Per-camera pending frame queue before frames are dropped.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not burn capture timestamps into recorded videos.")
    parser.add_argument("--scan", action="store_true")
    return parser.parse_args()


def list_devices():
    return FilterGraph().get_input_devices()


def now_label():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def draw_text_box(frame, text, origin=(12, 34), font_scale=0.72):
    frame = frame.copy()
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(frame, (x - 6, y - text_height - 8), (x + text_width + 8, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_record_overlay(frame, camera_id, fps, recording, remaining):
    status = "REC" if recording else "READY"
    text = f"cam {camera_id} | {fps:4.1f} FPS | {status}"
    if recording:
        text += f" {remaining:4.1f}s"
    frame = draw_text_box(frame, text, (12, 30), 0.62)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 22, 22), 8, (0, 0, 255), -1, cv2.LINE_AA)
    return frame


def draw_timestamp_overlay(frame, camera_id, source_frame_index, capture_perf, session_perf_start, unix_ns):
    elapsed = capture_perf - session_perf_start
    wall_ms = unix_ns // 1_000_000
    text = f"cam {camera_id} frame {source_frame_index} t={elapsed:.6f}s wall_ms={wall_ms}"
    return draw_text_box(frame, text, (14, frame.shape[0] - 18), 0.62)


class RecordingCamera:
    def __init__(self, camera_id, format_index, max_queue):
        self.camera_id = camera_id
        self.format_index = format_index
        self.max_queue = max_queue
        self.graph = FilterGraph()
        self.latest_frame = None
        self.frame_count = 0
        self.started_at = None
        self.running = False
        self.capture_thread = None
        self.writer_thread = None
        self.latest_lock = threading.Lock()
        self.record_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=max_queue)
        self.active_session = None
        self.completed_sessions = []
        self.dropped_frames = 0

    def start(self):
        self.graph.add_video_input_device(self.camera_id)
        self.graph.get_input_device().set_format(self.format_index)
        self.graph.add_sample_grabber(self._on_frame)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.started_at = time.perf_counter()
        self.running = True
        self.capture_thread = threading.Thread(target=self._request_loop, name=f"dshow-timer-capture-{self.camera_id}", daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, name=f"dshow-timer-writer-{self.camera_id}", daemon=True)
        self.capture_thread.start()
        self.writer_thread.start()

    def _on_frame(self, frame):
        capture_perf = time.perf_counter()
        unix_ns = time.time_ns()
        frame_copy = frame.copy()
        self.frame_count += 1
        source_frame_index = self.frame_count
        with self.latest_lock:
            self.latest_frame = frame_copy

        with self.record_lock:
            session = self.active_session
            should_record = session is not None and capture_perf <= session["stop_perf"]
        if should_record:
            record = {
                "session": session,
                "frame": frame_copy.copy(),
                "source_frame_index": source_frame_index,
                "capture_perf": capture_perf,
                "unix_ns": unix_ns,
            }
            try:
                self.frame_queue.put_nowait(record)
            except queue.Full:
                with self.record_lock:
                    self.dropped_frames += 1

    def _request_loop(self):
        while self.running:
            self.graph.grab_frame()
            time.sleep(0.001)

    def _writer_loop(self):
        writers = {}
        csv_handles = {}
        csv_writers = {}
        while self.running or not self.frame_queue.empty():
            try:
                record = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                self._close_finished_writers(writers, csv_handles, csv_writers)
                continue

            session = record["session"]
            session_id = session["session_id"]
            if session_id not in writers:
                height, width = self._record_frame(record).shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(session["video_path"]), fourcc, session["record_fps"], (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open VideoWriter for {session['video_path']}")
                csv_handle = session["csv_path"].open("w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(
                    csv_handle,
                    fieldnames=[
                        "written_frame_index",
                        "source_frame_index",
                        "capture_perf",
                        "elapsed_s",
                        "unix_ns",
                        "unix_ms",
                    ],
                )
                csv_writer.writeheader()
                writers[session_id] = writer
                csv_handles[session_id] = csv_handle
                csv_writers[session_id] = csv_writer

            frame = self._record_frame(record)
            writers[session_id].write(frame)
            session["frames_written"] += 1
            elapsed = record["capture_perf"] - session["start_perf"]
            csv_writers[session_id].writerow(
                {
                    "written_frame_index": session["frames_written"],
                    "source_frame_index": record["source_frame_index"],
                    "capture_perf": f"{record['capture_perf']:.9f}",
                    "elapsed_s": f"{elapsed:.9f}",
                    "unix_ns": record["unix_ns"],
                    "unix_ms": record["unix_ns"] // 1_000_000,
                }
            )
            self.frame_queue.task_done()
            self._close_finished_writers(writers, csv_handles, csv_writers)

        for session_id, writer in list(writers.items()):
            writer.release()
            csv_handles[session_id].close()

    def _record_frame(self, record):
        session = record["session"]
        frame = record["frame"]
        frame = rotate_frame(frame, session["rotation_steps"])
        if session["overlay_timestamps"]:
            frame = draw_timestamp_overlay(
                frame,
                self.camera_id,
                record["source_frame_index"],
                record["capture_perf"],
                session["start_perf"],
                record["unix_ns"],
            )
        return frame

    def _close_finished_writers(self, writers, csv_handles, csv_writers):
        with self.record_lock:
            active_session_ids = {self.active_session["session_id"]} if self.active_session is not None else set()
        for session_id in list(writers.keys()):
            if session_id not in active_session_ids and self.frame_queue.empty():
                writers[session_id].release()
                csv_handles[session_id].close()
                del writers[session_id]
                del csv_handles[session_id]
                del csv_writers[session_id]

    def start_recording(self, session_id, session_dir, duration, record_fps, rotation_steps, overlay_timestamps, start_perf, start_unix_ns):
        video_path = session_dir / f"cam_{self.camera_id}.mp4"
        csv_path = session_dir / f"cam_{self.camera_id}_frames.csv"
        with self.record_lock:
            if self.active_session is not None:
                return False
            self.dropped_frames = 0
            self.active_session = {
                "session_id": session_id,
                "video_path": video_path,
                "csv_path": csv_path,
                "duration": duration,
                "record_fps": record_fps,
                "rotation_steps": rotation_steps,
                "overlay_timestamps": overlay_timestamps,
                "start_perf": start_perf,
                "start_unix_ns": start_unix_ns,
                "stop_perf": start_perf + duration,
                "frames_written": 0,
            }
        return True

    def finish_recording_if_due(self, now_perf):
        with self.record_lock:
            session = self.active_session
            if session is None or now_perf < session["stop_perf"]:
                return None
            session = dict(session)
            session["dropped_frames"] = self.dropped_frames
            self.completed_sessions.append(session)
            self.active_session = None
            return session

    def get_frame(self):
        with self.latest_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        return self.frame_count / max(time.perf_counter() - self.started_at, 1e-6)

    def recording_state(self):
        with self.record_lock:
            session = self.active_session
            if session is None:
                return False, 0.0
            remaining = max(0.0, session["stop_perf"] - time.perf_counter())
            return True, remaining

    def stop(self):
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=3.0)
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()


def make_grid(cameras, cols, display_height, rotation_steps):
    entries = []
    for camera in cameras:
        frame = camera.get_frame()
        recording, remaining = camera.recording_state()
        if frame is None:
            frame = np.zeros((display_height, int(display_height * 16 / 9), 3), dtype=np.uint8)
            cv2.putText(frame, f"cam {camera.camera_id} waiting", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        else:
            frame = rotate_frame(frame, rotation_steps)
            frame = resize_to_height(frame, display_height)
        frame = draw_record_overlay(frame, camera.camera_id, camera.average_fps(), recording, remaining)
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


def start_recording(cameras, args):
    session_id = now_label()
    session_dir = args.output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    start_perf = time.perf_counter()
    start_unix_ns = time.time_ns()
    rotation_steps = args.rotation // 90
    for camera in cameras:
        camera.start_recording(
            session_id,
            session_dir,
            args.duration,
            args.record_fps,
            rotation_steps,
            not args.no_overlay,
            start_perf,
            start_unix_ns,
        )
    summary = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "started_unix_ns": start_unix_ns,
        "started_perf": start_perf,
        "duration_s": args.duration,
        "record_fps": args.record_fps,
        "source": {
            "format": args.format,
            "width": args.width,
            "height": args.height,
            "rotation": args.rotation,
        },
        "cameras": [camera.camera_id for camera in cameras],
    }
    (session_dir / "session_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Recording {len(cameras)} cameras for {args.duration:.1f}s -> {session_dir}", flush=True)
    return session_dir


def print_controls():
    print("controls:", flush=True)
    print("  S: record a synchronized 10 second clip from every camera", flush=True)
    print("  q/Esc: quit", flush=True)


def main():
    args = parse_args()
    if args.scan:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return

    if args.cameras is None:
        args.cameras = [index for index, device in enumerate(list_devices()) if device == "USB Camera"][:10]
    if len(args.cameras) != 10:
        print(f"Warning: using {len(args.cameras)} cameras, not exactly 10.", flush=True)

    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.cameras))))
    cameras = []
    active_session_dir = None
    completed_session_dirs = set()
    try:
        for camera_id in args.cameras:
            print(f"starting camera {camera_id}...", flush=True)
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = RecordingCamera(camera_id, format_index, args.queue_size)
            camera.start()
            cameras.append(camera)
            print(f"started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.get_frame() is not None for camera in cameras):
            time.sleep(0.05)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print_controls()
        while True:
            now_perf = time.perf_counter()
            finished = [camera.finish_recording_if_due(now_perf) for camera in cameras]
            finished = [session for session in finished if session is not None]
            if active_session_dir is not None and len(finished) == len(cameras):
                completed_session_dirs.add(active_session_dir)
                print(f"Recording finished -> {active_session_dir}", flush=True)
                active_session_dir = None

            grid = make_grid(cameras, cols, args.display_height, args.rotation // 90)
            cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("s"), ord("S")):
                if active_session_dir is None:
                    active_session_dir = start_recording(cameras, args)
                else:
                    print("Already recording; wait for the current 10 second capture to finish.", flush=True)
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()
        if active_session_dir is not None and active_session_dir not in completed_session_dirs:
            print(f"Stopped with recording data in progress -> {active_session_dir}", flush=True)


if __name__ == "__main__":
    main()
