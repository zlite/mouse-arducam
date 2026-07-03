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


WINDOW_NAME = "Caliscope Extrinsic Recorder"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record synchronized-ish DirectShow camera videos in Caliscope calibration/extrinsic layout."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=None, help="Defaults to DirectShow devices named USB Camera.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--record-fps", type=float, default=30.0, help="Nominal FPS written into MP4 headers.")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to record.")
    parser.add_argument("--workspace", type=Path, default=Path("caliscope_workspace"))
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--display-height", type=int, default=220)
    parser.add_argument("--preview-rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--queue-size", type=int, default=300)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--scan", action="store_true")
    return parser.parse_args()


def list_devices():
    return FilterGraph().get_input_devices()


def default_usb_cameras():
    return [index for index, name in enumerate(list_devices()) if name == "USB Camera"]


def draw_overlay(frame, camera_id, fps, state, remaining=None):
    frame = np.ascontiguousarray(frame.copy())
    text = f"cam {camera_id} | {fps:4.1f} FPS | {state}"
    if remaining is not None:
        text += f" {remaining:4.1f}s"
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 360), 38), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if state == "REC":
        cv2.circle(frame, (frame.shape[1] - 18, 18), 7, (0, 0, 255), -1, cv2.LINE_AA)
    return frame


def make_grid(cameras, cols, display_height, rotation_steps, state, remaining):
    tiles = []
    for camera in cameras:
        frame = camera.get_frame()
        if frame is None:
            continue
        frame = rotate_frame(frame, rotation_steps)
        frame = resize_to_height(frame, display_height)
        tiles.append(draw_overlay(frame, camera.camera_id, camera.average_fps(), state, remaining))
    if not tiles:
        return None
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    cols = max(1, cols)
    for start in range(0, len(tiles), cols):
        row = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + cols]]
        while len(row) < cols:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


class CaliscopeCamera:
    def __init__(self, camera_id, format_index, frame_queue_size):
        self.camera_id = camera_id
        self.format_index = format_index
        self.graph = FilterGraph()
        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.latest_frame = None
        self.latest_lock = threading.Lock()
        self.frame_count = 0
        self.started_at = None
        self.running = False
        self.recording = False
        self.capture_thread = None
        self.writer_thread = None
        self.writer = None
        self.timestamps_writer = None
        self.timestamps_lock = None
        self.diagnostics_writer = None
        self.diagnostics_handle = None
        self.frames_written = 0
        self.frames_dropped = 0

    def start(self):
        self.graph.add_video_input_device(self.camera_id)
        self.graph.get_input_device().set_format(self.format_index)
        self.graph.add_sample_grabber(self._on_frame)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.started_at = time.perf_counter()
        self.running = True
        self.capture_thread = threading.Thread(target=self._request_loop, name=f"caliscope-capture-{self.camera_id}", daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, name=f"caliscope-writer-{self.camera_id}", daemon=True)
        self.capture_thread.start()
        self.writer_thread.start()

    def configure_recording(self, extrinsic_dir, record_fps, frame_size, timestamps_writer, timestamps_lock):
        video_path = extrinsic_dir / f"cam_{self.camera_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(video_path), fourcc, record_fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {video_path}")
        self.timestamps_writer = timestamps_writer
        self.timestamps_lock = timestamps_lock
        self.diagnostics_handle = (extrinsic_dir / f"cam_{self.camera_id}_frames.csv").open("w", newline="", encoding="utf-8")
        self.diagnostics_writer = csv.DictWriter(
            self.diagnostics_handle,
            fieldnames=["written_frame_index", "source_frame_index", "frame_time", "unix_ns", "queue_depth"],
        )
        self.diagnostics_writer.writeheader()

    def begin_recording(self):
        self.frames_written = 0
        self.frames_dropped = 0
        self.recording = True

    def stop_recording(self):
        self.recording = False
        self.frame_queue.join()

    def _on_frame(self, frame):
        frame_time = time.perf_counter()
        unix_ns = time.time_ns()
        self.frame_count += 1
        source_frame_index = self.frame_count
        frame_copy = frame.copy()
        with self.latest_lock:
            self.latest_frame = frame_copy
        if self.recording:
            item = {
                "frame": frame_copy,
                "frame_time": frame_time,
                "unix_ns": unix_ns,
                "source_frame_index": source_frame_index,
            }
            try:
                self.frame_queue.put_nowait(item)
            except queue.Full:
                self.frames_dropped += 1

    def _request_loop(self):
        while self.running:
            self.graph.grab_frame()
            time.sleep(0.001)

    def _writer_loop(self):
        while self.running or not self.frame_queue.empty():
            try:
                item = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.writer is not None:
                self.writer.write(item["frame"])
                self.frames_written += 1
                with self.timestamps_lock:
                    self.timestamps_writer.writerow({"cam_id": self.camera_id, "frame_time": f"{item['frame_time']:.9f}"})
                self.diagnostics_writer.writerow(
                    {
                        "written_frame_index": self.frames_written - 1,
                        "source_frame_index": item["source_frame_index"],
                        "frame_time": f"{item['frame_time']:.9f}",
                        "unix_ns": item["unix_ns"],
                        "queue_depth": self.frame_queue.qsize(),
                    }
                )
            self.frame_queue.task_done()

    def get_frame(self):
        with self.latest_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        return self.frame_count / max(time.perf_counter() - self.started_at, 1e-6)

    def stop(self):
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=3.0)
        if self.writer is not None:
            self.writer.release()
        if self.diagnostics_handle is not None:
            self.diagnostics_handle.close()
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()


def wait_for_initial_frames(cameras, timeout_s):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if all(camera.get_frame() is not None for camera in cameras):
            return True
        time.sleep(0.02)
    return all(camera.get_frame() is not None for camera in cameras)


def main():
    args = parse_args()
    if args.scan:
        for index, name in enumerate(list_devices()):
            print(f"{index}: {name}")
        return

    camera_ids = args.cameras if args.cameras is not None else default_usb_cameras()
    if not camera_ids:
        raise RuntimeError("No cameras selected.")

    extrinsic_dir = args.workspace / "calibration" / "extrinsic"
    extrinsic_dir.mkdir(parents=True, exist_ok=True)

    devices = list_devices()
    print("Selected cameras:")
    for camera_id in camera_ids:
        name = devices[camera_id] if camera_id < len(devices) else "<missing>"
        print(f"  cam_{camera_id}: DirectShow {camera_id} ({name})")

    cameras = []
    timestamps_path = extrinsic_dir / "timestamps.csv"
    timestamps_handle = timestamps_path.open("w", newline="", encoding="utf-8")
    timestamps_writer = csv.DictWriter(timestamps_handle, fieldnames=["cam_id", "frame_time"])
    timestamps_writer.writeheader()
    timestamps_lock = threading.Lock()

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(args.workspace),
        "extrinsic_dir": str(extrinsic_dir),
        "duration_s": args.duration,
        "record_fps": args.record_fps,
        "source": {"width": args.width, "height": args.height, "format": args.format},
        "cameras": {},
    }

    try:
        for camera_id in camera_ids:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = CaliscopeCamera(camera_id, format_index, args.queue_size)
            camera.start()
            cameras.append(camera)
            metadata["cameras"][str(camera_id)] = {"directshow_index": camera_id, "format_index": format_index}
            print(f"Started cam_{camera_id}: {args.format.upper()} {args.width}x{args.height}")

        if not wait_for_initial_frames(cameras, args.startup_timeout):
            missing = [camera.camera_id for camera in cameras if camera.get_frame() is None]
            print(f"Warning: no initial frame from cameras: {missing}", flush=True)

        frame_size = (args.width, args.height)
        for camera in cameras:
            camera.configure_recording(extrinsic_dir, args.record_fps, frame_size, timestamps_writer, timestamps_lock)

        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        print(f"Recording {args.duration:.1f}s into {extrinsic_dir}")
        print("Move/rotate the fiducial board slowly. Press q/Esc to stop early.", flush=True)
        for camera in cameras:
            camera.begin_recording()

        start = time.perf_counter()
        stop_at = start + args.duration
        while time.perf_counter() < stop_at:
            remaining = max(0.0, stop_at - time.perf_counter())
            if not args.no_display:
                grid = make_grid(cameras, args.cols, args.display_height, args.preview_rotation // 90, "REC", remaining)
                if grid is not None:
                    cv2.imshow(WINDOW_NAME, grid)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.02)

        for camera in cameras:
            camera.stop_recording()

        metadata["actual_duration_s"] = time.perf_counter() - start
        for camera in cameras:
            metadata["cameras"][str(camera.camera_id)].update(
                {
                    "frames_written": camera.frames_written,
                    "frames_dropped": camera.frames_dropped,
                    "average_capture_fps": camera.average_fps(),
                    "video": f"cam_{camera.camera_id}.mp4",
                    "diagnostics_csv": f"cam_{camera.camera_id}_frames.csv",
                }
            )
            print(
                f"cam_{camera.camera_id}: wrote {camera.frames_written} frames, "
                f"dropped {camera.frames_dropped}, avg capture {camera.average_fps():.1f} FPS",
                flush=True,
            )
    finally:
        for camera in cameras:
            camera.stop()
        timestamps_handle.close()
        cv2.destroyAllWindows()

    metadata_path = extrinsic_dir / "recording_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {timestamps_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
