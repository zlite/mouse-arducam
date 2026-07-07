#!/usr/bin/env python3
import argparse
import json
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import ten_v4l2_camera_grid as grid


WINDOW_NAME = "Caliscope V4L2 Recorder"
DEFAULT_WORKSPACE = Path("/home/cat/calibration")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record Ubuntu/V4L2 camera videos in Caliscope's "
            "workspace/calibration/{intrinsic,extrinsic}/cam_N.mp4 layout."
        )
    )
    parser.add_argument("--mode", choices=("intrinsic", "extrinsic"), default="intrinsic")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--camera-count", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--display-height", type=int, default=160)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--queue-size", type=int, default=180)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--positions-config", type=Path, default=grid.CONFIG_PATH)
    parser.add_argument(
        "--include-unassigned",
        action="store_true",
        help="Also record cameras that have not been assigned an enclosure position.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cam_N.mp4 files in the selected mode directory.",
    )
    parser.add_argument(
        "--no-extrinsic-seed",
        action="store_true",
        help="Do not create tiny calibration/extrinsic/cam_N.mp4 seed files.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--scan", action="store_true")
    return parser.parse_args()


def cam_id_for_position(position):
    if position is None:
        return None
    return grid.POSITION_ORDER.get(position)


def selected_camera_specs(devices, assignments, include_unassigned):
    specs = []
    used_cam_ids = set()
    next_unassigned_id = len(grid.POSITION_KEYS)

    for device in devices:
        stable_id = grid.stable_device_id(device)
        position = assignments.get(stable_id)
        cam_id = cam_id_for_position(position)
        if cam_id is None:
            if not include_unassigned:
                continue
            while next_unassigned_id in used_cam_ids:
                next_unassigned_id += 1
            cam_id = next_unassigned_id
            next_unassigned_id += 1
        used_cam_ids.add(cam_id)
        specs.append(
            {
                "cam_id": cam_id,
                "device": device,
                "stable_id": stable_id,
                "position": position,
            }
        )

    return sorted(specs, key=lambda spec: spec["cam_id"])


def assigned_devices(assignments):
    devices = []
    for stable_id, position in sorted(assignments.items(), key=lambda item: grid.POSITION_ORDER[item[1]]):
        path = Path(stable_id)
        if not path.exists():
            print(f"Warning: assigned camera is not currently plugged in: {stable_id}", flush=True)
            continue
        devices.append(str(path.resolve()))
    return devices


class RecordingCamera:
    def __init__(self, spec, width, height, fps, fourcc, queue_size):
        self.cam_id = spec["cam_id"]
        self.device = spec["device"]
        self.stable_id = spec["stable_id"]
        self.position = spec["position"]
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.latest_frame = None
        self.latest_shape = None
        self.frame_count = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self.decode_errors = 0
        self.capture_started_at = None
        self.running = False
        self.recording = False
        self.cap = None
        self.writer = None
        self.lock = threading.Lock()
        self.capture_thread = None
        self.writer_thread = None
        self.video_path = None

    def start(self):
        self.cap = grid.open_capture(self.device, self.width, self.height, self.fps, self.fourcc, 1)
        if self.cap is None:
            raise RuntimeError(f"Could not open {self.device}")

        actual_width = int(self.cap.get(grid.cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(grid.cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(grid.cv2.CAP_PROP_FPS)
        print(
            f"cam_{self.cam_id}: {self.device} requested {self.fourcc.upper()} "
            f"{self.width}x{self.height}@{self.fps:g}, opened "
            f"{actual_width}x{actual_height}@{actual_fps:.1f}",
            flush=True,
        )

        self.capture_started_at = time.perf_counter()
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, name=f"intrinsic-cap-{self.cam_id}", daemon=True)
        self.writer_thread = threading.Thread(target=self._writer_loop, name=f"intrinsic-writer-{self.cam_id}", daemon=True)
        self.capture_thread.start()
        self.writer_thread.start()

    def configure_writer(self, intrinsic_dir, record_fps, overwrite):
        self.video_path = intrinsic_dir / f"cam_{self.cam_id}.mp4"
        if self.video_path.exists() and not overwrite:
            raise FileExistsError(f"{self.video_path} already exists; pass --overwrite to replace it")
        fourcc = grid.cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = grid.cv2.VideoWriter(str(self.video_path), fourcc, record_fps, (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {self.video_path}")

    def begin_recording(self):
        self.frames_written = 0
        self.frames_dropped = 0
        self.decode_errors = 0
        self.recording = True

    def stop_recording(self):
        self.recording = False
        self.frame_queue.join()

    def _capture_loop(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
            except grid.cv2.error as exc:
                self.decode_errors += 1
                if self.decode_errors <= 3 or self.decode_errors % 100 == 0:
                    print(
                        f"cam_{self.cam_id}: decode error {self.decode_errors}; continuing ({exc})",
                        flush=True,
                    )
                time.sleep(0.005)
                continue
            if not ok:
                time.sleep(0.005)
                continue
            self.frame_count += 1
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = grid.cv2.resize(frame, (self.width, self.height), interpolation=grid.cv2.INTER_AREA)
            with self.lock:
                self.latest_frame = frame
                self.latest_shape = frame.shape
            if self.recording:
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    self.frames_dropped += 1

    def _writer_loop(self):
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.writer is not None:
                self.writer.write(frame)
                self.frames_written += 1
            self.frame_queue.task_done()

    def read_latest(self, copy_frame=True):
        with self.lock:
            if self.latest_frame is None:
                return False, None, None, self.current_fps()
            frame = self.latest_frame.copy() if copy_frame else self.latest_frame
            return True, frame, self.latest_shape, self.current_fps()

    def current_fps(self):
        if self.capture_started_at is None:
            return 0.0
        return self.frame_count / max(time.perf_counter() - self.capture_started_at, 1e-6)

    def has_frame(self):
        with self.lock:
            return self.latest_frame is not None

    def latest_or_blank(self):
        with self.lock:
            if self.latest_frame is None:
                return grid.np.zeros((self.height, self.width, 3), dtype=grid.np.uint8)
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=3.0)
        if self.writer is not None:
            self.writer.release()
        if self.cap is not None:
            self.cap.release()


def wait_for_initial_frames(cameras, timeout_s):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if all(camera.has_frame() for camera in cameras):
            return True
        time.sleep(0.02)
    return all(camera.has_frame() for camera in cameras)


def write_extrinsic_seed_videos(cameras, extrinsic_dir, fps):
    extrinsic_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        path = extrinsic_dir / f"cam_{camera.cam_id}.mp4"
        if path.exists():
            continue
        writer = grid.cv2.VideoWriter(
            str(path),
            grid.cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (camera.width, camera.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create extrinsic seed video {path}")
        frame = camera.latest_or_blank()
        for _ in range(max(2, int(round(fps)))):
            writer.write(frame)
        writer.release()
        print(f"seeded Caliscope camera set with {path}", flush=True)


def write_manifest(path, args, cameras, output_dir):
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(args.workspace),
        "mode": args.mode,
        "output_dir": str(output_dir),
        "intrinsic_dir": str(args.workspace / "calibration" / "intrinsic"),
        "extrinsic_seed_dir": str(args.workspace / "calibration" / "extrinsic"),
        "duration_s": args.duration,
        "record_fps": args.fps,
        "source": {
            "width": args.width,
            "height": args.height,
            "fourcc": args.fourcc,
        },
        "cameras": {
            str(camera.cam_id): {
                "device": camera.device,
                "stable_id": camera.stable_id,
                "position": camera.position,
                "video": f"cam_{camera.cam_id}.mp4",
                "frames_written": camera.frames_written,
                "frames_dropped": camera.frames_dropped,
                "decode_errors": camera.decode_errors,
                "average_capture_fps": round(camera.current_fps(), 3),
            }
            for camera in cameras
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    grid.require_opencv()
    if args.opencv_threads > 0:
        grid.cv2.setNumThreads(args.opencv_threads)

    if args.scan:
        grid.scan_devices()
        return

    assignments = grid.load_position_assignments(args.positions_config)
    devices = args.devices
    if devices is None and assignments:
        devices = assigned_devices(assignments)
        print(f"selected {len(devices)} cameras from {args.positions_config}", flush=True)
    if devices is None or not devices:
        devices = grid.auto_select_devices(args.width, args.height, args.fps, args.fourcc, args.camera_count)

    specs = selected_camera_specs(devices, assignments, args.include_unassigned)
    if not specs:
        raise RuntimeError(
            "No assigned cameras selected. Run ten_v4l2_camera_grid.py to assign positions, "
            "or pass --include-unassigned."
        )

    intrinsic_dir = args.workspace / "calibration" / "intrinsic"
    extrinsic_dir = args.workspace / "calibration" / "extrinsic"
    output_dir = intrinsic_dir if args.mode == "intrinsic" else extrinsic_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caliscope {args.mode} recording map:", flush=True)
    for spec in specs:
        position = grid.format_position(spec["position"]) if spec["position"] else "unassigned"
        print(f"  cam_{spec['cam_id']}: {position} | {spec['device']} | {spec['stable_id']}", flush=True)

    cameras = [
        RecordingCamera(spec, args.width, args.height, args.fps, args.fourcc, args.queue_size)
        for spec in specs
    ]
    display_period = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    stop_at = None

    try:
        active_cameras = []
        for camera in cameras:
            try:
                camera.start()
            except RuntimeError as exc:
                print(f"Warning: skipping cam_{camera.cam_id}: {exc}", flush=True)
                continue
            active_cameras.append(camera)
        cameras = active_cameras
        if not cameras:
            raise RuntimeError("No cameras could be opened.")
        wait_for_initial_frames(cameras, args.startup_timeout)
        for camera in cameras:
            if not camera.has_frame():
                print(f"Warning: cam_{camera.cam_id} has not delivered an initial frame yet.", flush=True)
            camera.configure_writer(output_dir, args.fps, args.overwrite)

        if not args.no_display:
            grid.cv2.namedWindow(WINDOW_NAME, grid.cv2.WINDOW_NORMAL | grid.cv2.WINDOW_KEEPRATIO)

        print(f"Recording {args.duration:.1f}s of {args.mode} videos into {output_dir}", flush=True)
        if args.mode == "extrinsic":
            print("Move/rotate the calibration board through the shared camera volume. Press q/Esc to stop early.", flush=True)
        else:
            print("Move the Charuco/chessboard through each camera view. Press q/Esc to stop early.", flush=True)
        for camera in cameras:
            camera.begin_recording()

        started_at = time.perf_counter()
        stop_at = started_at + args.duration
        while time.perf_counter() < stop_at:
            loop_started_at = time.perf_counter()
            remaining = max(0.0, stop_at - time.perf_counter())
            if not args.no_display:
                canvas = grid.make_grid(
                    cameras,
                    max(1, args.cols or math.ceil(math.sqrt(len(cameras)))),
                    args.display_height,
                    args.rotation,
                    False,
                    args.width / max(args.height, 1),
                    assignments,
                )
                grid.cv2.putText(
                    canvas,
                    f"REC {remaining:4.1f}s",
                    (10, canvas.shape[0] - 10),
                    grid.cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    grid.cv2.LINE_AA,
                )
                grid.cv2.imshow(WINDOW_NAME, canvas)
                elapsed = time.perf_counter() - loop_started_at
                wait_ms = 1
                if display_period > 0:
                    wait_ms = max(1, int(round((display_period - elapsed) * 1000)))
                key = grid.cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                time.sleep(0.02)

        for camera in cameras:
            camera.stop_recording()

        if args.mode == "intrinsic" and not args.no_extrinsic_seed:
            write_extrinsic_seed_videos(cameras, extrinsic_dir, args.fps)

        manifest_path = output_dir / f"{args.mode}_recording_manifest.json"
        write_manifest(manifest_path, args, cameras, output_dir)
        for camera in cameras:
            print(
                f"cam_{camera.cam_id}: wrote {camera.frames_written} frames, "
                f"dropped {camera.frames_dropped}, decode errors {camera.decode_errors}, "
                f"avg capture {camera.current_fps():.1f} FPS",
                flush=True,
            )
        print(f"Wrote {manifest_path}", flush=True)
    finally:
        for camera in cameras:
            camera.stop()
        grid.cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
