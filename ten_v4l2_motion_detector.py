#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import ten_v4l2_camera_grid as grid
import gstreamer_mjpeg as gst_mjpeg


WINDOW_NAME = "Ten V4L2 Motion Detector"
CONFIG_PATH = Path(__file__).with_name("v4l2_motion_detector_config.json")
ROLE_GROUP_KEYS = {
    ord("s"): "side",
    ord("t"): "top",
    ord("f"): "front",
    ord("b"): "back",
    ord("l"): "left",
    ord("r"): "right",
}
ROLE_GROUP_LIMITS = {
    "side": 10,
    "front": 2,
    "back": 2,
    "left": 2,
    "right": 2,
    "top": 4,
}
DEFAULT_ROLE_ROIS = {
    "side": "0.1,0.5,0.8,0.5",
    "top": "0,0,1,1",
    "front": "0.1,0.5,0.8,0.5",
    "back": "0.1,0.5,0.8,0.5",
    "left": "0,0.15,1,0.85",
    "right": "0,0.15,1,0.85",
    "unassigned": "0,0,1,1",
}
CALISCOPE_ROLE_IDS = {
    "front_1": 0,
    "front_2": 1,
    "back_1": 2,
    "back_2": 3,
    "side_1": 4,
    "right_1": 4,
    "side_2": 5,
    "right_2": 5,
    "top_1": 8,
    "top_2": 9,
    "top_3": 10,
    "top_4": 11,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Low-compute multi-camera motion detector for an inward-facing mouse cage rig. "
            "A camera becomes active only when motion exceeds frame coverage, persists for "
            "enough frames, and covers enough of the selected ROI."
        )
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Camera devices. Defaults to the first --camera-count /dev/video* devices that open.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=800, help="Requested capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS.")
    parser.add_argument("--display-fps", type=float, default=15.0, help="Preview and detection FPS. Use 0 for uncapped.")
    parser.add_argument("--fourcc", default="MJPG", help="Requested V4L2 pixel format.")
    parser.add_argument(
        "--capture-backend",
        choices=("gstreamer", "opencv"),
        default="gstreamer",
        help="GStreamer preserves all compressed 30 FPS frames; OpenCV is the legacy decoded path.",
    )
    parser.add_argument(
        "--encoded-buffer-frames",
        type=int,
        default=180,
        help="Compressed MJPEG packet ring per camera for the GStreamer backend.",
    )
    parser.add_argument("--camera-count", type=int, default=10, help="Number of cameras to auto-select.")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns.")
    parser.add_argument("--display-height", type=int, default=200, help="Displayed height per tile.")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after seconds. 0 runs until q/Esc.")
    parser.add_argument("--no-display", action="store_true", help="Run headless and print/log events only.")
    parser.add_argument("--no-overlay", action="store_true", help="Hide overlays.")
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV worker threads. 1 often lowers CPU contention with many cameras. Use 0 for OpenCV default.",
    )
    parser.add_argument("--scan", action="store_true", help="List V4L2 video devices and exit.")

    parser.add_argument("--detector-scale", type=float, default=0.20, help="Downscale factor used for detection.")
    parser.add_argument("--pixel-threshold", type=int, default=35, help="Per-pixel grayscale background difference threshold.")
    parser.add_argument(
        "--foreground-polarity",
        choices=("dark", "absolute"),
        default="dark",
        help="Detect dark foreground on the white cage, or use absolute background difference.",
    )
    parser.add_argument("--background-alpha", type=float, default=0.02, help="Running background update rate.")
    parser.add_argument(
        "--no-lighting-compensation",
        action="store_false",
        dest="lighting_compensation",
        help="Disable global brightness-shift compensation.",
    )
    parser.set_defaults(lighting_compensation=True)
    parser.add_argument(
        "--max-roi-motion-ratio",
        type=float,
        default=0.60,
        help="Treat changes covering more than this fraction of the ROI as lighting changes.",
    )
    parser.add_argument(
        "--lighting-background-alpha",
        type=float,
        default=0.20,
        help="Background update rate after a broad lighting change.",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=2.0,
        help="Initial seconds used only for background learning before detections are allowed.",
    )
    parser.add_argument("--open-kernel", type=int, default=3, help="Odd morphology open kernel size. 0 disables.")
    parser.add_argument("--close-kernel", type=int, default=9, help="Odd morphology close kernel size. 0 disables.")
    parser.add_argument(
        "--min-frame-motion-ratio",
        type=float,
        default=0.01,
        help="Largest moving component must cover at least this fraction of the full camera frame.",
    )
    parser.add_argument(
        "--min-roi-motion-ratio",
        type=float,
        default=0.03,
        help="Moving pixels must cover at least this fraction of the ROI.",
    )
    parser.add_argument(
        "--motion-frames",
        type=int,
        default=3,
        help="Consecutive qualifying frames required before a camera is active.",
    )
    parser.add_argument(
        "--presence-hold-sec",
        type=float,
        default=0.75,
        help="Keep a detected silhouette present through short stops or occlusions.",
    )
    parser.add_argument(
        "--axis-smoothing",
        type=float,
        default=0.35,
        help="Update weight for temporal long-axis smoothing in the range (0, 1].",
    )
    parser.add_argument(
        "--roi",
        default="0,0,1,1",
        help="Fallback ROI as x,y,w,h. Values <= 1 are fractions of the rotated frame.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="JSON file storing camera role assignments and per-role ROIs.",
    )
    parser.add_argument(
        "--role-roi",
        action="append",
        default=[],
        help="Override a role ROI as role=x,y,w,h. May be passed multiple times.",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("all", "any", "min-cameras"),
        default="min-cameras",
        help="How camera detections form a synced rig event.",
    )
    parser.add_argument(
        "--min-active-cameras",
        type=int,
        default=4,
        help="Active camera count required when --sync-mode min-cameras is used.",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help="Optional CSV path for camera and rig motion events.",
    )
    parser.add_argument(
        "--record-events",
        action="store_true",
        help="When the rig trigger is active, write synchronized clips from every camera.",
    )
    parser.add_argument("--record-dir", type=Path, default=Path("motion_recordings"), help="Root folder for motion-triggered clips.")
    parser.add_argument("--pre-roll-sec", type=float, default=0.5, help="Seconds of RAM-buffered frames saved before activation.")
    parser.add_argument("--post-roll-sec", type=float, default=1.0, help="Seconds to keep writing after motion drops.")
    parser.add_argument("--max-clip-sec", type=float, default=30.0, help="Split long motion clips after this many seconds. 0 disables.")
    parser.add_argument("--csv-flush-sec", type=float, default=1.0, help="Flush frame timestamp CSV at this interval. 0 flushes only on close.")
    parser.add_argument("--record-queue-frames", type=int, default=300, help="Per-camera async writer queue size in frame records.")
    parser.add_argument(
        "--record-scale",
        type=float,
        default=1.0,
        help="Scale recorded MP4 frames before queueing/writing. 1.0 keeps full resolution; 0.5 halves width and height.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="FPS stored in MP4 metadata. 0 uses --display-fps, then --fps.",
    )
    return parser.parse_args()


def load_config(path):
    if not path.exists():
        return {"assignments": {}, "role_rois": dict(DEFAULT_ROLE_ROIS)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not load {path}: {exc}", flush=True)
        return {"assignments": {}, "role_rois": dict(DEFAULT_ROLE_ROIS)}
    data.setdefault("assignments", {})
    role_rois = dict(DEFAULT_ROLE_ROIS)
    role_rois.update(data.get("role_rois", {}))
    data["role_rois"] = role_rois
    return data


def save_config(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"saved motion detector config to {path}", flush=True)


def apply_role_roi_overrides(config, overrides):
    for override in overrides:
        if "=" not in override:
            raise ValueError("--role-roi must look like role=x,y,w,h")
        role, roi_text = override.split("=", 1)
        role = role.strip()
        parse_roi(roi_text)
        config["role_rois"][role] = roi_text


def camera_role(config, camera):
    return config.get("assignments", {}).get(camera.stable_id, "unassigned")


def role_roi(config, fallback_roi, role):
    roi_text = config.get("role_rois", {}).get(role)
    if roi_text is None and "_" in role:
        roi_text = config.get("role_rois", {}).get(role.rsplit("_", 1)[0])
    if roi_text is None:
        return fallback_roi
    return parse_roi(roi_text)


def role_roi_text(config, role, fallback_text):
    roi_text = config.get("role_rois", {}).get(role)
    if roi_text is None and "_" in role:
        roi_text = config.get("role_rois", {}).get(role.rsplit("_", 1)[0])
    return roi_text or fallback_text


def odd_kernel(size):
    size = int(size)
    if size <= 0:
        return None
    if size % 2 == 0:
        size += 1
    return grid.cv2.getStructuringElement(grid.cv2.MORPH_ELLIPSE, (size, size))


def parse_roi(roi_text):
    values = [float(part.strip()) for part in roi_text.split(",")]
    if len(values) != 4:
        raise ValueError("--roi must contain four values: x,y,w,h")
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("--roi width and height must be positive")
    return x, y, width, height


def roi_pixels(roi, shape):
    frame_height, frame_width = shape[:2]
    x, y, width, height = roi
    if max(abs(value) for value in roi) <= 1.0:
        x *= frame_width
        width *= frame_width
        y *= frame_height
        height *= frame_height
    x1 = max(0, min(frame_width - 1, int(round(x))))
    y1 = max(0, min(frame_height - 1, int(round(y))))
    x2 = max(x1 + 1, min(frame_width, int(round(x + width))))
    y2 = max(y1 + 1, min(frame_height, int(round(y + height))))
    return x1, y1, x2, y2


def crop_box_to_full(box, roi_x, roi_y, scale_x, scale_y):
    x, y, width, height = box
    return (
        int(round(roi_x + x / scale_x)),
        int(round(roi_y + y / scale_y)),
        int(round(width / scale_x)),
        int(round(height / scale_y)),
    )


class LowComputeMotionDetector:
    def __init__(self, args, roi):
        self.args = args
        self.roi = roi
        self.background = None
        self.open_kernel = odd_kernel(args.open_kernel)
        self.close_kernel = odd_kernel(args.close_kernel)
        self.consecutive_hits = 0
        self.active = False
        self.presence_until = 0.0
        self.axis_state = None
        self.last_bbox = None
        self.last = self.empty_result()

    def set_roi(self, roi, reset_background=False):
        self.roi = roi
        if reset_background:
            self.background = None
            self.consecutive_hits = 0
            self.active = False
            self.presence_until = 0.0
            self.axis_state = None
            self.last_bbox = None
            self.last = self.empty_result(status="learning_bg")

    def _smooth_axis(self, axis):
        np = grid.np
        (x1, y1), (x2, y2) = axis
        center = np.asarray(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float64)
        vector = np.asarray((x2 - x1, y2 - y1), dtype=np.float64)
        length = float(np.linalg.norm(vector))
        if length < 1.0:
            return None
        direction = vector / length

        if self.axis_state is not None:
            previous_center, previous_direction, previous_length = self.axis_state
            if float(np.dot(direction, previous_direction)) < 0:
                direction = -direction
            alpha = min(1.0, max(1e-6, float(self.args.axis_smoothing)))
            center = (1.0 - alpha) * previous_center + alpha * center
            direction = (1.0 - alpha) * previous_direction + alpha * direction
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-9:
                direction /= direction_norm
            length = (1.0 - alpha) * previous_length + alpha * length

        self.axis_state = (center, direction, length)
        half = direction * (length * 0.5)
        return (
            (int(round(center[0] - half[0])), int(round(center[1] - half[1]))),
            (int(round(center[0] + half[0])), int(round(center[1] + half[1]))),
        )

    def update(self, frame, elapsed_sec):
        cv2 = grid.cv2
        np = grid.np
        scale = self.args.detector_scale
        if not 0 < scale <= 1:
            raise ValueError("--detector-scale must be in the range (0, 1]")

        roi_x1_full, roi_y1_full, roi_x2_full, roi_y2_full = roi_pixels(self.roi, frame.shape)
        roi_frame = frame[roi_y1_full:roi_y2_full, roi_x1_full:roi_x2_full]
        roi_height, roi_width = roi_frame.shape[:2]
        small_width = max(1, int(round(roi_width * scale)))
        small_height = max(1, int(round(roi_height * scale)))
        scale_x = small_width / max(roi_width, 1)
        scale_y = small_height / max(roi_height, 1)
        small = cv2.resize(roi_frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        roi_area = max(1, gray.shape[0] * gray.shape[1])
        frame_area = max(1, int(round(frame.shape[0] * scale)) * int(round(frame.shape[1] * scale)))

        if self.background is None or self.background.shape != gray.shape:
            self.background = gray.astype(np.float32)
            self.last = self.empty_result(
                status="learning_bg",
                roi=(roi_x1_full, roi_y1_full, roi_x2_full - roi_x1_full, roi_y2_full - roi_y1_full),
            )
            return self.last

        bg = cv2.convertScaleAbs(self.background)
        if self.args.lighting_compensation:
            brightness_shift = cv2.mean(gray)[0] - cv2.mean(bg)[0]
            comparison = cv2.addWeighted(gray, 1.0, bg, 0.0, -brightness_shift)
        else:
            comparison = gray
        delta = cv2.subtract(bg, comparison) if self.args.foreground_polarity == "dark" else cv2.absdiff(bg, comparison)

        _, mask = cv2.threshold(delta, self.args.pixel_threshold, 255, cv2.THRESH_BINARY)
        if self.open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        if self.close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        roi_motion_pixels = cv2.countNonZero(mask)
        roi_motion_ratio = roi_motion_pixels / roi_area
        broad_change = roi_motion_ratio > self.args.max_roi_motion_ratio

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        largest_area = 0
        largest_label = None
        largest_box = None
        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area > largest_area:
                largest_area = int(area)
                largest_label = label
                largest_box = (int(x), int(y), int(width), int(height))

        current_bbox = (
            crop_box_to_full(largest_box, roi_x1_full, roi_y1_full, scale_x, scale_y)
            if largest_box is not None and not broad_change
            else None
        )
        current_axis = None
        axis_candidate = (
            largest_area / frame_area >= self.args.min_frame_motion_ratio
            and roi_motion_ratio >= self.args.min_roi_motion_ratio
        )
        if largest_label is not None and not broad_change and largest_area >= 3 and axis_candidate:
            ys, xs = np.nonzero(_labels == largest_label)
            if len(xs) >= 3:
                points = np.column_stack((xs, ys)).astype(np.float32)
                vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1).astype(float)
                projections = (points[:, 0] - cx) * vx + (points[:, 1] - cy) * vy
                low = float(projections.min())
                high = float(projections.max())
                current_axis = self._smooth_axis(
                    (
                        (
                            roi_x1_full + (cx + low * vx) / scale_x,
                            roi_y1_full + (cy + low * vy) / scale_y,
                        ),
                        (
                            roi_x1_full + (cx + high * vx) / scale_x,
                            roi_y1_full + (cy + high * vy) / scale_y,
                        ),
                    )
                )
                self.last_bbox = current_bbox

        frame_motion_ratio = largest_area / frame_area
        qualifies = (
            elapsed_sec >= self.args.warmup_sec
            and not broad_change
            and frame_motion_ratio >= self.args.min_frame_motion_ratio
            and roi_motion_ratio >= self.args.min_roi_motion_ratio
        )
        if qualifies:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        was_active = self.active
        self.active = self.consecutive_hits >= self.args.motion_frames
        if self.active:
            self.presence_until = elapsed_sec + max(0.0, self.args.presence_hold_sec)
        present = self.active or (self.axis_state is not None and elapsed_sec <= self.presence_until)
        if elapsed_sec < self.args.warmup_sec:
            status = "learning_bg"
        elif broad_change:
            status = "lighting_change"
        elif self.active:
            status = "active"
        elif present:
            status = "present_hold"
        elif qualifies:
            status = "candidate"
        else:
            status = "idle"

        if not qualifies:
            alpha = self.args.lighting_background_alpha if broad_change else self.args.background_alpha
            cv2.accumulateWeighted(gray, self.background, alpha)

        self.last = {
            "active": self.active,
            "present": present,
            "just_activated": self.active and not was_active,
            "qualifies": qualifies,
            "status": status,
            "consecutive_hits": self.consecutive_hits,
            "frame_motion_ratio": float(frame_motion_ratio),
            "roi_motion_ratio": float(roi_motion_ratio),
            "largest_area": int(round(largest_area / max(scale_x * scale_y, 1e-9))),
            "bbox": current_bbox if current_bbox is not None else self.last_bbox if present else None,
            "axis": current_axis if current_axis is not None else self._axis_from_state() if present else None,
            "roi": (roi_x1_full, roi_y1_full, roi_x2_full - roi_x1_full, roi_y2_full - roi_y1_full),
        }
        return self.last

    def _axis_from_state(self):
        if self.axis_state is None:
            return None
        center, direction, length = self.axis_state
        half = direction * (length * 0.5)
        return (
            (int(round(center[0] - half[0])), int(round(center[1] - half[1]))),
            (int(round(center[0] + half[0])), int(round(center[1] + half[1]))),
        )

    @staticmethod
    def empty_result(status="idle", roi=None):
        return {
            "active": False,
            "present": False,
            "just_activated": False,
            "qualifies": False,
            "status": status,
            "consecutive_hits": 0,
            "frame_motion_ratio": 0.0,
            "roi_motion_ratio": 0.0,
            "largest_area": 0,
            "bbox": None,
            "axis": None,
            "roi": roi,
        }


class EventLogger:
    def __init__(self, path):
        self.path = path
        self.handle = None
        self.writer = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.handle)
            self.writer.writerow(
                [
                    "unix_time",
                    "perf_time",
                    "elapsed_sec",
                    "event_type",
                    "device",
                    "active_cameras",
                    "frame_motion_ratio",
                    "roi_motion_ratio",
                    "consecutive_hits",
                    "bbox",
                ]
            )

    def write_camera_event(self, elapsed_sec, camera, result):
        if self.writer is None:
            return
        now = time.time()
        self.writer.writerow(
            [
                f"{now:.6f}",
                f"{time.perf_counter():.9f}",
                f"{elapsed_sec:.6f}",
                "camera_active",
                camera.device,
                "",
                f"{result['frame_motion_ratio']:.6f}",
                f"{result['roi_motion_ratio']:.6f}",
                result["consecutive_hits"],
                result["bbox"],
            ]
        )
        self.handle.flush()

    def write_rig_event(self, elapsed_sec, active_cameras):
        if self.writer is None:
            return
        now = time.time()
        self.writer.writerow(
            [
                f"{now:.6f}",
                f"{time.perf_counter():.9f}",
                f"{elapsed_sec:.6f}",
                "rig_active",
                "",
                " ".join(camera.device for camera in active_cameras),
                "",
                "",
                "",
                "",
            ]
        )
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()


def safe_name(text):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text)).strip("_") or "camera"


class CameraMotionClipRecorder:
    def __init__(
        self,
        camera,
        role,
        output_root,
        fps,
        pre_roll_sec,
        post_roll_sec,
        max_clip_sec,
        csv_flush_sec,
        queue_frames,
        record_scale,
    ):
        self.camera = camera
        self.role = role
        self.output_root = output_root
        self.fps = fps
        self.post_roll_sec = post_roll_sec
        self.max_clip_sec = max_clip_sec
        self.csv_flush_sec = csv_flush_sec
        self.record_scale = record_scale
        self.folder = output_root / safe_name(role)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=max(1, int(round(pre_roll_sec * fps))))
        self.queue = queue.Queue(maxsize=max(4, int(queue_frames)))
        self.thread = threading.Thread(target=self._writer_loop, name=f"motion-writer-{safe_name(role)}", daemon=True)
        self.active_until = 0.0
        self.recording = False
        self.segment_started_elapsed = None
        self.dropped_records = 0
        self.last_record = None
        self.thread.start()

    def add_snapshot(self, snapshot, rig_active, event_stamp=None, event_started_unix=None):
        if not snapshot["ok"]:
            if self.last_record is None:
                self._maybe_close(snapshot.get("elapsed_sec", 0.0))
                return
            record = dict(self.last_record)
            record.update(
                {
                    "unix_time": snapshot["unix_time"],
                    "perf_time": snapshot["perf_time"],
                    "source_frame_index": None,
                    "capture_unix_time": None,
                    "capture_perf_time": None,
                    "elapsed_sec": snapshot["elapsed_sec"],
                    "active": False,
                    "present": False,
                    "rig_active": bool(rig_active),
                    "frame_motion_ratio": 0.0,
                    "roi_motion_ratio": 0.0,
                    "bbox": None,
                    "axis": None,
                }
            )
        else:
            record = {
                "frame": self._record_frame(snapshot["frame"]),
                "unix_time": snapshot["unix_time"],
                "perf_time": snapshot["perf_time"],
                "source_frame_index": snapshot.get("source_frame_index"),
                "capture_unix_time": snapshot.get("capture_unix_time"),
                "capture_perf_time": snapshot.get("capture_perf_time"),
                "elapsed_sec": snapshot["elapsed_sec"],
                "active": bool(snapshot["result"]["active"]),
                "present": bool(snapshot["result"]["present"]),
                "rig_active": bool(rig_active),
                "frame_motion_ratio": snapshot["result"]["frame_motion_ratio"],
                "roi_motion_ratio": snapshot["result"]["roi_motion_ratio"],
                "bbox": snapshot["result"]["bbox"],
                "axis": snapshot["result"]["axis"],
            }
        self.last_record = record
        self.buffer.append(record)

        if rig_active:
            split_segment = False
            self.active_until = max(self.active_until, record["elapsed_sec"] + self.post_roll_sec)
            if self.recording and self.max_clip_sec > 0 and self.segment_started_elapsed is not None:
                if record["elapsed_sec"] - self.segment_started_elapsed >= self.max_clip_sec:
                    self._enqueue(("close", record["elapsed_sec"]))
                    self.recording = False
                    self.segment_started_elapsed = None
                    split_segment = True
            if not self.recording:
                self.recording = True
                self.segment_started_elapsed = record["elapsed_sec"]
                stamp = (
                    datetime.fromtimestamp(record["unix_time"]).strftime("%Y%m%d_%H%M%S_%f")
                    if split_segment
                    else event_stamp or datetime.fromtimestamp(record["unix_time"]).strftime("%Y%m%d_%H%M%S_%f")
                )
                self._enqueue(
                    (
                        "start",
                        {
                            "records": list(self.buffer),
                            "stamp": stamp,
                            "started_unix": record["unix_time"]
                            if split_segment
                            else event_started_unix or record["unix_time"],
                        },
                    )
                )
                return

        if self.recording:
            if record["elapsed_sec"] <= self.active_until:
                self._enqueue(("write", record))
            else:
                self._enqueue(("close", record["elapsed_sec"]))
                self.recording = False
                self.segment_started_elapsed = None

    def close(self):
        if self.recording:
            self._enqueue(("close", None), block=True)
            self.recording = False
        self._enqueue(("stop", None), block=True)
        self.thread.join(timeout=5.0)
        if self.dropped_records:
            print(f"{self.role}: dropped {self.dropped_records} queued recording frames", flush=True)

    def _record_frame(self, frame):
        if self.record_scale == 1.0:
            return frame.copy()
        cv2 = grid.cv2
        height, width = frame.shape[:2]
        scaled_width = max(1, int(round(width * self.record_scale)))
        scaled_height = max(1, int(round(height * self.record_scale)))
        return cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    def _enqueue(self, item, block=False):
        try:
            if block:
                self.queue.put(item, timeout=2.0)
            else:
                self.queue.put_nowait(item)
        except queue.Full:
            self.dropped_records += 1

    def _writer_loop(self):
        writer = None
        csv_handle = None
        csv_writer = None
        event_json_path = None
        video_path = None
        written_index = 0
        event_started_elapsed = None
        event_started_unix = None
        last_csv_flush = time.perf_counter()
        while True:
            command, payload = self.queue.get()
            try:
                if command == "start":
                    if writer is not None:
                        writer, csv_handle, csv_writer = self._close_event(
                            writer,
                            csv_handle,
                            csv_writer,
                            event_json_path,
                            video_path,
                            written_index,
                            event_started_elapsed,
                            event_started_unix,
                            None,
                        )
                    records = payload["records"]
                    if not records:
                        continue
                    writer, csv_handle, csv_writer, event_json_path, video_path = self._open_event(
                        records[-1], payload["stamp"]
                    )
                    written_index = 0
                    event_started_elapsed = records[-1]["elapsed_sec"]
                    event_started_unix = payload["started_unix"]
                    for record in records:
                        written_index = self._write_record(writer, csv_writer, written_index, record)
                    last_csv_flush = time.perf_counter()
                elif command == "write":
                    if writer is not None:
                        written_index = self._write_record(writer, csv_writer, written_index, payload)
                        if self.csv_flush_sec > 0 and time.perf_counter() - last_csv_flush >= self.csv_flush_sec:
                            csv_handle.flush()
                            last_csv_flush = time.perf_counter()
                elif command == "close":
                    if writer is not None:
                        writer, csv_handle, csv_writer = self._close_event(
                            writer,
                            csv_handle,
                            csv_writer,
                            event_json_path,
                            video_path,
                            written_index,
                            event_started_elapsed,
                            event_started_unix,
                            payload,
                        )
                        event_json_path = None
                        video_path = None
                        written_index = 0
                        event_started_elapsed = None
                        event_started_unix = None
                elif command == "stop":
                    if writer is not None:
                        self._close_event(
                            writer,
                            csv_handle,
                            csv_writer,
                            event_json_path,
                            video_path,
                            written_index,
                            event_started_elapsed,
                            event_started_unix,
                            None,
                        )
                    return
            finally:
                self.queue.task_done()

    def _open_event(self, record, stem):
        cv2 = grid.cv2
        height, width = record["frame"].shape[:2]
        video_path = self.folder / f"{stem}.mp4"
        csv_path = self.folder / f"{stem}.csv"
        event_json_path = self.folder / f"{stem}.json"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        csv_handle = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            [
                "written_index",
                "unix_time",
                "perf_time",
                "source_frame_index",
                "capture_unix_time",
                "capture_perf_time",
                "elapsed_sec",
                "active",
                "present",
                "rig_active",
                "frame_motion_ratio",
                "roi_motion_ratio",
                "bbox",
                "axis",
            ]
        )
        return writer, csv_handle, csv_writer, event_json_path, video_path

    @staticmethod
    def _write_record(writer, csv_writer, written_index, record):
        writer.write(record["frame"])
        csv_writer.writerow(
            [
                written_index,
                f"{record['unix_time']:.6f}",
                f"{record['perf_time']:.9f}",
                record["source_frame_index"],
                f"{record['capture_unix_time']:.6f}" if record["capture_unix_time"] is not None else "",
                f"{record['capture_perf_time']:.9f}" if record["capture_perf_time"] is not None else "",
                f"{record['elapsed_sec']:.6f}",
                int(record["active"]),
                int(record["present"]),
                int(record["rig_active"]),
                f"{record['frame_motion_ratio']:.6f}",
                f"{record['roi_motion_ratio']:.6f}",
                record["bbox"],
                record["axis"],
            ]
        )
        return written_index + 1

    def _maybe_close(self, elapsed_sec):
        if self.recording and elapsed_sec > self.active_until:
            self._enqueue(("close", elapsed_sec))
            self.recording = False
            self.segment_started_elapsed = None

    def _close_event(
        self,
        writer,
        csv_handle,
        csv_writer,
        event_json_path,
        video_path,
        written_index,
        event_started_elapsed,
        event_started_unix,
        ended_elapsed,
    ):
        writer.release()
        csv_handle.close()
        metadata = {
            "mp4": video_path.name,
            "frames": written_index,
            "start_time": datetime.fromtimestamp(event_started_unix).astimezone().isoformat(timespec="milliseconds"),
        }
        event_json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"wrote motion clip {video_path}", flush=True)
        return None, None, None


class MotionRecordingManager:
    def __init__(
        self,
        cameras,
        config,
        output_root,
        fps,
        pre_roll_sec,
        post_roll_sec,
        max_clip_sec,
        csv_flush_sec,
        queue_frames,
        record_scale,
    ):
        self.recorders = {
            camera.stable_id: CameraMotionClipRecorder(
                camera,
                camera_role(config, camera),
                output_root,
                fps,
                pre_roll_sec,
                post_roll_sec,
                max_clip_sec,
                csv_flush_sec,
                queue_frames,
                record_scale,
            )
            for camera in cameras
        }
        self.rig_was_active = False
        self.event_stamp = None
        self.event_started_unix = None

    def update(self, snapshots, rig_active):
        if rig_active and not self.rig_was_active:
            event_started_unix = max(
                (snapshot["unix_time"] for snapshot in snapshots if snapshot["ok"]),
                default=time.time(),
            )
            self.event_started_unix = event_started_unix
            self.event_stamp = datetime.fromtimestamp(event_started_unix).strftime("%Y%m%d_%H%M%S_%f")
        for snapshot in snapshots:
            recorder = self.recorders.get(snapshot["camera"].stable_id)
            if recorder is not None:
                recorder.add_snapshot(
                    snapshot,
                    rig_active,
                    event_stamp=self.event_stamp,
                    event_started_unix=self.event_started_unix,
                )
        self.rig_was_active = rig_active

    def close(self):
        for recorder in self.recorders.values():
            recorder.close()


class CompressedRigRecordingManager:
    def __init__(
        self,
        cameras,
        config,
        output_root,
        fps,
        pre_roll_sec,
        post_roll_sec,
        max_clip_sec,
        min_active_cameras,
    ):
        self.cameras = cameras
        self.output_root = output_root
        self.fps = int(round(fps))
        self.pre_roll_sec = pre_roll_sec
        self.post_roll_sec = post_roll_sec
        self.max_clip_sec = max_clip_sec
        self.min_active_cameras = min_active_cameras
        self.roles = {camera.stable_id: camera_role(config, camera) for camera in cameras}
        used_ids = set()
        self.camera_ids = {}
        fallback_id = 100
        for camera in cameras:
            role = self.roles[camera.stable_id]
            camera_id = CALISCOPE_ROLE_IDS.get(role)
            if camera_id is None or camera_id in used_ids:
                while fallback_id in used_ids:
                    fallback_id += 1
                camera_id = fallback_id
                fallback_id += 1
            self.camera_ids[camera.stable_id] = camera_id
            used_ids.add(camera_id)

        pre_roll_frames = max(1, int(math.ceil(pre_roll_sec * self.fps)) + 2)
        self.prebuffers = {camera.stable_id: deque(maxlen=pre_roll_frames) for camera in cameras}
        self.last_indices = {camera.stable_id: None for camera in cameras}
        self.capture_misses = {camera.stable_id: 0 for camera in cameras}
        self.recording = False
        self.event_packets = None
        self.event_stamp = None
        self.event_trigger_unix = None
        self.active_until_unix = 0.0
        self.writer_queue = queue.Queue(maxsize=2)
        self.writer_thread = threading.Thread(target=self._writer_loop, name="aligned-event-writer", daemon=True)
        self.writer_thread.start()

    def update(self, snapshots, rig_active):
        now_unix = max((snapshot["unix_time"] for snapshot in snapshots), default=time.time())
        packet_batches = {}
        for snapshot in snapshots:
            camera = snapshot["camera"]
            stable_id = camera.stable_id
            packets, missed = camera.encoded_packets_after(self.last_indices[stable_id])
            if packets:
                self.last_indices[stable_id] = packets[-1]["source_frame_index"]
            self.capture_misses[stable_id] += missed
            result = snapshot["result"]
            annotated = []
            for packet in packets:
                record = dict(packet)
                record.update(
                    {
                        "active": bool(result["active"]),
                        "present": bool(result["present"]),
                        "frame_motion_ratio": result["frame_motion_ratio"],
                        "roi_motion_ratio": result["roi_motion_ratio"],
                        "bbox": result["bbox"],
                        "axis": result["axis"],
                    }
                )
                annotated.append(record)
                self.prebuffers[stable_id].append(record)
            packet_batches[stable_id] = annotated

        started_now = False
        if rig_active:
            self.active_until_unix = now_unix + self.post_roll_sec
            if not self.recording:
                self._start_event(now_unix)
                started_now = True

        if self.recording and not started_now:
            for stable_id, packets in packet_batches.items():
                if packets:
                    self.event_packets[stable_id].extend(packets)

        if self.recording:

            duration = now_unix - self.event_trigger_unix
            should_split = self.max_clip_sec > 0 and duration >= self.max_clip_sec
            should_close = not rig_active and now_unix > self.active_until_unix
            if should_split or should_close:
                self._finish_event()
                if should_split and rig_active:
                    self._start_event(now_unix)

    def _start_event(self, trigger_unix):
        self.recording = True
        self.event_trigger_unix = trigger_unix
        self.event_stamp = datetime.fromtimestamp(trigger_unix).strftime("%Y%m%d_%H%M%S_%f")
        cutoff = trigger_unix - self.pre_roll_sec
        self.event_packets = {
            stable_id: [packet for packet in buffer if packet["capture_unix_time"] >= cutoff]
            for stable_id, buffer in self.prebuffers.items()
        }

    def _finish_event(self):
        if not self.recording:
            return
        payload = {
            "stamp": self.event_stamp,
            "trigger_unix": self.event_trigger_unix,
            "packets": self.event_packets,
        }
        self.writer_queue.put(payload)
        self.recording = False
        self.event_packets = None
        self.event_stamp = None
        self.event_trigger_unix = None

    @staticmethod
    def _nearest_packet(packets, times, target):
        position = bisect.bisect_left(times, target)
        candidates = []
        if position < len(packets):
            candidates.append(packets[position])
        if position > 0:
            candidates.append(packets[position - 1])
        if not candidates:
            return None, math.inf
        packet = min(candidates, key=lambda item: abs(item["capture_unix_time"] - target))
        return packet, abs(packet["capture_unix_time"] - target)

    @staticmethod
    def _longest_true_run(values):
        best = None
        start = None
        for index, value in enumerate(values + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if best is None or index - start > best[1] - best[0]:
                    best = (start, index)
                start = None
        return best

    def _align_event(self, payload):
        packet_sets = {stable_id: packets for stable_id, packets in payload["packets"].items() if packets}
        if len(packet_sets) < self.min_active_cameras:
            return None
        start = max(packets[0]["capture_unix_time"] for packets in packet_sets.values())
        end = min(packets[-1]["capture_unix_time"] for packets in packet_sets.values())
        if end <= start:
            return None
        period = 1.0 / self.fps
        first_tick = math.ceil(start * self.fps) / self.fps
        frame_count = int(math.floor((end - first_tick) * self.fps)) + 1
        targets = [first_tick + index * period for index in range(max(0, frame_count))]
        tolerance = period * 0.75
        times = {
            stable_id: [packet["capture_unix_time"] for packet in packets]
            for stable_id, packets in packet_sets.items()
        }
        aligned = []
        usable = []
        for target in targets:
            row = {}
            visible_count = 0
            for stable_id, packets in packet_sets.items():
                packet, error = self._nearest_packet(packets, times[stable_id], target)
                valid = packet is not None and error <= tolerance
                visible = valid and packet["present"]
                visible_count += int(visible)
                row[stable_id] = (packet, error, valid, visible)
            aligned.append(row)
            usable.append(visible_count >= self.min_active_cameras)
        run = self._longest_true_run(usable)
        if run is None:
            return None
        start_index, end_index = run
        targets = targets[start_index:end_index]
        aligned = aligned[start_index:end_index]
        selected_ids = {
            stable_id
            for row in aligned
            for stable_id, (_packet, _error, _valid, visible) in row.items()
            if visible
        }
        if len(selected_ids) < self.min_active_cameras:
            return None
        return targets, aligned, selected_ids

    def _write_event(self, payload):
        alignment = self._align_event(payload)
        if alignment is None:
            print(f"discarded {payload['stamp']}: no continuous four-camera 30 FPS interval", flush=True)
            return
        targets, aligned, selected_ids = alignment
        event_dir = self.output_root / "aligned" / payload["stamp"]
        event_dir.mkdir(parents=True, exist_ok=True)
        timestamp_path = event_dir / "timestamps.csv"
        timestamp_handle = timestamp_path.open("w", newline="", encoding="utf-8")
        timestamp_writer = csv.writer(timestamp_handle)
        timestamp_writer.writerow(
            [
                "sync_index",
                "target_unix_time",
                "cam_id",
                "role",
                "source_frame_index",
                "capture_unix_time",
                "time_error_sec",
                "valid",
                "present",
                "bbox",
                "axis",
            ]
        )
        try:
            for stable_id in sorted(selected_ids, key=lambda item: self.camera_ids[item]):
                camera_id = self.camera_ids[stable_id]
                role = self.roles[stable_id]
                video_stem = event_dir / f"cam_{camera_id}"
                csv_path = event_dir / f"cam_{camera_id}.csv"
                json_path = event_dir / f"cam_{camera_id}.json"
                camera = next(camera for camera in self.cameras if camera.stable_id == stable_id)
                writer = gst_mjpeg.create_aligned_video_writer(
                    video_stem, camera.width, camera.height, self.fps
                )
                video_path = writer.path
                csv_handle = csv_path.open("w", newline="", encoding="utf-8")
                camera_writer = csv.writer(csv_handle)
                camera_writer.writerow(
                    [
                        "sync_index",
                        "target_unix_time",
                        "source_frame_index",
                        "capture_unix_time",
                        "time_error_sec",
                        "valid",
                        "present",
                        "bbox",
                        "axis",
                    ]
                )
                try:
                    for sync_index, (target, row) in enumerate(zip(targets, aligned)):
                        packet, error, valid, visible = row[stable_id]
                        writer.write(packet)
                        values = [
                            sync_index,
                            f"{target:.6f}",
                            packet["source_frame_index"],
                            f"{packet['capture_unix_time']:.6f}",
                            f"{error:.6f}",
                            int(valid),
                            int(visible),
                            packet["bbox"],
                            packet["axis"],
                        ]
                        camera_writer.writerow(values)
                        timestamp_writer.writerow(
                            [sync_index, values[1], camera_id, role] + values[2:]
                        )
                finally:
                    writer.close()
                    csv_handle.close()
                metadata = {
                    "video": video_path.name,
                    "frames": len(targets),
                    "start_time": datetime.fromtimestamp(targets[0]).astimezone().isoformat(timespec="milliseconds"),
                }
                json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        finally:
            timestamp_handle.close()
        print(
            f"wrote aligned event {event_dir}: {len(selected_ids)} cameras, "
            f"{len(targets)} frames at {self.fps} FPS",
            flush=True,
        )

    def _writer_loop(self):
        while True:
            payload = self.writer_queue.get()
            try:
                if payload is None:
                    return
                self._write_event(payload)
            except Exception as exc:
                print(f"aligned event writer failed: {exc}", flush=True)
            finally:
                self.writer_queue.task_done()

    def close(self):
        self._finish_event()
        self.writer_queue.put(None)
        self.writer_thread.join(timeout=60.0)
        for stable_id, missed in self.capture_misses.items():
            if missed:
                print(f"{self.roles[stable_id]}: lost {missed} compressed packets before draining", flush=True)


def rig_is_active(args, active_cameras, camera_count):
    if args.sync_mode == "any":
        return bool(active_cameras)
    if args.sync_mode == "min-cameras":
        return len(active_cameras) >= args.min_active_cameras
    return len(active_cameras) == camera_count and camera_count > 0


def draw_motion_overlay_display(frame, camera, result, source_shape, capture_fps, thresholds):
    cv2 = grid.cv2
    source_height, source_width = source_shape[:2]
    scale_x = frame.shape[1] / max(source_width, 1)
    scale_y = frame.shape[0] / max(source_height, 1)

    if result["roi"] is not None:
        x, y, width, height = result["roi"]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + width) * scale_x))
        y2 = int(round((y + height) * scale_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
    if result["bbox"] is not None:
        x, y, width, height = result["bbox"]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + width) * scale_x))
        y2 = int(round((y + height) * scale_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
    if result["axis"] is not None:
        (x1, y1), (x2, y2) = result["axis"]
        start = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
        end = (int(round(x2 * scale_x)), int(round(y2 * scale_y)))
        cv2.line(frame, start, end, (255, 220, 40), 2, cv2.LINE_AA)
    return frame


def update_detectors(cameras, detectors, rotation, started_at, logger):
    active_cameras = []
    snapshots = []
    now_perf = time.perf_counter()
    now_unix = time.time()
    elapsed_sec = now_perf - started_at
    for camera, detector in zip(cameras, detectors):
        ok, frame, source_shape, fps, source_frame_index, capture_perf, capture_unix = camera.read_latest_packet(
            copy_frame=False
        )
        if not ok:
            snapshots.append(
                {
                    "camera": camera,
                    "ok": False,
                    "frame": None,
                    "source_shape": source_shape,
                    "fps": fps,
                    "result": detector.last,
                    "unix_time": now_unix,
                    "perf_time": now_perf,
                    "source_frame_index": source_frame_index,
                    "capture_perf_time": capture_perf,
                    "capture_unix_time": capture_unix,
                    "elapsed_sec": elapsed_sec,
                }
            )
            continue
        frame = grid.rotate_frame(frame, rotation)
        result = detector.update(frame, elapsed_sec)
        snapshots.append(
            {
                "camera": camera,
                "ok": True,
                "frame": frame,
                "source_shape": frame.shape,
                "fps": fps,
                "result": result,
                "unix_time": now_unix,
                "perf_time": now_perf,
                "source_frame_index": source_frame_index,
                "capture_perf_time": capture_perf,
                "capture_unix_time": capture_unix,
                "elapsed_sec": elapsed_sec,
            }
        )
        if result["present"]:
            active_cameras.append(camera)
        if result["just_activated"]:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} camera active {camera.device} "
                f"frame={result['frame_motion_ratio'] * 100:.1f}% "
                f"roi={result['roi_motion_ratio'] * 100:.1f}%",
                flush=True,
            )
            logger.write_camera_event(elapsed_sec, camera, result)
    return active_cameras, snapshots


def draw_assignment_bar(canvas, selected_camera, selected_role, pending_group=None):
    bar_height = 28
    if pending_group is None:
        status = "select tile: 1-9/0 | assign group: f/b/l/r/t/s then slot number | u clear"
    else:
        status = f"assign {pending_group}: press slot 1-{ROLE_GROUP_LIMITS[pending_group]}"
    if selected_camera is not None:
        status = f"selected {selected_camera.device}: {selected_role} | " + status
    bar = grid.np.zeros((bar_height, canvas.shape[1], 3), dtype=grid.np.uint8)
    grid.cv2.putText(bar, status, (8, 19), grid.cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, grid.cv2.LINE_AA)
    return grid.np.vstack([canvas, bar])


def make_motion_grid(
    snapshots,
    cols,
    display_height,
    rotation,
    no_overlay,
    source_aspect,
    thresholds,
    config,
    selected_index=None,
    pending_group=None,
    return_layout=False,
):
    np = grid.np
    tile_height = display_height
    tile_width = grid.tile_width_for(display_height, source_aspect, rotation)
    rows, cols = grid.grid_shape(len(snapshots), cols)
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)
    selected_camera = None
    selected_role = "unassigned"
    tile_rects = []

    for index, snapshot in enumerate(snapshots):
        camera = snapshot["camera"]
        row = index // cols
        col = index % cols
        x = col * tile_width
        y = row * tile_height
        tile_rects.append((x, y, x + tile_width, y + tile_height, index, camera))
        if snapshot["ok"]:
            frame = snapshot["frame"]
            source_shape = snapshot["source_shape"]
            displayed = grid.paste_letterboxed(canvas, frame, x, y, tile_width, tile_height)
            if not no_overlay:
                draw_motion_overlay_display(displayed, camera, snapshot["result"], source_shape, snapshot["fps"], thresholds)
        else:
            frame = grid.make_waiting_frame(camera.device, display_height, source_aspect)
            canvas[y : y + tile_height, x : x + tile_width] = grid.fit_to_tile(frame, tile_width, tile_height)
        if index == selected_index:
            selected_camera = camera
            selected_role = camera_role(config, camera)
            grid.cv2.rectangle(canvas, (x + 2, y + 2), (x + tile_width - 3, y + tile_height - 3), (0, 255, 255), 3)
    if return_layout:
        return canvas, tile_rects
    return canvas


def handle_assignment_key(key, cameras, selected_index, pending_group, config, config_path, fallback_roi, detectors):
    tile_keys = "1234567890"
    if key in (ord(char) for char in tile_keys):
        number = tile_keys.index(chr(key)) + 1
        if pending_group is not None:
            limit = ROLE_GROUP_LIMITS[pending_group]
            if number <= limit and selected_index is not None and selected_index < len(cameras):
                role = f"{pending_group}_{number}"
                camera = cameras[selected_index]
                config.setdefault("assignments", {})[camera.stable_id] = role
                detectors[selected_index].set_roi(role_roi(config, fallback_roi, role), reset_background=True)
                save_config(config_path, config)
                print(
                    f"assigned {camera.device} to {role}; ROI {role_roi_text(config, role, 'fallback')}",
                    flush=True,
                )
                pending_group = None
            else:
                print(f"{pending_group} only has slots 1-{limit}", flush=True)
            return selected_index, pending_group

        selected_index = number - 1
        if selected_index < len(cameras):
            camera = cameras[selected_index]
            print(f"selected tile {selected_index + 1}: {camera.device} [{camera.stable_id}]", flush=True)
        return selected_index, pending_group

    if key in ROLE_GROUP_KEYS:
        pending_group = ROLE_GROUP_KEYS[key]
        print(f"pending assignment group: {pending_group}; press slot 1-{ROLE_GROUP_LIMITS[pending_group]}", flush=True)
        return selected_index, pending_group

    if key == ord("u") and selected_index is not None and selected_index < len(cameras):
        camera = cameras[selected_index]
        config["assignments"].pop(camera.stable_id, None)
        detectors[selected_index].set_roi(fallback_roi, reset_background=True)
        save_config(config_path, config)
        print(f"cleared role for {camera.device}", flush=True)
        return selected_index, pending_group

    return selected_index, pending_group


def main():
    args = parse_args()
    if args.scan:
        grid.scan_devices()
        return

    grid.require_opencv()
    cv2 = grid.cv2
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    fallback_roi = parse_roi(args.roi)
    if not 0 < args.record_scale <= 1.0:
        raise ValueError("--record-scale must be in the range (0, 1]")
    if args.capture_backend == "gstreamer" and args.rotation != 0:
        raise ValueError("GStreamer MJPEG pass-through requires --rotation 0 so calibration pixels remain unchanged")
    if args.capture_backend == "gstreamer" and args.record_scale != 1.0:
        raise ValueError("GStreamer MJPEG pass-through records full resolution; use --record-scale 1")
    if not 0 < args.axis_smoothing <= 1.0:
        raise ValueError("--axis-smoothing must be in the range (0, 1]")
    if args.presence_hold_sec < 0:
        raise ValueError("--presence-hold-sec must be non-negative")
    if args.min_active_cameras < 1:
        raise ValueError("--min-active-cameras must be at least 1")
    config = load_config(args.config)
    apply_role_roi_overrides(config, args.role_roi)
    if args.role_roi:
        save_config(args.config, config)
    if args.devices is None:
        args.devices = grid.auto_select_devices(args.width, args.height, args.fps, args.fourcc, args.camera_count)

    if not args.devices:
        print("No cameras selected.", flush=True)
        print("Run: python ten_v4l2_motion_detector.py --scan", flush=True)
        return

    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.devices))))
    source_aspect = args.width / max(args.height, 1)
    if args.capture_backend == "gstreamer":
        cameras = [
            gst_mjpeg.GStreamerMJPEGCamera(
                device,
                args.width,
                args.height,
                args.fps,
                args.fourcc,
                args.encoded_buffer_frames,
            )
            for device in args.devices
        ]
    else:
        cameras = [
            grid.V4L2Camera(device, args.width, args.height, args.fps, args.fourcc)
            for device in args.devices
        ]
    detectors = [
        LowComputeMotionDetector(args, role_roi(config, fallback_roi, camera_role(config, camera)))
        for camera in cameras
    ]
    logger = EventLogger(args.event_log)
    if args.capture_backend == "gstreamer":
        record_fps = args.record_fps if args.record_fps > 0 else args.fps
        recording_manager = (
            CompressedRigRecordingManager(
                cameras,
                config,
                args.record_dir,
                record_fps,
                args.pre_roll_sec,
                args.post_roll_sec,
                args.max_clip_sec,
                args.min_active_cameras,
            )
            if args.record_events
            else None
        )
    else:
        record_fps = args.record_fps if args.record_fps > 0 else args.display_fps if args.display_fps > 0 else args.fps
        recording_manager = (
            MotionRecordingManager(
                cameras,
                config,
                args.record_dir,
                record_fps,
                args.pre_roll_sec,
                args.post_roll_sec,
                args.max_clip_sec,
                args.csv_flush_sec,
                args.record_queue_frames,
                args.record_scale,
            )
            if args.record_events
            else None
        )
    window_sized = False
    stop_at = None
    display_period = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    started_at = time.perf_counter()
    rig_was_active = False
    thresholds = {
        "frame": args.min_frame_motion_ratio,
        "roi": args.min_roi_motion_ratio,
        "frames": args.motion_frames,
    }
    selected_index = None
    pending_group = None
    tile_rects = []
    displayed_image_shape = [0, 0]

    try:
        for camera in cameras:
            camera.start()

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.has_frame() for camera in cameras):
            time.sleep(0.02)
        stop_at = time.perf_counter() + args.duration if args.duration else None

        print(
            f"motion ROI={args.roi}; frame>={args.min_frame_motion_ratio * 100:g}%; "
            f"ROI>={args.min_roi_motion_ratio * 100:g}%; "
            f"frames>={args.motion_frames}; sync={args.sync_mode}; "
            f"min cameras={args.min_active_cameras}; presence hold={args.presence_hold_sec:g}s",
            flush=True,
        )
        print(f"role config: {args.config}", flush=True)
        if recording_manager is not None:
            print(f"motion clips: {args.record_dir}", flush=True)
        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            print("q/Esc: quit; 1-9/0 select; f/b/l/r/t/s then slot number assigns numbered role; u clear", flush=True)

            def handle_mouse(event, x, y, _flags, _userdata):
                nonlocal selected_index, pending_group
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                image_height, image_width = displayed_image_shape
                candidate_points = [(x, y)]
                if image_width > 0 and image_height > 0:
                    candidate_points.append(grid.map_window_to_image_point(WINDOW_NAME, x, y, image_width, image_height))
                for point_x, point_y in candidate_points:
                    for x1, y1, x2, y2, index, camera in tile_rects:
                        if x1 <= point_x <= x2 and y1 <= point_y <= y2:
                            selected_index = index
                            pending_group = None
                            print(f"clicked tile {index + 1}: {camera.device} [{camera.stable_id}]", flush=True)
                            return
                print(f"click missed camera tile at ({x}, {y})", flush=True)

            cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

        while True:
            loop_started_at = time.perf_counter()
            active_cameras, snapshots = update_detectors(cameras, detectors, args.rotation, started_at, logger)
            rig_active = rig_is_active(args, active_cameras, len(cameras))
            if recording_manager is not None:
                recording_manager.update(snapshots, rig_active)
            if rig_active and not rig_was_active:
                elapsed_sec = time.perf_counter() - started_at
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} rig active "
                    f"{len(active_cameras)}/{len(cameras)} cameras: "
                    f"{' '.join(camera.device for camera in active_cameras)}",
                    flush=True,
                )
                logger.write_rig_event(elapsed_sec, active_cameras)
            rig_was_active = rig_active

            key = 255
            if not args.no_display:
                grid_frame, tile_rects = make_motion_grid(
                    snapshots,
                    cols,
                    args.display_height,
                    args.rotation,
                    args.no_overlay,
                    source_aspect,
                    thresholds,
                    config,
                    selected_index,
                    pending_group,
                    return_layout=True,
                )
                displayed_image_shape[:] = list(grid_frame.shape[:2])
                if not window_sized:
                    cv2.resizeWindow(WINDOW_NAME, grid_frame.shape[1], grid_frame.shape[0])
                    window_sized = True
                cv2.imshow(WINDOW_NAME, grid_frame)

            elapsed = time.perf_counter() - loop_started_at
            wait_ms = 1
            if display_period > 0:
                wait_ms = max(1, int(round((display_period - elapsed) * 1000)))
            if args.no_display:
                time.sleep(wait_ms / 1000.0)
            else:
                key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if not args.no_display and key != 255:
                selected_index, pending_group = handle_assignment_key(
                    key,
                    cameras,
                    selected_index,
                    pending_group,
                    config,
                    args.config,
                    fallback_roi,
                    detectors,
                )
            if stop_at is not None and time.perf_counter() >= stop_at:
                break
    except KeyboardInterrupt:
        print("stopping motion recorder", flush=True)
    finally:
        for camera in cameras:
            camera.stop()
        if recording_manager is not None:
            recording_manager.close()
        logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
