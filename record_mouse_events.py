import argparse
import csv
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index, fit_to_tile, resize_to_height


CONFIG_PATH = Path("dual_arducam_viewer_config.json")
WINDOW_NAME = "Mouse Event Recorder"
POSITION_ORDER = ("left_1", "left_2", "right_1", "right_2")
POSITION_BUTTONS = (
    ("left_1", "L1"),
    ("left_2", "L2"),
    ("right_1", "R1"),
    ("right_2", "R2"),
)
CONTROL_BAR_HEIGHT = 38


def parse_args():
    parser = argparse.ArgumentParser(description="Record timestamped raw camera clips when a mouse is visible.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cameras", type=int, nargs="+", default=None)
    parser.add_argument("--format", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("mouse_events"))
    parser.add_argument("--pre-roll-sec", type=float, default=1.0)
    parser.add_argument("--post-roll-sec", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--display-height", type=int, default=360)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. 0 runs until q/Esc.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--liberal-foreground", action="store_true", default=True, help="Accept any large foreground blob in the ROI; only size/ROI filters reject it.")
    parser.add_argument("--strict-object-filter", dest="liberal_foreground", action="store_false", help="Use shape and motion-support filters before accepting a blob.")
    parser.add_argument("--detector-scale", type=float, default=0.4)
    parser.add_argument("--roi-x-min", type=float, default=0.05, help="Ignore pixels left of this normalized x coordinate.")
    parser.add_argument("--roi-x-max", type=float, default=0.95, help="Ignore pixels right of this normalized x coordinate.")
    parser.add_argument("--roi-y-min", type=float, default=0.55, help="Ignore pixels above this normalized y coordinate.")
    parser.add_argument("--min-floor-overlap", type=float, default=0.30, help="Candidate box must overlap this fraction of the active lower ROI.")
    parser.add_argument("--threshold", type=int, default=12)
    parser.add_argument("--min-area", type=float, default=40.0)
    parser.add_argument("--min-box-area", type=float, default=1500.0, help="Reject tiny full-resolution boxes, useful for ignoring individual tags.")
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--min-solidity", type=float, default=0.35, help="Reject very hollow/noisy contours below this solidity.")
    parser.add_argument("--min-extent", type=float, default=0.20, help="Reject contours that fill too little of their bounding box.")
    parser.add_argument("--min-aspect", type=float, default=0.18, help="Reject very thin contours below this width/height ratio.")
    parser.add_argument("--max-aspect", type=float, default=5.5, help="Reject very thin contours above this width/height ratio.")
    parser.add_argument("--min-object-score", type=float, default=0.30, help="Object-likeness score required for a green detection.")
    parser.add_argument("--motion-threshold", type=int, default=8, help="Frame-to-frame difference threshold for real motion support.")
    parser.add_argument("--min-motion-support", type=float, default=0.006, help="Reject blobs with too little recent frame-to-frame motion inside the box.")
    parser.add_argument("--history-frames", type=int, default=1, help="Detection confidence smoothing window per camera.")
    parser.add_argument("--history-score", type=float, default=0.01, help="Average confidence required over the smoothing window.")
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=13)
    parser.add_argument("--bg-alpha", type=float, default=0.02)
    parser.add_argument("--warmup-sec", type=float, default=4.0)
    parser.add_argument("--background-alpha", type=float, default=0.20, help="Fast background-learning rate during warmup/reset.")
    parser.add_argument("--camera-queue-frames", type=int, default=180, help="Per-camera frame queue before dropping oldest frames.")
    parser.add_argument("--detection-every", type=int, default=1, help="Run motion detection every N captured frames per camera.")
    parser.add_argument("--trigger-frames", type=int, default=1, help="Consecutive detections needed to start/extend recording.")
    parser.add_argument(
        "--trigger-mode",
        choices=("both-sides", "any"),
        default="any",
        help="Start an event from one left camera plus one right camera, or from any single camera.",
    )
    parser.add_argument("--record-all-on-any", action="store_true", help="Once an event starts, record every camera instead of only the triggering cameras.")
    return parser.parse_args()


def load_config(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_config_defaults(args):
    config = load_config(args.config)
    if args.cameras is None:
        args.cameras = [int(camera_id) for camera_id in config.get("cameras", [0, 1, 2, 3])]
    if args.format is None:
        args.format = config.get("format", "MJPG")
    if args.width is None:
        args.width = int(config.get("width", 1280))
    if args.height is None:
        args.height = int(config.get("height", 800))
    if args.rotation is None:
        args.rotation = int(config.get("rotation", 180))
    positions = {
        position: int(camera_id)
        for position, camera_id in config.get("positions", {}).items()
    }
    labels = {camera_id: position for position, camera_id in positions.items()}
    return config, labels


def save_config(args, labels):
    config = load_config(args.config)
    positions = {
        position: int(camera_id)
        for camera_id, position in labels.items()
        if position in POSITION_ORDER
    }
    config.update(
        {
            "cameras": [int(camera_id) for camera_id in args.cameras],
            "positions": positions,
            "format": args.format,
            "width": int(args.width),
            "height": int(args.height),
            "rotation": int(args.rotation),
        }
    )
    args.config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved camera assignment config: {args.config}", flush=True)


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def camera_side(camera_id, labels):
    label = labels.get(camera_id, "")
    if label.startswith("left"):
        return "left"
    if label.startswith("right"):
        return "right"
    return None


def trigger_camera_ids(consecutive_hits, labels, trigger_frames, trigger_mode):
    ready = {
        camera_id
        for camera_id, hits in consecutive_hits.items()
        if hits >= trigger_frames
    }
    if trigger_mode == "any":
        return ready

    left_ready = {
        camera_id
        for camera_id in ready
        if camera_side(camera_id, labels) == "left"
    }
    right_ready = {
        camera_id
        for camera_id in ready
        if camera_side(camera_id, labels) == "right"
    }
    if left_ready and right_ready:
        return left_ready | right_ready
    return set()


def odd_kernel(size):
    size = int(size)
    if size <= 0:
        return None
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class QueuedDShowCamera:
    def __init__(self, device_index, format_index, queue_frames):
        self.device_index = device_index
        self.format_index = format_index
        self.graph = FilterGraph()
        self.frames = queue.Queue(maxsize=max(1, int(queue_frames)))
        self.latest_frame = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.started_at = None
        self.last_frame_at = None
        self.running = False
        self.thread = None

    def start(self):
        self.graph.add_video_input_device(self.device_index)
        self.graph.get_input_device().set_format(self.format_index)
        self.graph.add_sample_grabber(self._on_frame)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.started_at = time.perf_counter()
        self.running = True
        self.thread = threading.Thread(
            target=self._request_loop,
            name=f"queued-dshow-camera-{self.device_index}",
            daemon=True,
        )
        self.thread.start()

    def _on_frame(self, frame):
        now_perf = time.perf_counter()
        record = {
            "frame": frame.copy(),
            "unix_time": time.time(),
            "perf_time": now_perf,
            "source_frame_index": self.frame_count,
        }
        self.latest_frame = record["frame"]
        self.frame_count += 1
        self.last_frame_at = now_perf
        try:
            self.frames.put_nowait(record)
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(record)
            except queue.Full:
                self.dropped_frames += 1

    def _request_loop(self):
        while self.running:
            self.graph.grab_frame()
            time.sleep(0.001)

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return self.frame_count / elapsed

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()


class MotionDetector:
    def __init__(self, args):
        self.args = args
        self.backgrounds = {}
        self.previous_gray = {}
        self.open_kernel = odd_kernel(args.open_kernel)
        self.close_kernel = odd_kernel(args.close_kernel)

    def detect(self, camera_id, frame, allow_update=True, force_background_update=False):
        scale = self.args.detector_scale
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        bg = self.backgrounds.get(camera_id)
        if bg is None:
            self.backgrounds[camera_id] = gray.astype(np.float32)
            self.previous_gray[camera_id] = gray.copy()
            return self.empty_detection(gray.shape)

        if force_background_update:
            cv2.accumulateWeighted(gray, bg, self.args.background_alpha)
            self.previous_gray[camera_id] = gray.copy()
            detection = self.empty_detection(gray.shape)
            detection["status"] = "learning_bg"
            detection["reason"] = "background_update"
            detection["roi_y_min"] = self.args.roi_y_min
            detection["roi_x_min"] = self.args.roi_x_min
            detection["roi_x_max"] = self.args.roi_x_max
            return detection

        bg_u8 = cv2.convertScaleAbs(bg)
        diff = cv2.absdiff(gray, bg_u8)
        roi_x1 = int(round(gray.shape[1] * self.args.roi_x_min))
        roi_x2 = int(round(gray.shape[1] * self.args.roi_x_max))
        roi_x1 = max(0, min(gray.shape[1] - 1, roi_x1))
        roi_x2 = max(roi_x1 + 1, min(gray.shape[1], roi_x2))
        roi_y = int(round(gray.shape[0] * self.args.roi_y_min))
        diff[:roi_y, :] = 0
        diff[:, :roi_x1] = 0
        diff[:, roi_x2:] = 0
        _, mask = cv2.threshold(diff, self.args.threshold, 255, cv2.THRESH_BINARY)

        prev = self.previous_gray.get(camera_id)
        if prev is None:
            motion_mask = np.zeros_like(gray)
        else:
            frame_delta = cv2.absdiff(gray, prev)
            frame_delta[:roi_y, :] = 0
            frame_delta[:, :roi_x1] = 0
            frame_delta[:, roi_x2:] = 0
            _, motion_mask = cv2.threshold(frame_delta, self.args.motion_threshold, 255, cv2.THRESH_BINARY)
            if self.open_kernel is not None:
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.open_kernel)
            if self.close_kernel is not None:
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.close_kernel)
        self.previous_gray[camera_id] = gray.copy()

        if self.open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        if self.close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = gray.shape[0] * gray.shape[1] * self.args.max_area_ratio
        raw_contours = 0
        rejected_tiny = 0
        rejected_area = 0
        rejected_roi = 0
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                raw_contours += 1
            if area < self.args.min_area * scale * scale or area > max_area:
                rejected_area += 1
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi_overlap_w = max(0, min(x + w, roi_x2) - max(x, roi_x1))
            roi_overlap_h = max(0, min(y + h, gray.shape[0]) - max(y, roi_y))
            roi_overlap = (roi_overlap_w * roi_overlap_h) / max(w * h, 1)
            floor_overlap = roi_overlap_h / max(h, 1)
            if floor_overlap < self.args.min_floor_overlap:
                rejected_roi += 1
                continue
            full_box_area = (w / scale) * (h / scale)
            if full_box_area < self.args.min_box_area:
                rejected_tiny += 1
                continue
            bbox_area = max(w * h, 1)
            motion_support = cv2.countNonZero(motion_mask[y : y + h, x : x + w]) / bbox_area
            aspect = w / max(h, 1)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1.0)
            solidity = area / hull_area
            extent = area / bbox_area
            center_y = y + h * 0.5
            roi_depth = max(gray.shape[0] - roi_y, 1)
            floor_bias = min(1.0, max(0.0, (center_y - roi_y) / roi_depth))
            aspect_score = min(1.0, aspect / max(self.args.min_aspect, 1e-6), self.args.max_aspect / max(aspect, 1e-6))
            solidity_score = min(1.0, solidity / max(self.args.min_solidity, 1e-6))
            extent_score = min(1.0, extent / max(self.args.min_extent, 1e-6))
            area_score = min(1.0, area / max(self.args.min_area * scale * scale * 8.0, 1.0))
            motion_score = min(1.0, motion_support / max(self.args.min_motion_support, 1e-6))
            if self.args.liberal_foreground:
                object_score = max(0.01, 0.55 * area_score + 0.25 * floor_bias + 0.20 * min(1.0, full_box_area / 20000.0))
            else:
                object_score = (
                    0.28 * motion_score
                    + 0.22 * solidity_score
                    + 0.18 * extent_score
                    + 0.14 * aspect_score
                    + 0.10 * area_score
                    + 0.08 * floor_bias
                )
            reason = "ok"
            if not self.args.liberal_foreground and motion_support < self.args.min_motion_support:
                reason = "no_motion_support"
            elif not self.args.liberal_foreground and solidity < self.args.min_solidity:
                reason = "low_solidity"
            elif not self.args.liberal_foreground and extent < self.args.min_extent:
                reason = "low_extent"
            elif not self.args.liberal_foreground and (aspect < self.args.min_aspect or aspect > self.args.max_aspect):
                reason = "bad_aspect"
            elif not self.args.liberal_foreground and object_score < self.args.min_object_score:
                reason = "low_score"
            if reason != "ok":
                candidates.append((object_score, area, (x, y, w, h), contour, reason, aspect, solidity, extent, motion_support))
                continue
            candidates.append((object_score, area, (x, y, w, h), contour, reason, aspect, solidity, extent, motion_support))

        detected = bool(candidates)
        accepted = [candidate for candidate in candidates if candidate[4] == "ok"]
        if allow_update and not accepted:
            cv2.accumulateWeighted(gray, bg, self.args.bg_alpha)

        if not candidates:
            detection = self.empty_detection(gray.shape)
            detection["mask"] = mask
            detection["raw_motion"] = raw_contours > 0
            detection["raw_contours"] = raw_contours
            detection["roi_y_min"] = self.args.roi_y_min
            detection["roi_x_min"] = self.args.roi_x_min
            detection["roi_x_max"] = self.args.roi_x_max
            detection["rejected_area"] = rejected_area
            detection["rejected_roi"] = rejected_roi
            detection["rejected_tiny"] = rejected_tiny
            detection["status"] = "motion" if raw_contours else "idle"
            if rejected_area:
                reason = "area_reject"
            elif rejected_roi:
                reason = "roi_reject"
            elif rejected_tiny:
                reason = "tiny_box"
            else:
                reason = "no_candidate" if raw_contours else "no_motion"
            detection["reason"] = reason
            return detection

        best_pool = accepted or candidates
        object_score, area, bbox, contour, reason, aspect, solidity, extent, motion_support = max(best_pool, key=lambda item: item[0])
        x, y, w, h = bbox
        full_bbox = [
            int(round(x / scale)),
            int(round(y / scale)),
            int(round(w / scale)),
            int(round(h / scale)),
        ]
        score = float(area / max(gray.shape[0] * gray.shape[1], 1))
        detection = {
            "detected": bool(accepted),
            "raw_motion": raw_contours > 0,
            "raw_contours": raw_contours,
            "score": score,
            "object_score": float(object_score),
            "area": float(area / (scale * scale)),
            "aspect": float(aspect),
            "solidity": float(solidity),
            "extent": float(extent),
            "motion_support": float(motion_support),
            "roi_y_min": self.args.roi_y_min,
            "roi_x_min": self.args.roi_x_min,
            "roi_x_max": self.args.roi_x_max,
            "rejected_area": rejected_area,
            "rejected_roi": rejected_roi,
            "rejected_tiny": rejected_tiny,
            "bbox": full_bbox,
            "mask": mask,
            "status": "object" if accepted else "motion",
            "reason": reason,
        }
        return detection

    @staticmethod
    def empty_detection(_shape):
        return {
            "detected": False,
            "raw_motion": False,
            "raw_contours": 0,
            "score": 0.0,
            "object_score": 0.0,
            "area": 0.0,
            "aspect": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "motion_support": 0.0,
            "roi_y_min": 0.0,
            "roi_x_min": 0.0,
            "roi_x_max": 1.0,
            "rejected_area": 0,
            "rejected_roi": 0,
            "rejected_tiny": 0,
            "bbox": None,
            "mask": None,
            "status": "idle",
            "reason": "init",
            "history_score": 0.0,
            "history_hits": 0,
        }


class CameraEventRecorder:
    def __init__(self, camera_id, label, output_dir, fps, pre_roll_sec, post_roll_sec):
        self.camera_id = camera_id
        self.label = label or f"cam_{camera_id}"
        self.output_dir = output_dir
        self.fps = fps
        self.pre_roll_frames = max(1, int(round(pre_roll_sec * fps)))
        self.post_roll_sec = post_roll_sec
        self.buffer = deque(maxlen=self.pre_roll_frames)
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.event_dir = None
        self.frame_index = 0
        self.active_until = 0.0
        self.event_count = 0
        self.lock = threading.RLock()

    def is_recording(self):
        with self.lock:
            return self.writer is not None

    def add_to_buffer(self, frame_record):
        with self.lock:
            self.buffer.append(frame_record)

    def trigger(self, now, frame_shape):
        with self.lock:
            self.active_until = max(self.active_until, now + self.post_roll_sec)
            if self.writer is not None:
                return False

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self.event_count += 1
            self.event_dir = self.output_dir / f"event_{stamp}_{self.label}_cam_{self.camera_id}"
            self.event_dir.mkdir(parents=True, exist_ok=True)
            video_path = self.event_dir / f"{self.label}_cam_{self.camera_id}.mp4"
            csv_path = self.event_dir / f"{self.label}_cam_{self.camera_id}_frames.csv"
            height, width = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
            self.csv_file = csv_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "frame_index",
                    "source_frame_index",
                    "unix_time",
                    "perf_time",
                    "camera_id",
                    "label",
                    "detected",
                    "instant_detected",
                    "status",
                    "reason",
                    "score",
                    "object_score",
                    "history_score",
                    "history_hits",
                    "area",
                    "aspect",
                    "solidity",
                    "extent",
                    "motion_support",
                    "raw_contours",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                ],
            )
            self.csv_writer.writeheader()
            self.frame_index = 0
            metadata = {
                "camera_id": self.camera_id,
                "label": self.label,
                "video": video_path.name,
                "frames_csv": csv_path.name,
                "fps": self.fps,
                "started_at_unix": time.time(),
                "recorder": "threaded_per_camera_queue",
            }
            (self.event_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            for record in list(self.buffer):
                self.write(record)
            return True

    def write_if_active(self, now, frame_record):
        with self.lock:
            if self.writer is None:
                return
            if now <= self.active_until:
                self.write(frame_record)
                return
            self.close()

    def write(self, frame_record):
        frame = frame_record["frame"]
        detection = frame_record["detection"]
        self.writer.write(frame)
        bbox = detection.get("bbox") or ["", "", "", ""]
        self.csv_writer.writerow(
            {
                "frame_index": self.frame_index,
                "source_frame_index": frame_record.get("source_frame_index", ""),
                "unix_time": f"{frame_record['unix_time']:.6f}",
                "perf_time": f"{frame_record['perf_time']:.6f}",
                "camera_id": self.camera_id,
                "label": self.label,
                "detected": int(bool(detection.get("detected"))),
                "instant_detected": int(bool(detection.get("instant_detected", detection.get("detected")))),
                "status": detection.get("status", ""),
                "reason": detection.get("reason", ""),
                "score": f"{float(detection.get('score', 0.0)):.6f}",
                "object_score": f"{float(detection.get('object_score', 0.0)):.6f}",
                "history_score": f"{float(detection.get('history_score', 0.0)):.6f}",
                "history_hits": detection.get("history_hits", 0),
                "area": f"{float(detection.get('area', 0.0)):.2f}",
                "aspect": f"{float(detection.get('aspect', 0.0)):.4f}",
                "solidity": f"{float(detection.get('solidity', 0.0)):.4f}",
                "extent": f"{float(detection.get('extent', 0.0)):.4f}",
                "motion_support": f"{float(detection.get('motion_support', 0.0)):.4f}",
                "raw_contours": detection.get("raw_contours", 0),
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
            }
        )
        self.frame_index += 1

    def close(self):
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            if self.csv_file is not None:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None


class CameraWorker(threading.Thread):
    def __init__(
        self,
        camera,
        recorder,
        args,
        warmup_until,
        stop_event,
        state_lock,
        detection_state,
        latest_for_display,
        background_learning,
    ):
        super().__init__(name=f"camera-worker-{camera.device_index}", daemon=True)
        self.camera = camera
        self.recorder = recorder
        self.args = args
        self.warmup_until = warmup_until
        self.stop_event = stop_event
        self.state_lock = state_lock
        self.detection_state = detection_state
        self.latest_for_display = latest_for_display
        self.background_learning = background_learning
        self.detector = MotionDetector(args)
        self.last_detection = MotionDetector.empty_detection((0, 0))
        self.processed_frames = 0
        self.consecutive_hits = 0
        self.confidence_history = deque(maxlen=max(1, int(args.history_frames)))

    def run(self):
        while not self.stop_event.is_set():
            try:
                source_record = self.camera.frames.get(timeout=0.05)
            except queue.Empty:
                continue

            self.processed_frames += 1
            frame = rotate_frame(source_record["frame"], self.args.rotation)
            should_detect = self.processed_frames % max(1, self.args.detection_every) == 0

            if should_detect:
                with self.state_lock:
                    learning_until = self.background_learning.get("until", self.warmup_until)
                force_background_update = source_record["perf_time"] < learning_until
                allow_bg_update = (
                    not force_background_update
                    and not self.recorder.is_recording()
                )
                detection = self.detector.detect(
                    self.camera.device_index,
                    frame,
                    allow_update=allow_bg_update,
                    force_background_update=force_background_update,
                )
                if force_background_update:
                    detection["detected"] = False
                    detection["status"] = "learning_bg"
                instant_confidence = detection.get("object_score", 0.0) if detection["detected"] else 0.0
                self.confidence_history.append(float(instant_confidence))
                history_score = sum(self.confidence_history) / max(len(self.confidence_history), 1)
                history_hits = sum(1 for value in self.confidence_history if value >= self.args.min_object_score)
                detection["instant_detected"] = bool(detection["detected"])
                detection["history_score"] = float(history_score)
                detection["history_hits"] = int(history_hits)
                required_history = min(self.args.trigger_frames, self.confidence_history.maxlen)
                detection["detected"] = (
                    len(self.confidence_history) >= max(1, required_history)
                    and history_score >= self.args.history_score
                    and history_hits >= max(1, required_history)
                )
                if detection["detected"]:
                    detection["status"] = "ready"
                    self.consecutive_hits += 1
                else:
                    self.consecutive_hits = 0
                self.last_detection = detection
            else:
                detection = self.last_detection

            frame_record = {
                "frame": frame,
                "unix_time": source_record["unix_time"],
                "perf_time": source_record["perf_time"],
                "source_frame_index": source_record["source_frame_index"],
                "detection": detection,
            }
            self.recorder.add_to_buffer(frame_record)
            self.recorder.write_if_active(source_record["perf_time"], frame_record)

            with self.state_lock:
                self.detection_state[self.camera.device_index] = {
                    "consecutive_hits": self.consecutive_hits,
                    "latest_shape": frame.shape,
                    "processed_frames": self.processed_frames,
                    "queued_frames": self.camera.frames.qsize(),
                    "dropped_frames": self.camera.dropped_frames,
                    "camera_fps": self.camera.average_fps(),
                }
                self.latest_for_display[self.camera.device_index] = (
                    frame,
                    detection,
                    self.camera.average_fps(),
                )


def draw_detection(frame, label, detection, is_recording, fps):
    shown = frame.copy()
    status = "RECORDING" if is_recording else detection.get("status", "idle")
    if is_recording:
        color = (0, 255, 0)
    elif detection.get("detected"):
        color = (0, 255, 0)
    elif detection.get("raw_motion"):
        color = (0, 220, 255)
    else:
        color = (180, 180, 180)
    text = (
        f"{label} | {status} | obj {detection.get('object_score', 0.0):.2f} "
        f"hist {detection.get('history_score', 0.0):.2f}/{detection.get('history_hits', 0)} | {fps:.1f} FPS"
    )
    reason = (
        f"{detection.get('reason', '')} | area {detection.get('area', 0.0):.0f} "
        f"asp {detection.get('aspect', 0.0):.2f} sol {detection.get('solidity', 0.0):.2f} "
        f"ext {detection.get('extent', 0.0):.2f} mot {detection.get('motion_support', 0.0):.3f} "
        f"rej a/r/t {detection.get('rejected_area', 0)}/{detection.get('rejected_roi', 0)}/{detection.get('rejected_tiny', 0)}"
    )
    cv2.rectangle(shown, (4, 4), (min(shown.shape[1] - 4, 980), 92), (0, 0, 0), -1)
    cv2.putText(shown, text, (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.88, color, 2, cv2.LINE_AA)
    cv2.putText(shown, reason, (14, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
    if detection["bbox"] is not None:
        x, y, w, h = detection["bbox"]
        cv2.rectangle(shown, (x, y), (x + w, y + h), color, 2)
    roi_y = int(round(frame.shape[0] * float(detection.get("roi_y_min", 0.55))))
    roi_x1 = int(round(frame.shape[1] * float(detection.get("roi_x_min", 0.05))))
    roi_x2 = int(round(frame.shape[1] * float(detection.get("roi_x_max", 0.95))))
    shaded = shown.copy()
    cv2.rectangle(shaded, (0, 0), (shown.shape[1] - 1, max(0, roi_y - 1)), (0, 0, 0), -1)
    cv2.rectangle(shaded, (0, roi_y), (max(0, roi_x1 - 1), shown.shape[0] - 1), (0, 0, 0), -1)
    cv2.rectangle(shaded, (min(shown.shape[1] - 1, roi_x2), roi_y), (shown.shape[1] - 1, shown.shape[0] - 1), (0, 0, 0), -1)
    shown = cv2.addWeighted(shaded, 0.18, shown, 0.82, 0)
    cv2.line(shown, (0, roi_y), (shown.shape[1] - 1, roi_y), (0, 255, 255), 3)
    cv2.line(shown, (roi_x1, roi_y), (roi_x1, shown.shape[0] - 1), (0, 255, 255), 3)
    cv2.line(shown, (roi_x2, roi_y), (roi_x2, shown.shape[0] - 1), (0, 255, 255), 3)
    cv2.putText(shown, "ROI", (8, max(24, roi_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return shown


def draw_button(canvas, x, y, w, h, text, active=False):
    fill = (45, 95, 45) if active else (35, 35, 35)
    border = (120, 255, 120) if active else (210, 210, 210)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)
    cv2.putText(canvas, text, (x + 7, y + h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def make_grid(frames, labels, recorders, display_height, selected_camera=None, show_controls=False):
    order = sorted(frames.keys(), key=lambda camera_id: POSITION_ORDER.index(labels.get(camera_id, "")) if labels.get(camera_id, "") in POSITION_ORDER else 99)
    tiles = []
    tile_camera_ids = []
    for camera_id in order:
        frame, detection, fps = frames[camera_id]
        label = labels.get(camera_id, f"cam_{camera_id}")
        tile = draw_detection(frame, label, detection, recorders[camera_id].is_recording(), fps)
        if camera_id == selected_camera:
            cv2.rectangle(tile, (2, 2), (tile.shape[1] - 3, tile.shape[0] - 3), (0, 255, 0), 5)
        tiles.append(resize_to_height(tile, display_height))
        tile_camera_ids.append(camera_id)
    if not tiles:
        return None, [], []
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    tile_rects = []
    for start in range(0, len(tiles), 2):
        row_tiles = []
        row_index = start // 2
        for col_index, tile in enumerate(tiles[start : start + 2]):
            row_tiles.append(fit_to_tile(tile, tile_width, tile_height))
            camera_id = tile_camera_ids[start + col_index]
            x1 = col_index * tile_width
            y1 = row_index * tile_height
            tile_rects.append({"camera_id": camera_id, "rect": (x1, y1, x1 + tile_width, y1 + tile_height)})
        row = row_tiles
        if len(row) < 2:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    if not show_controls:
        return grid, tile_rects, []

    bar = np.zeros((CONTROL_BAR_HEIGHT, grid.shape[1], 3), dtype=np.uint8)
    selected_text = f"cam {selected_camera}" if selected_camera is not None else "click camera"
    cv2.putText(bar, selected_text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(bar, "assign", (96, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    button_rects = []
    x = 154
    for position_key, text in POSITION_BUTTONS:
        active = selected_camera is not None and labels.get(selected_camera) == position_key
        draw_button(bar, x, 5, 38, 27, text, active)
        button_rects.append({"action": "assign", "position": position_key, "rect": (x, grid.shape[0] + 5, x + 38, grid.shape[0] + 32)})
        x += 44
    draw_button(bar, x + 4, 5, 52, 27, "Save", False)
    button_rects.append({"action": "save", "rect": (x + 4, grid.shape[0] + 5, x + 56, grid.shape[0] + 32)})
    cv2.putText(bar, "any camera can trigger; all cameras record", (x + 68, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
    return np.vstack([grid, bar]), tile_rects, button_rects


def main():
    args = parse_args()
    _config, labels = apply_config_defaults(args)
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = session_dir
    (session_dir / "session_metadata.json").write_text(
        json.dumps(
            {
                "session": session_name,
                "started_at_unix": time.time(),
                "started_at_local": datetime.now().isoformat(timespec="seconds"),
                "recorder": "record_mouse_events.py",
                "cameras": [int(camera_id) for camera_id in args.cameras] if args.cameras else None,
                "trigger_mode": args.trigger_mode,
                "record_all_on_any": bool(args.record_all_on_any),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Recording session directory: {session_dir}", flush=True)

    cameras = []
    workers = []
    stop_event = threading.Event()
    state_lock = threading.RLock()
    detection_state = {}
    latest_for_display = {}
    background_learning = {"until": 0.0}
    selected_camera = {"id": None}
    ui_rects = {"tiles": [], "buttons": []}
    show_controls = {"value": False}
    recorders = {
        camera_id: CameraEventRecorder(
            camera_id,
            labels.get(camera_id, f"cam_{camera_id}"),
            args.output_dir,
            args.fps,
            args.pre_roll_sec,
            args.post_roll_sec,
        )
        for camera_id in args.cameras
    }

    try:
        print(
            f"Trigger mode: {args.trigger_mode}"
            f"{' | recording all cameras per event' if args.record_all_on_any else ' | recording triggering cameras only'}",
            flush=True,
        )
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = QueuedDShowCamera(camera_id, format_index, args.camera_queue_frames)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        warmup_until = time.perf_counter() + args.warmup_sec
        background_learning["until"] = warmup_until
        stop_at = time.perf_counter() + args.duration if args.duration > 0 else None
        for camera in cameras:
            worker = CameraWorker(
                camera,
                recorders[camera.device_index],
                args,
                warmup_until,
                stop_event,
                state_lock,
                detection_state,
                latest_for_display,
                background_learning,
            )
            worker.start()
            workers.append(worker)

        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            def handle_mouse(event, x, y, flags, userdata):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                for button in ui_rects["buttons"]:
                    x1, y1, x2, y2 = button["rect"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        if button["action"] == "save":
                            save_config(args, labels)
                            return
                        if button["action"] == "assign" and selected_camera["id"] is not None:
                            position = button["position"]
                            for camera_id, assigned_position in list(labels.items()):
                                if assigned_position == position:
                                    labels.pop(camera_id, None)
                            camera_id = selected_camera["id"]
                            labels[camera_id] = position
                            recorders[camera_id].label = position
                            print(f"Assigned cam {camera_id} -> {position}", flush=True)
                            return
                for tile in ui_rects["tiles"]:
                    x1, y1, x2, y2 = tile["rect"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_camera["id"] = tile["camera_id"]
                        print(f"Selected cam {selected_camera['id']}", flush=True)
                        return

            cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
            print("Controls: click camera then 1=L1 2=L2 3=R1 4=R2, s=save, b=learn background, a=toggle buttons, q=quit", flush=True)

        while True:
            now_perf = time.perf_counter()
            with state_lock:
                consecutive_hits = {
                    camera_id: state.get("consecutive_hits", 0)
                    for camera_id, state in detection_state.items()
                }
                latest_shapes = {
                    camera_id: state.get("latest_shape")
                    for camera_id, state in detection_state.items()
                    if state.get("latest_shape") is not None
                }
                display_snapshot = dict(latest_for_display)

            triggered_camera_ids = trigger_camera_ids(
                consecutive_hits,
                labels,
                args.trigger_frames,
                args.trigger_mode,
            )
            if triggered_camera_ids:
                recorder_ids = args.cameras if args.record_all_on_any else sorted(triggered_camera_ids)
                for camera_id in recorder_ids:
                    frame_shape = latest_shapes.get(camera_id)
                    if frame_shape is not None:
                        recorders[camera_id].trigger(now_perf, frame_shape)

            if not args.no_display and display_snapshot:
                grid, tile_rects, button_rects = make_grid(
                    display_snapshot,
                    labels,
                    recorders,
                    args.display_height,
                    selected_camera["id"],
                    show_controls["value"],
                )
                if grid is not None:
                    ui_rects["tiles"] = tile_rects
                    ui_rects["buttons"] = button_rects
                    cv2.imshow(WINDOW_NAME, grid)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    show_controls["value"] = not show_controls["value"]
                elif key == ord("b"):
                    with state_lock:
                        background_learning["until"] = time.perf_counter() + args.warmup_sec
                    print(f"Learning background for {args.warmup_sec:.1f}s. Keep fingers/object out of the enclosure.", flush=True)
                elif key == ord("s"):
                    save_config(args, labels)
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")) and selected_camera["id"] is not None:
                    position = {
                        ord("1"): "left_1",
                        ord("2"): "left_2",
                        ord("3"): "right_1",
                        ord("4"): "right_2",
                    }[key]
                    for camera_id, assigned_position in list(labels.items()):
                        if assigned_position == position:
                            labels.pop(camera_id, None)
                    camera_id = selected_camera["id"]
                    labels[camera_id] = position
                    recorders[camera_id].label = position
                    print(f"Assigned cam {camera_id} -> {position}", flush=True)

            if stop_at is not None and now_perf >= stop_at:
                break
            time.sleep(0.001)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
        for recorder in recorders.values():
            recorder.close()
        for camera in cameras:
            camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
