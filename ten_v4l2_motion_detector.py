#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path

import ten_v4l2_camera_grid as grid


WINDOW_NAME = "Ten V4L2 Motion Detector"


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
    parser.add_argument("--display-fps", type=float, default=20.0, help="Preview and detection FPS. Use 0 for uncapped.")
    parser.add_argument("--fourcc", default="MJPG", help="Requested V4L2 pixel format.")
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

    parser.add_argument("--detector-scale", type=float, default=0.25, help="Downscale factor used for detection.")
    parser.add_argument("--pixel-threshold", type=int, default=25, help="Per-pixel grayscale background difference threshold.")
    parser.add_argument("--background-alpha", type=float, default=0.02, help="Running background update rate.")
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
        default=0.02,
        help="Largest moving component must cover at least this fraction of the full camera frame.",
    )
    parser.add_argument(
        "--min-roi-motion-ratio",
        type=float,
        default=0.06,
        help="Moving pixels must cover at least this fraction of the ROI.",
    )
    parser.add_argument(
        "--motion-frames",
        type=int,
        default=3,
        help="Consecutive qualifying frames required before a camera is active.",
    )
    parser.add_argument(
        "--roi",
        default="0,0,1,1",
        help="ROI as x,y,w,h. Values <= 1 are fractions of the rotated frame. Default is the full frame.",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("all", "any", "min-cameras"),
        default="all",
        help="How camera detections form a synced rig event.",
    )
    parser.add_argument(
        "--min-active-cameras",
        type=int,
        default=2,
        help="Active camera count required when --sync-mode min-cameras is used.",
    )
    parser.add_argument(
        "--event-log",
        type=Path,
        default=None,
        help="Optional CSV path for camera and rig motion events.",
    )
    return parser.parse_args()


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


def scale_box(box, scale):
    x, y, width, height = box
    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(width / scale)),
        int(round(height / scale)),
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
        self.last = self.empty_result()

    def update(self, frame, elapsed_sec):
        cv2 = grid.cv2
        np = grid.np
        scale = self.args.detector_scale
        if not 0 < scale <= 1:
            raise ValueError("--detector-scale must be in the range (0, 1]")

        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_pixels(self.roi, gray.shape)
        roi_area = max(1, (roi_x2 - roi_x1) * (roi_y2 - roi_y1))
        frame_area = max(1, gray.shape[0] * gray.shape[1])

        if self.background is None:
            self.background = gray.astype(np.float32)
            self.last = self.empty_result(status="learning_bg", roi=(roi_x1, roi_y1, roi_x2, roi_y2))
            return self.last

        bg = cv2.convertScaleAbs(self.background)
        delta = cv2.absdiff(bg, gray)
        delta[:roi_y1, :] = 0
        delta[roi_y2:, :] = 0
        delta[:, :roi_x1] = 0
        delta[:, roi_x2:] = 0

        _, mask = cv2.threshold(delta, self.args.pixel_threshold, 255, cv2.THRESH_BINARY)
        if self.open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        if self.close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        roi_mask = mask[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_motion_pixels = cv2.countNonZero(roi_mask)
        roi_motion_ratio = roi_motion_pixels / roi_area

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        largest_area = 0
        largest_box = None
        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area > largest_area:
                largest_area = int(area)
                largest_box = (int(x), int(y), int(width), int(height))

        frame_motion_ratio = largest_area / frame_area
        qualifies = (
            elapsed_sec >= self.args.warmup_sec
            and frame_motion_ratio >= self.args.min_frame_motion_ratio
            and roi_motion_ratio >= self.args.min_roi_motion_ratio
        )
        if qualifies:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        was_active = self.active
        self.active = self.consecutive_hits >= self.args.motion_frames
        if elapsed_sec < self.args.warmup_sec:
            status = "learning_bg"
        elif self.active:
            status = "active"
        elif qualifies:
            status = "candidate"
        else:
            status = "idle"

        if not qualifies:
            cv2.accumulateWeighted(gray, self.background, self.args.background_alpha)

        self.last = {
            "active": self.active,
            "just_activated": self.active and not was_active,
            "qualifies": qualifies,
            "status": status,
            "consecutive_hits": self.consecutive_hits,
            "frame_motion_ratio": float(frame_motion_ratio),
            "roi_motion_ratio": float(roi_motion_ratio),
            "largest_area": int(round(largest_area / (scale * scale))),
            "bbox": scale_box(largest_box, scale) if largest_box is not None else None,
            "roi": scale_box(
                (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1),
                scale,
            ),
        }
        return self.last

    @staticmethod
    def empty_result(status="idle", roi=None):
        return {
            "active": False,
            "just_activated": False,
            "qualifies": False,
            "status": status,
            "consecutive_hits": 0,
            "frame_motion_ratio": 0.0,
            "roi_motion_ratio": 0.0,
            "largest_area": 0,
            "bbox": None,
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

    color = (0, 0, 255) if result["active"] else (0, 210, 255) if result["qualifies"] else (80, 220, 80)
    if result["roi"] is not None:
        x, y, width, height = result["roi"]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + width) * scale_x))
        y2 = int(round((y + height) * scale_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)

    if result["bbox"] is not None:
        x, y, width, height = result["bbox"]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + width) * scale_x))
        y2 = int(round((y + height) * scale_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = (
        f"{camera.device} {capture_fps:4.1f} FPS {result['status']} "
        f"{result['consecutive_hits']}/{thresholds['frames']} "
        f"F {result['frame_motion_ratio'] * 100:4.1f}%/{thresholds['frame'] * 100:g}% "
        f"ROI {result['roi_motion_ratio'] * 100:4.1f}%/{thresholds['roi'] * 100:g}%"
    )
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, text_width + 12), text_height + baseline + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, text_height + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    if result["active"]:
        cv2.rectangle(frame, (1, 1), (frame.shape[1] - 2, frame.shape[0] - 2), color, 3)
    return frame


def update_detectors(cameras, detectors, rotation, started_at, logger):
    active_cameras = []
    for camera, detector in zip(cameras, detectors):
        ok, frame, _source_shape, _fps = camera.read_latest(copy_frame=False)
        if not ok:
            continue
        frame = grid.rotate_frame(frame, rotation)
        elapsed_sec = time.perf_counter() - started_at
        result = detector.update(frame, elapsed_sec)
        if result["active"]:
            active_cameras.append(camera)
        if result["just_activated"]:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} camera active {camera.device} "
                f"frame={result['frame_motion_ratio'] * 100:.1f}% "
                f"roi={result['roi_motion_ratio'] * 100:.1f}%",
                flush=True,
            )
            logger.write_camera_event(elapsed_sec, camera, result)
    return active_cameras


def make_motion_grid(cameras, detectors, cols, display_height, rotation, no_overlay, source_aspect, thresholds):
    np = grid.np
    tile_height = display_height
    tile_width = grid.tile_width_for(display_height, source_aspect, rotation)
    rows, cols = grid.grid_shape(len(cameras), cols)
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    for index, (camera, detector) in enumerate(zip(cameras, detectors)):
        row = index // cols
        col = index % cols
        x = col * tile_width
        y = row * tile_height
        ok, frame, source_shape, fps = camera.read_latest(copy_frame=False)
        if ok:
            frame = grid.rotate_frame(frame, rotation)
            source_shape = frame.shape
            displayed = grid.paste_letterboxed(canvas, frame, x, y, tile_width, tile_height)
            if not no_overlay:
                draw_motion_overlay_display(displayed, camera, detector.last, source_shape, fps, thresholds)
        else:
            frame = grid.make_waiting_frame(camera.device, display_height, source_aspect)
            canvas[y : y + tile_height, x : x + tile_width] = grid.fit_to_tile(frame, tile_width, tile_height)
    return canvas


def main():
    args = parse_args()
    if args.scan:
        grid.scan_devices()
        return

    grid.require_opencv()
    cv2 = grid.cv2
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    roi = parse_roi(args.roi)
    if args.devices is None:
        args.devices = grid.auto_select_devices(args.width, args.height, args.fps, args.fourcc, args.camera_count)

    if not args.devices:
        print("No cameras selected.", flush=True)
        print("Run: python ten_v4l2_motion_detector.py --scan", flush=True)
        return

    cols = max(1, args.cols or math.ceil(math.sqrt(len(args.devices))))
    source_aspect = args.width / max(args.height, 1)
    cameras = [
        grid.V4L2Camera(device, args.width, args.height, args.fps, args.fourcc)
        for device in args.devices
    ]
    detectors = [LowComputeMotionDetector(args, roi) for _device in args.devices]
    logger = EventLogger(args.event_log)
    window_sized = False
    stop_at = time.perf_counter() + args.duration if args.duration else None
    display_period = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    started_at = time.perf_counter()
    rig_was_active = False
    thresholds = {
        "frame": args.min_frame_motion_ratio,
        "roi": args.min_roi_motion_ratio,
        "frames": args.motion_frames,
    }

    try:
        for camera in cameras:
            camera.start()

        deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < deadline and not all(camera.has_frame() for camera in cameras):
            time.sleep(0.02)

        print(
            f"motion ROI={args.roi}; frame>={args.min_frame_motion_ratio * 100:g}%; "
            f"ROI>={args.min_roi_motion_ratio * 100:g}%; "
            f"frames>={args.motion_frames}; sync={args.sync_mode}",
            flush=True,
        )
        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            print("q/Esc: quit", flush=True)

        while True:
            loop_started_at = time.perf_counter()
            active_cameras = update_detectors(cameras, detectors, args.rotation, started_at, logger)
            rig_active = rig_is_active(args, active_cameras, len(cameras))
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
                grid_frame = make_motion_grid(
                    cameras,
                    detectors,
                    cols,
                    args.display_height,
                    args.rotation,
                    args.no_overlay,
                    source_aspect,
                    thresholds,
                )
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
            if stop_at is not None and time.perf_counter() >= stop_at:
                break
    finally:
        for camera in cameras:
            camera.stop()
        logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
