#!/usr/bin/env python3
import argparse
import math
import time

import ten_v4l2_camera_grid as grid


WINDOW_NAME = "Ten V4L2 Motion Detector"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect large motion in the bottom half of multiple Ubuntu/V4L2 USB cameras."
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
    parser.add_argument("--display-fps", type=float, default=20.0, help="Preview and detection FPS. Use 0 for uncapped.")
    parser.add_argument("--fourcc", default="MJPG", help="Requested V4L2 pixel format.")
    parser.add_argument("--camera-count", type=int, default=10, help="Number of cameras to auto-select.")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns.")
    parser.add_argument("--display-height", type=int, default=200, help="Displayed height per tile.")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after seconds. 0 runs until q/Esc.")
    parser.add_argument(
        "--min-motion-ratio",
        type=float,
        default=0.20,
        help="Minimum largest motion blob area as a fraction of the full frame.",
    )
    parser.add_argument(
        "--motion-frames",
        type=int,
        default=3,
        help="Consecutive qualifying frames required before motion is reported.",
    )
    parser.add_argument("--pixel-threshold", type=int, default=25, help="Per-pixel grayscale difference threshold.")
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=0.03,
        help="Running background update rate. Lower is less sensitive to slow lighting changes.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Hide overlays.")
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV worker threads. 1 often lowers CPU contention with many cameras. Use 0 for OpenCV default.",
    )
    parser.add_argument("--scan", action="store_true", help="List V4L2 video devices and exit.")
    return parser.parse_args()


class BottomHalfMotionDetector:
    def __init__(self, min_motion_ratio, motion_frames, pixel_threshold, background_alpha):
        self.min_motion_ratio = min_motion_ratio
        self.motion_frames = motion_frames
        self.pixel_threshold = pixel_threshold
        self.background_alpha = background_alpha
        self.background = None
        self.consecutive_motion_frames = 0
        self.motion_active = False
        self.last_area_ratio = 0.0
        self.last_boxes = []

    def update(self, frame):
        cv2 = grid.cv2
        roi_y = frame.shape[0] // 2
        roi = frame[roi_y:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if self.background is None:
            self.background = gray.astype("float")
            self.last_area_ratio = 0.0
            self.last_boxes = []
            return False, False, roi_y, []

        background = cv2.convertScaleAbs(self.background)
        delta = cv2.absdiff(background, gray)
        cv2.accumulateWeighted(gray, self.background, self.background_alpha)
        mask = cv2.threshold(delta, self.pixel_threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=3)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        full_frame_area = frame.shape[0] * frame.shape[1]
        min_area = self.min_motion_ratio * full_frame_area
        boxes = []
        largest_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            largest_area = max(largest_area, area)
            if area >= min_area:
                x, y, width, height = cv2.boundingRect(contour)
                boxes.append((x, y + roi_y, width, height, area))

        self.last_area_ratio = largest_area / max(full_frame_area, 1)
        self.last_boxes = boxes
        if boxes:
            self.consecutive_motion_frames += 1
        else:
            self.consecutive_motion_frames = 0

        was_active = self.motion_active
        self.motion_active = self.consecutive_motion_frames >= self.motion_frames
        just_activated = self.motion_active and not was_active
        return self.motion_active, just_activated, roi_y, boxes


def draw_motion_overlay(frame, camera, detector, roi_y, boxes, capture_fps):
    cv2 = grid.cv2
    frame = frame.copy()
    cv2.line(frame, (0, roi_y), (frame.shape[1] - 1, roi_y), (255, 180, 0), 2)
    roi_text = "ROI: bottom half"
    cv2.putText(frame, roi_text, (12, roi_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2, cv2.LINE_AA)

    color = (0, 0, 255) if detector.motion_active else (80, 220, 80)
    status = "MOTION" if detector.motion_active else "clear"
    text = (
        f"{camera.device} | {capture_fps:4.1f} FPS | {status} | "
        f"{detector.consecutive_motion_frames}/{detector.motion_frames} | "
        f"area {detector.last_area_ratio * 100:4.1f}%"
    )
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (6, 6), (min(frame.shape[1] - 4, text_width + 20), text_height + baseline + 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    for x, y, width, height, _area in boxes:
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
    if detector.motion_active:
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), color, 5)
    return frame


def draw_motion_overlay_display(frame, camera, detector, source_shape, roi_y, boxes, capture_fps):
    cv2 = grid.cv2
    source_height, source_width = source_shape[:2]
    scale_x = frame.shape[1] / max(source_width, 1)
    scale_y = frame.shape[0] / max(source_height, 1)
    roi_y_display = int(round(roi_y * scale_y))

    cv2.line(frame, (0, roi_y_display), (frame.shape[1] - 1, roi_y_display), (255, 180, 0), 1)
    cv2.putText(
        frame,
        "ROI",
        (8, min(frame.shape[0] - 8, roi_y_display + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )

    color = (0, 0, 255) if detector.motion_active else (80, 220, 80)
    status = "MOTION" if detector.motion_active else "clear"
    text = (
        f"{camera.device} {capture_fps:4.1f} FPS {status} "
        f"{detector.consecutive_motion_frames}/{detector.motion_frames} "
        f"{detector.last_area_ratio * 100:4.1f}%"
    )
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, text_width + 12), text_height + baseline + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, text_height + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    for x, y, width, height, _area in boxes:
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + width) * scale_x))
        y2 = int(round((y + height) * scale_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if detector.motion_active:
        cv2.rectangle(frame, (1, 1), (frame.shape[1] - 2, frame.shape[0] - 2), color, 3)
    return frame


def make_motion_grid(cameras, detectors, cols, display_height, rotation, no_overlay, source_aspect):
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
        ok, frame, _source_shape, fps = camera.read_latest(copy_frame=False)
        if ok:
            frame = grid.rotate_frame(frame, rotation)
            source_shape = frame.shape
            _motion_active, just_activated, roi_y, boxes = detector.update(frame)
            if just_activated:
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} motion {camera.device} "
                    f"area={detector.last_area_ratio * 100:.1f}%",
                    flush=True,
                )
            displayed = grid.paste_letterboxed(canvas, frame, x, y, tile_width, tile_height)
            if not no_overlay:
                draw_motion_overlay_display(displayed, camera, detector, source_shape, roi_y, boxes, fps)
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
    detectors = [
        BottomHalfMotionDetector(
            args.min_motion_ratio,
            args.motion_frames,
            args.pixel_threshold,
            args.background_alpha,
        )
        for _device in args.devices
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

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        print(
            f"motion ROI: bottom half; min blob: {args.min_motion_ratio * 100:g}% of full frame; "
            f"threshold: {args.motion_frames} consecutive frames",
            flush=True,
        )
        print("q/Esc: quit", flush=True)
        while True:
            loop_started_at = time.perf_counter()
            grid_frame = make_motion_grid(
                cameras,
                detectors,
                cols,
                args.display_height,
                args.rotation,
                args.no_overlay,
                source_aspect,
            )
            if not window_sized:
                cv2.resizeWindow(WINDOW_NAME, grid_frame.shape[1], grid_frame.shape[0])
                window_sized = True
            cv2.imshow(WINDOW_NAME, grid_frame)

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
