import argparse
import json
import os
import time
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

import cv2
import numpy as np

from dshow_arducam_viewer import DShowCamera, find_format_index, fit_to_tile, resize_to_height


cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a checkerboard on the arena floor and save empirical camera-to-floor warps."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("floor_checkerboard_warps.json"))
    parser.add_argument("--pattern-cols", type=int, required=True, help="Number of inner checkerboard corners across.")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Number of inner checkerboard corners along.")
    parser.add_argument("--square-mm", type=float, required=True, help="Measured checkerboard square size in millimeters.")
    parser.add_argument(
        "--origin-x-mm",
        type=float,
        default=None,
        help="World x coordinate of the first detected inner corner, in mm. Defaults to centering the pattern.",
    )
    parser.add_argument(
        "--origin-y-mm",
        type=float,
        default=None,
        help="World y coordinate of the first detected inner corner, in mm. Defaults to centering the pattern.",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        help="Reverse checkerboard x world coordinates if the saved warp is mirrored.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Reverse checkerboard y world coordinates if the saved warp is mirrored.",
    )
    parser.add_argument("--display-height", type=int, default=320)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("floor_checkerboard_frames"))
    parser.add_argument(
        "--full-frame-only",
        action="store_true",
        help="Disable the extra bottom-half crop searches used for floor-mounted checkerboards.",
    )
    return parser.parse_args()


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def load_geometry(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_world_corners(args, geometry, pattern_size=None):
    square_m = args.square_mm / 1000.0
    cols, rows = pattern_size if pattern_size is not None else (args.pattern_cols, args.pattern_rows)
    pattern_width = (cols - 1) * square_m
    pattern_length = (rows - 1) * square_m

    if args.origin_x_mm is None:
        origin_x = -pattern_width / 2.0
    else:
        origin_x = args.origin_x_mm / 1000.0
    if args.origin_y_mm is None:
        origin_y = -pattern_length / 2.0
    else:
        origin_y = args.origin_y_mm / 1000.0

    xs = np.arange(cols, dtype=np.float32) * square_m + origin_x
    ys = np.arange(rows, dtype=np.float32) * square_m + origin_y
    if args.flip_x:
        xs = xs[::-1]
    if args.flip_y:
        ys = ys[::-1]

    points = []
    for y in ys:
        for x in xs:
            points.append([x, y])
    points = np.asarray(points, dtype=np.float32)

    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])
    outside = (
        (points[:, 0] < -arena_width / 2.0)
        | (points[:, 0] > arena_width / 2.0)
        | (points[:, 1] < -arena_length / 2.0)
        | (points[:, 1] > arena_length / 2.0)
    )
    if np.any(outside):
        count = int(np.count_nonzero(outside))
        print(f"Warning: {count} checkerboard corners are outside the arena extents.", flush=True)
    return points


def detection_variants(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.2)
    sharpened = cv2.addWeighted(clahe, 1.7, blurred, -0.7, 0)
    return [
        ("gray", gray),
        ("clahe", clahe),
        ("sharp", sharpened),
        ("inv", cv2.bitwise_not(gray)),
        ("inv_clahe", cv2.bitwise_not(clahe)),
    ]


def find_checkerboard_in_gray(gray, pattern_size):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
    flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    else:
        found, corners = False, None
    if not found:
        legacy_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, legacy_flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    if not found:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def find_checkerboard(frame, pattern_size, full_frame_only=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    y_starts = [0] if full_frame_only else [0, int(height * 0.20), int(height * 0.30), int(height * 0.40)]
    pattern_sizes = [pattern_size]
    if pattern_size[0] != pattern_size[1]:
        pattern_sizes.append((pattern_size[1], pattern_size[0]))

    for y_start in y_starts:
        crop = gray[y_start:height, :]
        if crop.shape[0] < 80:
            continue
        for variant_name, variant in detection_variants(crop):
            for candidate_size in pattern_sizes:
                corners = find_checkerboard_in_gray(variant, candidate_size)
                if corners is None:
                    continue
                corners[:, 1] += y_start
                return corners, candidate_size, variant_name, y_start
    return None, pattern_size, None, None


def fit_homography(world_points, image_points):
    homography, inlier_mask = cv2.findHomography(
        world_points.astype(np.float32),
        image_points.astype(np.float32),
        cv2.RANSAC,
        4.0,
    )
    if homography is None:
        return None, None, None
    projected = cv2.perspectiveTransform(world_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else np.ones(len(errors), dtype=bool)
    rms = float(np.sqrt(np.mean(np.square(errors[inliers])))) if np.any(inliers) else float("nan")
    return homography, inliers, rms


def draw_detection(frame, pattern_size, corners, rms=None, method=None):
    drawn = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(drawn, pattern_size, corners.reshape(-1, 1, 2), True)
    label = "checkerboard found" if corners is not None else "checkerboard not found"
    if rms is not None:
        label += f" | fit {rms:.2f}px"
    if method is not None:
        label += f" | {method}"
    cv2.rectangle(drawn, (8, 8), (min(drawn.shape[1] - 8, 520), 42), (0, 0, 0), -1)
    cv2.putText(drawn, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return drawn


def save_calibration(path, args, geometry, calibrations):
    payload = {
        "type": "checkerboard_floor_homography",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry": {
            "arena_width_m": float(geometry["measurements"]["arena_width_m"]),
            "arena_length_m": float(geometry["measurements"]["arena_length_m"]),
        },
        "pattern": {
            "inner_corners": [args.pattern_cols, args.pattern_rows],
            "square_m": args.square_mm / 1000.0,
            "origin_m": [
                None if args.origin_x_mm is None else args.origin_x_mm / 1000.0,
                None if args.origin_y_mm is None else args.origin_y_mm / 1000.0,
            ],
            "centered_if_origin_omitted": True,
            "flip_x": bool(args.flip_x),
            "flip_y": bool(args.flip_y),
        },
        "source": {
            "width": args.width,
            "height": args.height,
            "rotation": args.rotation,
            "format": args.format,
        },
        "cameras": calibrations,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_grid(tiles, cols=2):
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), cols):
        row = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + cols]]
        if len(row) < cols:
            row.extend(np.zeros((tile_height, tile_width, 3), dtype=np.uint8) for _ in range(cols - len(row)))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    args = parse_args()
    geometry = load_geometry(args.geometry)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    world_points = make_world_corners(args, geometry, pattern_size)
    cameras = []
    latest = {}
    calibrations = {}
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        cv2.namedWindow("Floor Checkerboard Calibration", cv2.WINDOW_NORMAL)
        print("Press s to save detections for all currently found cameras. Press q/Esc to quit.", flush=True)
        while True:
            tiles = []
            latest.clear()
            for camera in cameras:
                frame = camera.latest_frame
                if frame is None:
                    shown = np.zeros((args.display_height, int(args.display_height * 16 / 9), 3), dtype=np.uint8)
                    cv2.putText(
                        shown,
                        f"cam {camera.device_index} waiting",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    tiles.append(shown)
                    continue
                display = rotate_frame(frame, args.rotation)
                corners, detected_size, variant_name, y_start = find_checkerboard(
                    display,
                    pattern_size,
                    args.full_frame_only,
                )
                homography = None
                rms = None
                if corners is not None:
                    detected_world_points = (
                        world_points
                        if detected_size == pattern_size
                        else make_world_corners(args, geometry, detected_size)
                    )
                    homography, inliers, rms = fit_homography(detected_world_points, corners)
                    if homography is not None:
                        latest[camera.device_index] = {
                            "homography_world_to_image": homography.tolist(),
                            "rms_px": rms,
                            "inliers": int(np.count_nonzero(inliers)),
                            "points": int(len(corners)),
                            "detected_inner_corners": [int(detected_size[0]), int(detected_size[1])],
                            "detection_variant": variant_name,
                            "detection_y_start_px": int(y_start or 0),
                        }
                method = None
                if corners is not None:
                    method = f"{detected_size[0]}x{detected_size[1]} {variant_name} y>{int(y_start or 0)}"
                shown = draw_detection(display, detected_size, corners, rms, method)
                cv2.putText(
                    shown,
                    f"cam {camera.device_index}",
                    (18, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(resize_to_height(shown, args.display_height))

            cv2.imshow("Floor Checkerboard Calibration", make_grid(tiles))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if not latest:
                    print(
                        "No checkerboards currently detected; nothing saved. "
                        "OpenCV needs the requested inner-corner rectangle visible. "
                        "Try checking --pattern-cols/--pattern-rows, or rerun with a smaller visible inner-corner region.",
                        flush=True,
                    )
                    continue
                for camera in cameras:
                    frame = camera.latest_frame
                    if frame is not None:
                        display = rotate_frame(frame, args.rotation)
                        cv2.imwrite(str(args.snapshot_dir / f"cam_{camera.device_index}.png"), display)
                calibrations.update({str(camera_id): value for camera_id, value in latest.items()})
                save_calibration(args.output, args, geometry, calibrations)
                saved = ", ".join(str(camera_id) for camera_id in sorted(latest))
                print(f"Saved checkerboard floor warps for cameras: {saved} -> {args.output}", flush=True)
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
