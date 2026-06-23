import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Calibrate floor warps from a printed ArUco GridBoard.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("floor_checkerboard_warps.json"))
    parser.add_argument("--annotated", type=Path, default=Path("aruco_floor_calibration_detected.jpg"))
    parser.add_argument("--residual-json", type=Path, default=Path("aruco_floor_residuals.json"))
    parser.add_argument("--residual-csv", type=Path, default=Path("aruco_floor_residuals.csv"))
    parser.add_argument("--residual-annotated", type=Path, default=Path("aruco_floor_residuals.jpg"))
    parser.add_argument("--residual-warn-px", type=float, default=3.0)
    parser.add_argument("--residual-bad-px", type=float, default=6.0)
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--board-x-markers", type=int, default=7, help="Markers along arena X, the long enclosure axis.")
    parser.add_argument("--board-y-markers", type=int, default=5, help="Markers along arena Y, the short enclosure axis.")
    parser.add_argument("--marker-length-mm", type=float, default=27.0, help="Printed side length of one ArUco marker.")
    parser.add_argument(
        "--board-x-mm",
        type=float,
        default=0.0,
        help="Optional total printed board length along arena X. If omitted, computed from marker length and separation.",
    )
    parser.add_argument("--marker-separation-mm", type=float, default=7.0)
    parser.add_argument("--board-center-x-mm", type=float, default=0.0)
    parser.add_argument("--board-center-y-mm", type=float, default=0.0)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=30, help="Frames to aggregate for multi-frame calibration.")
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between aggregated calibration frames.")
    parser.add_argument(
        "--min-observations",
        type=int,
        default=3,
        help="Minimum repeated observations for a marker corner to be used.",
    )
    parser.add_argument(
        "--outlier-px",
        type=float,
        default=3.0,
        help="Reject repeated corner observations farther than this from their median.",
    )
    parser.add_argument("--min-corners", type=int, default=8, help="Minimum detected tag corners per camera to save a warp.")
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


def marker_length_m(args):
    return args.marker_length_mm / 1000.0


def board_dimensions_m(args):
    marker_m = marker_length_m(args)
    separation_m = args.marker_separation_mm / 1000.0
    if args.board_x_mm > 0:
        board_x_m = args.board_x_mm / 1000.0
    else:
        board_x_m = args.board_x_markers * marker_m + (args.board_x_markers - 1) * separation_m
    board_y_m = args.board_y_markers * marker_m + (args.board_y_markers - 1) * separation_m
    return board_x_m, board_y_m, marker_m, separation_m


def marker_world_corners(marker_id, args):
    board_x_m, board_y_m, marker_m, separation_m = board_dimensions_m(args)
    local_id = int(marker_id) - args.first_id
    if local_id < 0 or local_id >= args.board_x_markers * args.board_y_markers:
        return None

    # Printed page convention from the supplied board:
    # rows run along arena X, columns run along arena Y.
    row_x = local_id // args.board_y_markers
    col_y = local_id % args.board_y_markers
    pitch = marker_m + separation_m
    center_x = args.board_center_x_mm / 1000.0
    center_y = args.board_center_y_mm / 1000.0

    x0 = center_x - board_x_m / 2.0 + row_x * pitch
    x1 = x0 + marker_m
    y0 = center_y - board_y_m / 2.0 + col_y * pitch
    y1 = y0 + marker_m

    # OpenCV returns marker corners in printed-image order:
    # top-left, top-right, bottom-right, bottom-left.
    # In our arena convention, printed down is +X and printed right is +Y.
    return np.asarray(
        [
            [x0, y0],
            [x0, y1],
            [x1, y1],
            [x1, y0],
        ],
        dtype=np.float32,
    )


def make_detector(dictionary_name):
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"OpenCV has no aruco dictionary named {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 0.5
    params.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(dictionary, params)


def fit_homography(world_points, image_points):
    if len(world_points) < 4:
        return None, None, None
    homography, inlier_mask = cv2.findHomography(
        np.asarray(world_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        cv2.RANSAC,
        4.0,
    )
    if homography is None:
        return None, None, None
    projected = cv2.perspectiveTransform(np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 2), homography)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - np.asarray(image_points, dtype=np.float32), axis=1)
    inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else np.ones(len(errors), dtype=bool)
    rms = float(np.sqrt(np.mean(np.square(errors[inliers])))) if np.any(inliers) else float("nan")
    return homography, inliers, rms


def detect_camera(frame, detector, args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return [], np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    world_points = []
    image_points = []
    detections = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_world = marker_world_corners(int(marker_id), args)
        if marker_world is None:
            continue
        marker_image = marker_corners.reshape(4, 2).astype(np.float32)
        world_points.extend(marker_world)
        image_points.extend(marker_image)
        detections.append((int(marker_id), marker_image))
    return detections, np.asarray(world_points, dtype=np.float32), np.asarray(image_points, dtype=np.float32)


def add_observations(observations, detections, args):
    for marker_id, marker_image in detections:
        marker_world = marker_world_corners(marker_id, args)
        if marker_world is None:
            continue
        for corner_index in range(4):
            key = (marker_id, corner_index)
            if key not in observations:
                observations[key] = {
                    "world": marker_world[corner_index],
                    "images": [],
                }
            observations[key]["images"].append(marker_image[corner_index])


def aggregate_observations(observations, min_observations, outlier_px):
    world_points = []
    image_points = []
    point_meta = []
    stats = {
        "raw_observations": 0,
        "used_observations": 0,
        "used_corners": 0,
        "rejected_corners": 0,
    }
    for marker_id, corner_index in sorted(observations.keys()):
        entry = observations[(marker_id, corner_index)]
        images = np.asarray(entry["images"], dtype=np.float32)
        stats["raw_observations"] += len(images)
        if len(images) < min_observations:
            stats["rejected_corners"] += 1
            continue
        median = np.median(images, axis=0)
        distances = np.linalg.norm(images - median, axis=1)
        keep = distances <= outlier_px
        if np.count_nonzero(keep) < min_observations:
            stats["rejected_corners"] += 1
            continue
        kept = images[keep]
        world_points.append(entry["world"])
        image_point = np.mean(kept, axis=0)
        image_points.append(image_point)
        point_meta.append(
            {
                "marker_id": int(marker_id),
                "corner_index": int(corner_index),
                "observations": int(len(images)),
                "used_observations": int(len(kept)),
                "median_px": median.astype(float).tolist(),
                "mean_px": image_point.astype(float).tolist(),
                "jitter_px": float(np.sqrt(np.mean(np.square(np.linalg.norm(kept - image_point, axis=1))))),
            }
        )
        stats["used_observations"] += int(len(kept))
        stats["used_corners"] += 1
    return np.asarray(world_points, dtype=np.float32), np.asarray(image_points, dtype=np.float32), point_meta, stats


def compute_residual_report(camera_id, world_points, image_points, point_meta, homography, inliers, rms):
    projected = cv2.perspectiveTransform(np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 2), homography)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - np.asarray(image_points, dtype=np.float32), axis=1)
    inlier_flags = inliers if inliers is not None else np.ones(len(errors), dtype=bool)

    points = []
    tag_errors = {}
    for index, meta in enumerate(point_meta):
        marker_id = int(meta["marker_id"])
        error = float(errors[index])
        point = {
            **meta,
            "camera_id": int(camera_id),
            "world_m": np.asarray(world_points[index], dtype=float).tolist(),
            "image_px": np.asarray(image_points[index], dtype=float).tolist(),
            "projected_px": projected[index].astype(float).tolist(),
            "residual_px": error,
            "ransac_inlier": bool(inlier_flags[index]),
        }
        points.append(point)
        tag_errors.setdefault(marker_id, []).append(error)

    tags = []
    for marker_id, values in sorted(tag_errors.items()):
        values = np.asarray(values, dtype=np.float32)
        tags.append(
            {
                "marker_id": int(marker_id),
                "corners": int(len(values)),
                "mean_residual_px": float(np.mean(values)),
                "max_residual_px": float(np.max(values)),
            }
        )

    return {
        "camera_id": int(camera_id),
        "rms_px": float(rms),
        "points": int(len(points)),
        "inliers": int(np.count_nonzero(inlier_flags)),
        "max_residual_px": float(np.max(errors)) if len(errors) else None,
        "mean_residual_px": float(np.mean(errors)) if len(errors) else None,
        "tags": tags,
        "corners": points,
    }


def write_residual_reports(json_path, csv_path, reports):
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": reports,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "camera_id",
            "marker_id",
            "corner_index",
            "residual_px",
            "ransac_inlier",
            "observations",
            "used_observations",
            "jitter_px",
            "image_x",
            "image_y",
            "projected_x",
            "projected_y",
            "world_x_m",
            "world_y_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            for point in report["corners"]:
                writer.writerow(
                    {
                        "camera_id": point["camera_id"],
                        "marker_id": point["marker_id"],
                        "corner_index": point["corner_index"],
                        "residual_px": f"{point['residual_px']:.4f}",
                        "ransac_inlier": int(point["ransac_inlier"]),
                        "observations": point["observations"],
                        "used_observations": point["used_observations"],
                        "jitter_px": f"{point['jitter_px']:.4f}",
                        "image_x": f"{point['image_px'][0]:.3f}",
                        "image_y": f"{point['image_px'][1]:.3f}",
                        "projected_x": f"{point['projected_px'][0]:.3f}",
                        "projected_y": f"{point['projected_px'][1]:.3f}",
                        "world_x_m": f"{point['world_m'][0]:.6f}",
                        "world_y_m": f"{point['world_m'][1]:.6f}",
                    }
                )


def save_calibration(path, args, geometry, calibrations):
    board_x_m, board_y_m, marker_m, separation_m = board_dimensions_m(args)
    payload = {
        "type": "aruco_floor_homography",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry": {
            "arena_width_m": float(geometry["measurements"]["arena_width_m"]),
            "arena_length_m": float(geometry["measurements"]["arena_length_m"]),
            "coordinate_note": "ArUco calibration uses X=long enclosure axis, Y=short enclosure axis.",
        },
        "source": {
            "width": args.width,
            "height": args.height,
            "rotation": args.rotation,
            "format": args.format,
        },
        "aruco_board": {
            "dictionary": args.dictionary,
            "board_x_markers": args.board_x_markers,
            "board_y_markers": args.board_y_markers,
            "board_x_m": board_x_m,
            "board_y_m": board_y_m,
            "marker_m": marker_m,
            "marker_separation_m": separation_m,
            "center_m": [args.board_center_x_mm / 1000.0, args.board_center_y_mm / 1000.0],
            "first_id": args.first_id,
        },
        "cameras": calibrations,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def annotate(frame, detections, camera_id, rms=None):
    shown = frame.copy()
    if detections:
        corners = [image.reshape(1, 4, 2) for _marker_id, image in detections]
        ids = np.asarray([[marker_id] for marker_id, _image in detections], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(shown, corners, ids)
        for marker_id, image in detections:
            center = np.mean(image, axis=0).astype(int)
            cv2.putText(shown, str(marker_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    label = f"cam {camera_id}: {len(detections)} tags"
    if rms is not None:
        label += f" fit {rms:.2f}px"
    cv2.rectangle(shown, (8, 8), (min(shown.shape[1] - 8, 460), 48), (0, 0, 0), -1)
    cv2.putText(shown, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def annotate_residuals(frame, report, warn_px, bad_px):
    shown = frame.copy()
    by_marker = {}
    for point in report["corners"]:
        residual = point["residual_px"]
        inlier = point["ransac_inlier"]
        if not inlier or residual >= bad_px:
            color = (0, 0, 255)
        elif residual >= warn_px:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)
        image = tuple(np.round(point["image_px"]).astype(int))
        projected = tuple(np.round(point["projected_px"]).astype(int))
        cv2.circle(shown, image, 5, color, -1, cv2.LINE_AA)
        cv2.circle(shown, projected, 7, color, 1, cv2.LINE_AA)
        cv2.line(shown, image, projected, color, 1, cv2.LINE_AA)
        by_marker.setdefault(point["marker_id"], []).append(point)

    for marker_id, points in by_marker.items():
        center = np.mean([point["image_px"] for point in points], axis=0).astype(int)
        mean_error = np.mean([point["residual_px"] for point in points])
        cv2.putText(
            shown,
            f"{marker_id}:{mean_error:.1f}",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    rejected = report["points"] - report["inliers"]
    label = (
        f"cam {report['camera_id']}: rms {report['rms_px']:.2f}px "
        f"max {report['max_residual_px']:.2f}px rejected {rejected}"
    )
    cv2.rectangle(shown, (8, 8), (min(shown.shape[1] - 8, 650), 48), (0, 0, 0), -1)
    cv2.putText(shown, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def write_contact_sheet(path, frames):
    tiles = [resize_to_height(frame, 360) for frame in frames]
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), 2):
        row = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + 2]]
        if len(row) < 2:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    args = parse_args()
    geometry = load_geometry(args.geometry)
    board_x_m, board_y_m, marker_m, separation_m = board_dimensions_m(args)
    if marker_m <= 0:
        raise RuntimeError("Computed marker length is <= 0. Check --board-x-mm and --marker-separation-mm.")
    print(
        "ArUco board defaults: "
        f"{args.board_x_markers} markers along X, {args.board_y_markers} along Y, "
        f"marker={marker_m*1000:.2f}mm, separation={separation_m*1000:.2f}mm, "
        f"board={board_x_m*1000:.1f}x{board_y_m*1000:.1f}mm",
        flush=True,
    )

    detector = make_detector(args.dictionary)
    cameras = []
    calibrations = {}
    observations_by_camera = {}
    latest_detections = {}
    latest_frames = {}
    annotated = []
    residual_annotated = []
    residual_reports = []
    try:
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        end_warmup = time.perf_counter() + args.warmup
        while time.perf_counter() < end_warmup:
            time.sleep(0.01)

        for camera in cameras:
            observations_by_camera[camera.device_index] = {}

        for frame_index in range(args.frames):
            if args.interval > 0:
                time.sleep(args.interval)
            for camera in cameras:
                if camera.latest_frame is None:
                    continue
                frame = rotate_frame(camera.latest_frame, args.rotation)
                detections, _world_points, _image_points = detect_camera(frame, detector, args)
                add_observations(observations_by_camera[camera.device_index], detections, args)
                latest_detections[camera.device_index] = detections
                latest_frames[camera.device_index] = frame
            if (frame_index + 1) % max(1, args.frames // 5) == 0 or frame_index == args.frames - 1:
                counts = []
                for camera in cameras:
                    obs = observations_by_camera[camera.device_index]
                    counts.append(f"cam {camera.device_index}: {len(obs)} corners")
                print(f"Aggregated frame {frame_index + 1}/{args.frames} ({', '.join(counts)})", flush=True)

        for camera in cameras:
            observations = observations_by_camera[camera.device_index]
            if not observations:
                print(f"cam {camera.device_index}: no ArUco observations", flush=True)
                continue
            world_points, image_points, point_meta, stats = aggregate_observations(
                observations,
                args.min_observations,
                args.outlier_px,
            )
            homography, inliers, rms = fit_homography(world_points, image_points)
            detections = latest_detections.get(camera.device_index, [])
            frame = latest_frames.get(camera.device_index)
            if frame is not None:
                annotated.append(annotate(frame, detections, camera.device_index, rms))
            if homography is None or len(image_points) < args.min_corners:
                print(
                    f"cam {camera.device_index}: insufficient calibration points "
                    f"({len(detections)} latest tags, {len(image_points)} averaged corners)",
                    flush=True,
                )
                continue
            report = compute_residual_report(
                camera.device_index,
                world_points,
                image_points,
                point_meta,
                homography,
                inliers,
                rms,
            )
            residual_reports.append(report)
            if frame is not None:
                residual_annotated.append(annotate_residuals(frame, report, args.residual_warn_px, args.residual_bad_px))
            tags_used = len({marker_id for marker_id, _corner_index in observations.keys()})
            calibrations[str(camera.device_index)] = {
                "homography_world_to_image": homography.tolist(),
                "rms_px": rms,
                "inliers": int(np.count_nonzero(inliers)),
                "points": int(len(image_points)),
                "tags": int(tags_used),
                "method": "aruco_gridboard_multiframe_homography",
                "frames": int(args.frames),
                "raw_observations": int(stats["raw_observations"]),
                "used_observations": int(stats["used_observations"]),
                "used_corners": int(stats["used_corners"]),
                "rejected_corners": int(stats["rejected_corners"]),
                "residual_summary": {
                    "mean_px": report["mean_residual_px"],
                    "max_px": report["max_residual_px"],
                    "warn_threshold_px": args.residual_warn_px,
                    "bad_threshold_px": args.residual_bad_px,
                    "rejected_by_ransac": int(report["points"] - report["inliers"]),
                },
                "manual_pairs": [
                    {
                        "image_px": image_points[index].astype(float).tolist(),
                        "world_m": world_points[index].astype(float).tolist(),
                    }
                    for index in range(len(image_points))
                ],
            }
            print(
                f"cam {camera.device_index}: {tags_used} tags, {len(image_points)} averaged corners, "
                f"{stats['used_observations']}/{stats['raw_observations']} observations used, "
                f"{int(np.count_nonzero(inliers))} inliers, fit {rms:.2f}px",
                flush=True,
            )

        if annotated:
            write_contact_sheet(args.annotated, annotated)
            print(f"Wrote annotated detection sheet: {args.annotated}", flush=True)
        if residual_reports:
            write_residual_reports(args.residual_json, args.residual_csv, residual_reports)
            print(f"Wrote residual reports: {args.residual_json}, {args.residual_csv}", flush=True)
        if residual_annotated:
            write_contact_sheet(args.residual_annotated, residual_annotated)
            print(f"Wrote residual diagnostic sheet: {args.residual_annotated}", flush=True)
        if calibrations:
            save_calibration(args.output, args, geometry, calibrations)
            print(f"Saved ArUco floor warps -> {args.output}", flush=True)
        else:
            print("No camera had enough ArUco corners to save calibration.", flush=True)
    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
