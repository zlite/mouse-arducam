import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

import cv2
import numpy as np

from dshow_arducam_viewer import DShowCamera, find_format_index, fit_to_tile, resize_to_height
from nine_dshow_camera_grid import POSITION_LABELS, load_position_assignments


cv2.ocl.setUseOpenCL(False)


DEFAULT_CAMERA_BOARDS = {
    "front": ["front"],
    "back": ["back"],
    "side": ["side_1", "side_2"],
    "top": ["top"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate camera extrinsics from fixed ArUco/AprilTag boards and 9-grid camera position assignments."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=None, help="Defaults to assigned cameras.")
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--intrinsics", type=Path, default=Path("camera_intrinsics.npz"))
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--positions-config", type=Path, default=Path("nine_dshow_camera_grid_positions.json"))
    parser.add_argument("--board-config", type=Path, default=Path("composite_aruco_board_config.json"))
    parser.add_argument("--output", type=Path, default=Path("camera_poses_aruco.npz"))
    parser.add_argument("--annotated", type=Path, default=Path("aruco_extrinsics_detected.jpg"))
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--outlier-px", type=float, default=3.0)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--reprojection-warn-px", type=float, default=5.0)
    parser.add_argument("--write-board-template", action="store_true")
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


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_board_template(path, geometry_path):
    geometry = load_json(geometry_path)
    width = float(geometry["measurements"]["arena_width_m"])
    length = float(geometry["measurements"]["arena_length_m"])
    half_w = width / 2.0
    half_l = length / 2.0

    template = {
        "units": "meters",
        "note": "Edit board origins/axes and ID ranges/layouts to match the printed marker boards fixed in the enclosure.",
        "camera_boards": DEFAULT_CAMERA_BOARDS,
        "boards": {
            "front": {
                "first_id": 0,
                "markers_x": 4,
                "markers_y": 3,
                "marker_m": 0.018,
                "separation_m": 0.006,
                "origin_m": [-0.045, -half_l, 0.030],
                "x_axis_m": [1.0, 0.0, 0.0],
                "y_axis_m": [0.0, 0.0, 1.0],
            },
            "back": {
                "first_id": 50,
                "markers_x": 4,
                "markers_y": 3,
                "marker_m": 0.018,
                "separation_m": 0.006,
                "origin_m": [0.045, half_l, 0.030],
                "x_axis_m": [-1.0, 0.0, 0.0],
                "y_axis_m": [0.0, 0.0, 1.0],
            },
            "side_1": {
                "first_id": 100,
                "markers_x": 4,
                "markers_y": 3,
                "marker_m": 0.018,
                "separation_m": 0.006,
                "origin_m": [-half_w, -0.045, 0.030],
                "x_axis_m": [0.0, 1.0, 0.0],
                "y_axis_m": [0.0, 0.0, 1.0],
            },
            "side_2": {
                "first_id": 150,
                "markers_x": 4,
                "markers_y": 3,
                "marker_m": 0.018,
                "separation_m": 0.006,
                "origin_m": [half_w, 0.045, 0.030],
                "x_axis_m": [0.0, -1.0, 0.0],
                "y_axis_m": [0.0, 0.0, 1.0],
            },
            "top": {
                "first_id": 200,
                "markers_x": 5,
                "markers_y": 4,
                "marker_m": 0.018,
                "separation_m": 0.006,
                "origin_m": [-0.060, -0.045, 0.0],
                "x_axis_m": [1.0, 0.0, 0.0],
                "y_axis_m": [0.0, 1.0, 0.0],
            },
        },
    }
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {path}")
    path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def make_detector(dictionary_name):
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"OpenCV has no aruco dictionary named {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 0.5
    return cv2.aruco.ArucoDetector(dictionary, params)


def normalize(vector, name):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError(f"{name} has near-zero length")
    return vector / norm


def parse_marker_id(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    return int(match.group(0))


def normalized_id_grid(board):
    grid = board.get("id_grid")
    if grid is None:
        return None
    return [[parse_marker_id(cell) for cell in row] for row in grid]


def marker_grid_position(board, marker_id):
    marker_id = int(marker_id)
    grid = normalized_id_grid(board)
    if grid is not None:
        for row, row_ids in enumerate(grid):
            for col, grid_marker_id in enumerate(row_ids):
                if grid_marker_id == marker_id:
                    return row, col
        return None

    local_id = marker_id - int(board["first_id"])
    markers_x = int(board["markers_x"])
    markers_y = int(board["markers_y"])
    if local_id < 0 or local_id >= markers_x * markers_y:
        return None
    return local_id // markers_x, local_id % markers_x


def board_marker_ids(board):
    grid = normalized_id_grid(board)
    if grid is not None:
        for row in grid:
            for marker_id in row:
                if marker_id is not None:
                    yield marker_id
        return

    count = int(board["markers_x"]) * int(board["markers_y"])
    yield from range(int(board["first_id"]), int(board["first_id"]) + count)


def board_marker_world_corners(board, marker_id):
    position = marker_grid_position(board, marker_id)
    if position is None:
        return None
    row, col = position
    marker_m = float(board["marker_m"])
    pitch = marker_m + float(board.get("separation_m", 0.0))
    origin = np.asarray(board["origin_m"], dtype=np.float64).reshape(3)
    x_axis = normalize(board["x_axis_m"], "board x_axis_m")
    y_axis = normalize(board["y_axis_m"], "board y_axis_m")
    x0 = col * pitch
    y0 = row * pitch

    local_corners = [(x0, y0), (x0 + marker_m, y0), (x0 + marker_m, y0 + marker_m), (x0, y0 + marker_m)]
    corner_shift = int(board.get("corner_shift", 0)) % 4
    if corner_shift:
        local_corners = local_corners[corner_shift:] + local_corners[:corner_shift]
    return np.asarray([origin + x * x_axis + y * y_axis for x, y in local_corners], dtype=np.float32)


def build_marker_lookup(board_config):
    marker_lookup = {}
    for board_name, board in board_config["boards"].items():
        for marker_id in board_marker_ids(board):
            if marker_id in marker_lookup:
                raise ValueError(f"Marker id {marker_id} appears on more than one board")
            marker_lookup[marker_id] = board_name
    return marker_lookup


def expected_boards_for_position(position_key, board_config):
    family = position_key.split("_", 1)[0]
    camera_boards = board_config.get("camera_boards", DEFAULT_CAMERA_BOARDS)
    return set(camera_boards.get(position_key, camera_boards.get(family, board_config["boards"].keys())))


def camera_center_prior(position_key, geometry):
    measurements = geometry.get("measurements", {})
    width = float(measurements.get("arena_width_m", 0.29))
    length = float(measurements.get("arena_length_m", 0.38))
    side_z = float(measurements.get("side_camera_center_z_m", 0.05))
    top_z = float(measurements.get("top_camera_center_z_m", 0.25))
    wall_offset = float(measurements.get("camera_wall_offset_m", 0.0))
    family = position_key.split("_", 1)[0]

    if family == "front":
        return np.asarray([0.0, -length / 2.0 - wall_offset, side_z], dtype=np.float64)
    if family == "back":
        return np.asarray([0.0, length / 2.0 + wall_offset, side_z], dtype=np.float64)
    if family == "side":
        side_index = int(position_key.split("_", 1)[1]) if "_" in position_key else 1
        x = -width / 2.0 - wall_offset if side_index == 1 else width / 2.0 + wall_offset
        return np.asarray([x, 0.0, side_z], dtype=np.float64)
    if family == "top":
        return np.asarray([0.0, 0.0, top_z], dtype=np.float64)
    return None


def detect_points(frame, detector, board_config, marker_lookup, allowed_boards):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return [], np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    detections = []
    object_points = []
    image_points = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        board_name = marker_lookup.get(marker_id)
        if board_name is None or board_name not in allowed_boards:
            continue
        world_corners = board_marker_world_corners(board_config["boards"][board_name], marker_id)
        if world_corners is None:
            continue
        image_corners = marker_corners.reshape(4, 2).astype(np.float32)
        object_points.extend(world_corners)
        image_points.extend(image_corners)
        detections.append({"marker_id": marker_id, "board": board_name, "image": image_corners})
    return detections, np.asarray(object_points, dtype=np.float32), np.asarray(image_points, dtype=np.float32)


def add_observations(observations, detections, board_config):
    for detection in detections:
        marker_id = detection["marker_id"]
        world_corners = board_marker_world_corners(board_config["boards"][detection["board"]], marker_id)
        for corner_index, image_point in enumerate(detection["image"]):
            key = (marker_id, corner_index)
            observations.setdefault(key, {"world": world_corners[corner_index], "images": []})
            observations[key]["images"].append(image_point)


def aggregate_observations(observations, min_observations, outlier_px):
    object_points = []
    image_points = []
    used = 0
    raw = 0
    for key in sorted(observations):
        entry = observations[key]
        images = np.asarray(entry["images"], dtype=np.float32)
        raw += len(images)
        if len(images) < min_observations:
            continue
        median = np.median(images, axis=0)
        distances = np.linalg.norm(images - median, axis=1)
        keep = distances <= outlier_px
        if np.count_nonzero(keep) < min_observations:
            continue
        kept = images[keep]
        object_points.append(entry["world"])
        image_points.append(np.mean(kept, axis=0))
        used += int(len(kept))
    return np.asarray(object_points, dtype=np.float32), np.asarray(image_points, dtype=np.float32), raw, used


def solve_camera_pose(object_points, image_points, camera_matrix, dist_coeffs):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=6.0,
        iterationsCount=200,
        confidence=0.999,
    )
    if not ok:
        return None

    inlier_points = inliers.reshape(-1) if inliers is not None else np.arange(len(object_points))
    ok, rvec, tvec = cv2.solvePnP(
        object_points[inlier_points],
        image_points[inlier_points],
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "rvec": rvec,
        "tvec": tvec,
        "rotation": rotation,
        "projection": camera_matrix @ np.hstack([rotation, tvec]),
        "camera_center": -rotation.T @ tvec,
        "rms_px": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
        "inliers": int(len(inlier_points)),
    }


def prior_status(position_key, camera_center, geometry):
    prior = camera_center_prior(position_key, geometry)
    if prior is None:
        return None
    delta = camera_center.reshape(3) - prior
    return {
        "expected_center_m": prior.astype(float).tolist(),
        "delta_m": delta.astype(float).tolist(),
        "distance_m": float(np.linalg.norm(delta)),
    }


def annotate(frame, camera_id, position, detections, result=None):
    shown = frame.copy()
    if detections:
        corners = [detection["image"].reshape(1, 4, 2) for detection in detections]
        ids = np.asarray([[detection["marker_id"]] for detection in detections], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(shown, corners, ids)
    label = f"cam {camera_id} {POSITION_LABELS.get(position, position)}: {len(detections)} tags"
    if result is not None:
        label += f" rms {result['rms_px']:.2f}px"
    cv2.rectangle(shown, (8, 8), (min(shown.shape[1] - 8, 660), 48), (0, 0, 0), -1)
    cv2.putText(shown, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def write_contact_sheet(path, frames):
    if not frames:
        return
    tiles = [resize_to_height(frame, 360) for frame in frames]
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), 3):
        row = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + 3]]
        while len(row) < 3:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    args = parse_args()
    if args.write_board_template:
        write_board_template(args.board_config, args.geometry)
        return

    assignments = load_position_assignments(args.positions_config)
    if not assignments:
        raise RuntimeError(f"No position assignments found in {args.positions_config}. Assign cameras in the 9-camera grid first.")
    cameras_to_open = args.cameras or sorted(assignments)
    board_config = load_json(args.board_config)
    geometry = load_json(args.geometry)
    marker_lookup = build_marker_lookup(board_config)
    detector = make_detector(args.dictionary)
    intrinsics = np.load(args.intrinsics)

    cameras = []
    observations = {camera_id: {} for camera_id in cameras_to_open}
    latest_frames = {}
    latest_detections = {}
    outputs = {}
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_intrinsics": str(args.intrinsics),
        "source_positions": str(args.positions_config),
        "source_boards": str(args.board_config),
        "dictionary": args.dictionary,
        "cameras": {},
    }

    try:
        for camera_id in cameras_to_open:
            if camera_id not in assignments:
                print(f"Skipping camera {camera_id}: no grid position assignment", flush=True)
                continue
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id} as {POSITION_LABELS.get(assignments[camera_id], assignments[camera_id])}", flush=True)

        time.sleep(args.warmup)
        for frame_index in range(args.frames):
            if args.interval > 0:
                time.sleep(args.interval)
            for camera in cameras:
                frame = camera.latest_frame
                if frame is None:
                    continue
                camera_id = camera.device_index
                position = assignments[camera_id]
                allowed_boards = expected_boards_for_position(position, board_config)
                frame = rotate_frame(frame, args.rotation)
                detections, _object_points, _image_points = detect_points(
                    frame,
                    detector,
                    board_config,
                    marker_lookup,
                    allowed_boards,
                )
                add_observations(observations[camera_id], detections, board_config)
                latest_frames[camera_id] = frame
                latest_detections[camera_id] = detections
            if (frame_index + 1) % max(1, args.frames // 5) == 0 or frame_index == args.frames - 1:
                counts = [f"cam {camera.device_index}: {len(observations[camera.device_index])} corners" for camera in cameras]
                print(f"Aggregated frame {frame_index + 1}/{args.frames} ({', '.join(counts)})", flush=True)

        annotated = []
        for camera in cameras:
            camera_id = camera.device_index
            key = f"cam_{camera_id}"
            matrix_key = f"{key}_camera_matrix"
            dist_key = f"{key}_dist_coeffs"
            if matrix_key not in intrinsics or dist_key not in intrinsics:
                print(f"cam {camera_id}: missing intrinsics in {args.intrinsics}", flush=True)
                continue

            object_points, image_points, raw_obs, used_obs = aggregate_observations(
                observations[camera_id],
                args.min_observations,
                args.outlier_px,
            )
            detections = latest_detections.get(camera_id, [])
            if len(object_points) < args.min_corners:
                print(f"cam {camera_id}: not enough marker corners ({len(object_points)}/{args.min_corners})", flush=True)
                if camera_id in latest_frames:
                    annotated.append(annotate(latest_frames[camera_id], camera_id, assignments[camera_id], detections))
                continue

            camera_matrix = intrinsics[matrix_key]
            dist_coeffs = intrinsics[dist_key]
            result = solve_camera_pose(object_points, image_points, camera_matrix, dist_coeffs)
            if result is None:
                print(f"cam {camera_id}: solvePnP failed", flush=True)
                continue

            outputs[f"{key}_camera_matrix"] = camera_matrix
            outputs[f"{key}_dist_coeffs"] = dist_coeffs
            outputs[f"{key}_rvec"] = result["rvec"]
            outputs[f"{key}_tvec"] = result["tvec"]
            outputs[f"{key}_projection"] = result["projection"]
            outputs[f"{key}_camera_center"] = result["camera_center"]
            image_size_key = f"{key}_image_size"
            if image_size_key in intrinsics:
                outputs[image_size_key] = intrinsics[image_size_key]

            summary["cameras"][key] = {
                "position": assignments[camera_id],
                "position_label": POSITION_LABELS.get(assignments[camera_id], assignments[camera_id]),
                "allowed_boards": sorted(expected_boards_for_position(assignments[camera_id], board_config)),
                "points": int(len(object_points)),
                "raw_observations": int(raw_obs),
                "used_observations": int(used_obs),
                "inliers": result["inliers"],
                "rms_px": result["rms_px"],
                "mean_px": result["mean_px"],
                "max_px": result["max_px"],
                "camera_center_m": result["camera_center"].reshape(-1).astype(float).tolist(),
                "prior": prior_status(assignments[camera_id], result["camera_center"], geometry),
                "rvec": result["rvec"].reshape(-1).astype(float).tolist(),
                "tvec_m": result["tvec"].reshape(-1).astype(float).tolist(),
            }
            prior = summary["cameras"][key]["prior"]
            if prior is not None and prior["distance_m"] > 0.15:
                print(
                    f"cam {camera_id}: warning solved center is {prior['distance_m']*1000:.0f} mm from "
                    f"the {POSITION_LABELS.get(assignments[camera_id], assignments[camera_id])} prior",
                    flush=True,
                )
            if result["rms_px"] > args.reprojection_warn_px:
                print(f"cam {camera_id}: warning high reprojection RMS {result['rms_px']:.2f}px", flush=True)
            print(
                f"cam {camera_id}: solved {POSITION_LABELS.get(assignments[camera_id], assignments[camera_id])} "
                f"from {len(object_points)} corners, rms {result['rms_px']:.2f}px",
                flush=True,
            )
            if camera_id in latest_frames:
                annotated.append(annotate(latest_frames[camera_id], camera_id, assignments[camera_id], detections, result))

        if outputs:
            np.savez(args.output, **outputs)
            summary_path = args.output.with_suffix(".json")
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")
            print(f"Wrote {summary_path}")
        else:
            print("No camera had enough marker corners to save extrinsics.", flush=True)

        if annotated:
            write_contact_sheet(args.annotated, annotated)
            print(f"Wrote annotated detection sheet: {args.annotated}", flush=True)
    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
