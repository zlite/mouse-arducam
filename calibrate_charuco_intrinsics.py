import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from ChArUco frame captures.")
    parser.add_argument("--frames-dir", type=Path, default=Path("calibration_frames"))
    parser.add_argument("--output", type=Path, default=Path("camera_intrinsics.npz"))
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--model-camera", default=None, help="Camera ID whose solved intrinsics should be copied.")
    parser.add_argument("--copy-to-cameras", type=int, nargs="*", default=[], help="Copy --model-camera intrinsics to these camera IDs.")
    return parser.parse_args()


def make_board(args):
    if not hasattr(cv2.aruco, args.dictionary):
        raise RuntimeError(f"OpenCV has no aruco dictionary named {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )
    return dictionary, board


def detect_charuco(image, dictionary, board):
    detector = cv2.aruco.CharucoDetector(board)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None:
        return None, None
    return charuco_corners.reshape(-1, 2).astype(np.float32), charuco_ids.flatten().astype(np.int32)


def main():
    args = parse_args()
    dictionary, board = make_board(args)
    board_corners = board.getChessboardCorners().astype(np.float32)
    image_points_by_camera = defaultdict(list)
    object_points_by_camera = defaultdict(list)
    image_size_by_camera = {}

    for path in sorted(args.frames_dir.glob("set_*_cam_*.png")):
        camera_id = path.stem.split("_cam_")[-1]
        image = cv2.imread(str(path))
        if image is None:
            continue
        image_size_by_camera[camera_id] = (image.shape[1], image.shape[0])
        image_points, ids = detect_charuco(image, dictionary, board)
        if image_points is None or len(ids) < args.min_corners:
            print(f"{path.name}: skipped, not enough ChArUco corners")
            continue
        object_points = board_corners[ids]
        object_points_by_camera[camera_id].append(object_points)
        image_points_by_camera[camera_id].append(image_points)
        print(f"{path.name}: camera {camera_id}, {len(ids)} corners")

    output = {}
    summary = {}
    for camera_id in sorted(object_points_by_camera, key=int):
        object_points = object_points_by_camera[camera_id]
        image_points = image_points_by_camera[camera_id]
        image_size = image_size_by_camera[camera_id]
        if len(object_points) < 3:
            print(f"Camera {camera_id}: skipped, need at least 3 valid board views")
            continue
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        key = f"cam_{camera_id}"
        output[f"{key}_camera_matrix"] = camera_matrix
        output[f"{key}_dist_coeffs"] = dist_coeffs
        output[f"{key}_image_size"] = np.array(image_size)
        summary[key] = {
            "rms": float(rms),
            "valid_views": len(object_points),
            "image_size": list(image_size),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        }
        print(f"Camera {camera_id}: RMS={rms:.3f}, valid views={len(object_points)}")

    if args.copy_to_cameras:
        model_camera = args.model_camera
        if model_camera is None:
            if len(summary) != 1:
                raise RuntimeError("--model-camera is required when more than one camera was calibrated")
            model_camera = next(iter(summary)).removeprefix("cam_")
        model_key = f"cam_{model_camera}"
        matrix_key = f"{model_key}_camera_matrix"
        dist_key = f"{model_key}_dist_coeffs"
        size_key = f"{model_key}_image_size"
        if matrix_key not in output or dist_key not in output or size_key not in output:
            raise RuntimeError(f"Cannot copy intrinsics: {model_key} was not calibrated")
        for camera_id in args.copy_to_cameras:
            key = f"cam_{camera_id}"
            output[f"{key}_camera_matrix"] = output[matrix_key]
            output[f"{key}_dist_coeffs"] = output[dist_key]
            output[f"{key}_image_size"] = output[size_key]
            summary.setdefault(key, {})
            summary[key].update(
                {
                    "copied_from": model_key,
                    "image_size": output[size_key].tolist(),
                    "camera_matrix": output[matrix_key].tolist(),
                    "dist_coeffs": output[dist_key].reshape(-1).tolist(),
                }
            )
        print(f"Copied {model_key} intrinsics to cameras: {' '.join(str(i) for i in args.copy_to_cameras)}")

    np.savez(args.output, **output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
