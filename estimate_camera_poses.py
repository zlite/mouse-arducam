import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_charuco_intrinsics import detect_charuco, make_board


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate mounted camera poses from an origin ChArUco capture.")
    parser.add_argument("--origin-dir", type=Path, default=Path("origin_frames"))
    parser.add_argument("--intrinsics", type=Path, default=Path("camera_intrinsics.npz"))
    parser.add_argument("--output", type=Path, default=Path("camera_poses.npz"))
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--min-corners", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    dictionary, board = make_board(args)
    board_corners = board.getChessboardCorners().astype(np.float32)
    intrinsics = np.load(args.intrinsics)
    output = {}
    summary = {}

    for path in sorted(args.origin_dir.glob("set_*_cam_*.png")):
        camera_id = path.stem.split("_cam_")[-1]
        key = f"cam_{camera_id}"
        matrix_key = f"{key}_camera_matrix"
        dist_key = f"{key}_dist_coeffs"
        if matrix_key not in intrinsics:
            print(f"{path.name}: no intrinsics for camera {camera_id}")
            continue

        image = cv2.imread(str(path))
        if image is None:
            continue
        image_points, ids = detect_charuco(image, dictionary, board)
        if image_points is None or len(ids) < args.min_corners:
            print(f"{path.name}: skipped, not enough ChArUco corners")
            continue

        object_points = board_corners[ids]
        camera_matrix = intrinsics[matrix_key]
        dist_coeffs = intrinsics[dist_key]
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            print(f"{path.name}: solvePnP failed")
            continue

        rotation, _ = cv2.Rodrigues(rvec)
        projection = camera_matrix @ np.hstack([rotation, tvec])
        camera_center = -rotation.T @ tvec
        output[f"{key}_camera_matrix"] = camera_matrix
        output[f"{key}_dist_coeffs"] = dist_coeffs
        output[f"{key}_rvec"] = rvec
        output[f"{key}_tvec"] = tvec
        output[f"{key}_projection"] = projection
        output[f"{key}_camera_center"] = camera_center
        summary[key] = {
            "source": path.name,
            "corners": int(len(ids)),
            "rvec": rvec.reshape(-1).tolist(),
            "tvec_m": tvec.reshape(-1).tolist(),
            "camera_center_m": camera_center.reshape(-1).tolist(),
        }
        print(f"Camera {camera_id}: pose solved from {len(ids)} corners")

    np.savez(args.output, **output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
