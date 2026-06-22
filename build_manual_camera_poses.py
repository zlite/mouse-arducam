import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TEMPLATE = {
    "units": "meters",
    "coordinate_frame": {
        "origin": "arena center on floor",
        "x": "from left wall to right wall",
        "y": "along arena length, away from the end facing you",
        "z": "up",
    },
    "up_world": [0.0, 0.0, 1.0],
    "cameras": {
        "left_1": {
            "camera_index": 2,
            "center_m": [-0.30, -0.20, 0.20],
            "look_at_m": [0.0, -0.20, 0.05],
        },
        "left_2": {
            "camera_index": 1,
            "center_m": [-0.30, 0.20, 0.20],
            "look_at_m": [0.0, 0.20, 0.05],
        },
        "right_1": {
            "camera_index": 4,
            "center_m": [0.30, -0.20, 0.20],
            "look_at_m": [0.0, -0.20, 0.05],
        },
        "right_2": {
            "camera_index": 3,
            "center_m": [0.30, 0.20, 0.20],
            "look_at_m": [0.0, 0.20, 0.05],
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build approximate camera_poses.npz from manually measured rig geometry."
    )
    parser.add_argument("--intrinsics", type=Path, default=Path("camera_intrinsics.npz"))
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("camera_poses_manual.npz"))
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a starter manual_rig_geometry.json and exit.",
    )
    return parser.parse_args()


def normalize(vector, name):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError(f"{name} has near-zero length")
    return vector / norm


def look_at_to_rt(center, look_at, up_world):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3)
    up_world = normalize(up_world, "up_world")

    z_axis = normalize(look_at - center, "look direction")
    x_axis = normalize(np.cross(z_axis, up_world), "camera right axis")
    y_axis = np.cross(z_axis, x_axis)

    # Rows of R are camera axes in world coordinates.
    # OpenCV camera coordinates are x right, y down, z forward.
    rotation = np.vstack([x_axis, y_axis, z_axis])
    tvec = -rotation @ center.reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rotation)
    return rotation, rvec, tvec, center.reshape(3, 1)


def write_template(path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {path}")
    path.write_text(json.dumps(DEFAULT_TEMPLATE, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    print("Edit center_m and look_at_m with measured values, then rerun without --write-template.")


def main():
    args = parse_args()
    if args.write_template:
        write_template(args.geometry)
        return

    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    intrinsics = np.load(args.intrinsics)
    up_world = geometry.get("up_world", [0.0, 0.0, 1.0])
    output = {}
    summary = {
        "source_geometry": str(args.geometry),
        "source_intrinsics": str(args.intrinsics),
        "coordinate_frame": geometry.get("coordinate_frame", {}),
        "cameras": {},
    }

    for position_name, spec in geometry["cameras"].items():
        camera_id = int(spec["camera_index"])
        key = f"cam_{camera_id}"
        matrix_key = f"{key}_camera_matrix"
        dist_key = f"{key}_dist_coeffs"
        if matrix_key not in intrinsics:
            raise KeyError(f"Missing {matrix_key} in {args.intrinsics}")
        if dist_key not in intrinsics:
            raise KeyError(f"Missing {dist_key} in {args.intrinsics}")

        rotation, rvec, tvec, center = look_at_to_rt(spec["center_m"], spec["look_at_m"], up_world)
        camera_matrix = intrinsics[matrix_key]
        dist_coeffs = intrinsics[dist_key]
        projection = camera_matrix @ np.hstack([rotation, tvec])

        output[f"{key}_camera_matrix"] = camera_matrix
        output[f"{key}_dist_coeffs"] = dist_coeffs
        output[f"{key}_rvec"] = rvec
        output[f"{key}_tvec"] = tvec
        output[f"{key}_projection"] = projection
        output[f"{key}_camera_center"] = center

        image_size_key = f"{key}_image_size"
        if image_size_key in intrinsics:
            output[image_size_key] = intrinsics[image_size_key]

        summary["cameras"][key] = {
            "position": position_name,
            "center_m": center.reshape(-1).tolist(),
            "look_at_m": spec["look_at_m"],
            "rvec": rvec.reshape(-1).tolist(),
            "tvec_m": tvec.reshape(-1).tolist(),
        }
        print(f"Camera {camera_id} ({position_name}): manual pose built")

    np.savez(args.output, **output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
