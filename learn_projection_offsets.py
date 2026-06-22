import argparse
import csv
import json
from pathlib import Path

import numpy as np

import live_mouse_fusion as fusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn per-camera spatial-prior projection offsets from manual click datalogs."
    )
    parser.add_argument("--datalog", type=Path, default=Path("mouse_fusion_datalog.csv"))
    parser.add_argument("--poses", type=Path, default=Path("camera_poses_manual.npz"))
    parser.add_argument("--output", type=Path, default=Path("mouse_fusion_projection_offsets.json"))
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--point-transform", default="flip_xy", choices=("normal", "flip_x", "flip_y", "flip_xy"))
    parser.add_argument("--min-clicked-cameras", type=int, default=4)
    return parser.parse_args()


def clicked_camera_ids(row):
    return [int(item) for item in row["clicked_cameras"].split() if item.strip()]


def main():
    args = parse_args()
    rows = list(csv.DictReader(args.datalog.open(newline="", encoding="utf-8")))
    usable = [
        row
        for row in rows
        if row.get("xyz_x_m")
        and len(clicked_camera_ids(row)) >= args.min_clicked_cameras
        and not int(row.get("clamped", "0") or 0)
    ]
    if not usable:
        raise RuntimeError(f"No usable rows found in {args.datalog}")

    camera_models = fusion.load_camera_models(args.poses, args.cameras)
    raw_shape = (args.height, args.width, 3)
    corrections = {}
    for camera_id in args.cameras:
        deltas = []
        for row in usable:
            if not row.get(f"cam_{camera_id}_display_x"):
                continue
            xyz = np.array(
                [
                    float(row["xyz_x_m"]),
                    float(row["xyz_y_m"]),
                    float(row["xyz_z_m"]),
                ],
                dtype=np.float64,
            )
            projected = fusion.project_world_to_display(
                camera_id,
                xyz,
                camera_models,
                raw_shape,
                args.rotation,
                args.point_transform,
                projection_offsets=None,
            )
            if projected is None:
                continue
            clicked = np.array(
                [
                    float(row[f"cam_{camera_id}_display_x"]),
                    float(row[f"cam_{camera_id}_display_y"]),
                ],
                dtype=np.float32,
            )
            deltas.append(clicked - projected)
        if not deltas:
            continue
        median = np.median(np.asarray(deltas), axis=0)
        corrections[str(camera_id)] = {
            "dx": float(median[0]),
            "dy": float(median[1]),
            "samples": len(deltas),
        }
        print(
            f"cam {camera_id}: dx={median[0]:+.1f}px dy={median[1]:+.1f}px "
            f"from {len(deltas)} samples"
        )

    output = {
        "source": str(args.datalog),
        "point_transform": args.point_transform,
        "note": (
            "Per-camera display-pixel correction from manual mouse ground-contact clicks. "
            "This corrects the projected spatial-prior target; it is not a full physical extrinsics solve."
        ),
        "corrections_px": corrections,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
