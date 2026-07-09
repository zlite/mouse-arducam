#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import rtoml

from caliscope.api import CameraArray, CaptureVolume, Charuco, CharucoTracker, extract_image_points_multicam


def load_charuco(path: Path) -> Charuco:
    cfg = rtoml.load(path)
    return Charuco(
        columns=cfg["columns"],
        rows=cfg["rows"],
        board_height=cfg["board_height"],
        board_width=cfg["board_width"],
        dictionary=cfg.get("dictionary", "DICT_4X4_50"),
        units=cfg.get("units", "cm"),
        aruco_scale=cfg.get("aruco_scale", 0.75),
        square_size_override_cm=cfg.get("square_size_override_cm"),
        inverted=cfg.get("inverted", False),
        legacy_pattern=cfg.get("legacy_pattern", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Caliscope Charuco extrinsics headlessly.")
    parser.add_argument("--workspace", type=Path, default=Path("/home/cat/calibration"))
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--initial-nfev", type=int, default=120)
    parser.add_argument("--final-nfev", type=int, default=80)
    parser.add_argument("--filter-percent", type=float, default=2.5)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    extrinsic_dir = workspace / "calibration" / "extrinsic"
    target_path = workspace / "calibration" / "targets" / "intrinsic_charuco.toml"
    camera_array_path = workspace / "camera_array.toml"
    charuco_dir = extrinsic_dir / "CHARUCO"
    capture_volume_dir = extrinsic_dir / "capture_volume"

    camera_array = CameraArray.from_toml(camera_array_path)
    videos = {
        cam_id: extrinsic_dir / f"cam_{cam_id}.mp4"
        for cam_id in sorted(camera_array.cameras)
        if (extrinsic_dir / f"cam_{cam_id}.mp4").exists()
    }
    if not videos:
        raise SystemExit(f"No cam_*.mp4 videos found in {extrinsic_dir}")

    print(f"Workspace: {workspace}")
    print(f"Videos: {', '.join(f'cam_{cam_id}' for cam_id in videos)}")
    print(f"Frame step: {args.frame_step}")

    tracker = CharucoTracker(load_charuco(target_path))
    image_points = extract_image_points_multicam(videos, tracker, frame_step=args.frame_step)

    counts = image_points.df.groupby("cam_id").size().sort_index()
    sync_count = image_points.df["sync_index"].nunique()
    print(f"Detected {len(image_points.df)} Charuco observations across {sync_count} sync indices")
    for cam_id, count in counts.items():
        print(f"  cam_{int(cam_id)}: {int(count)} observations")

    charuco_dir.mkdir(parents=True, exist_ok=True)
    image_points.to_csv(charuco_dir / "image_points.csv")

    print("Bootstrapping camera poses...")
    capture_volume = CaptureVolume.bootstrap(image_points, camera_array)
    print(f"Bootstrap RMSE: {capture_volume.reprojection_report.overall_rmse:.3f}px")

    print(f"Initial optimization, max_nfev={args.initial_nfev}...")
    optimized = capture_volume.optimize(ftol=1e-6, max_nfev=args.initial_nfev, verbose=1, strict=False)
    print(f"Initial optimized RMSE: {optimized.reprojection_report.overall_rmse:.3f}px")

    print(f"Filtering worst {args.filter_percent}% per camera...")
    filtered = optimized.filter_by_percentile_error(args.filter_percent)

    print(f"Final optimization, max_nfev={args.final_nfev}...")
    final = filtered.optimize(ftol=1e-6, max_nfev=args.final_nfev, verbose=1, strict=False)
    report = final.reprojection_report
    status = final.optimization_status

    print(f"Final RMSE: {report.overall_rmse:.3f}px")
    print(f"Matched observations: {report.n_observations_matched}/{report.n_observations_total}")
    if status is not None:
        print(
            "Final optimizer: "
            f"converged={status.converged}, reason={status.termination_reason}, "
            f"iterations={status.iterations}, cost={status.final_cost:.3f}"
        )
    for cam_id, rmse in sorted(report.by_camera.items()):
        print(f"  cam_{cam_id}: {rmse:.3f}px")

    scale = final.compute_volumetric_scale_accuracy()
    print(f"Volumetric scale RMSE: {scale.pooled_rmse_mm:.2f}mm over {scale.n_frames_sampled} frames")

    capture_volume_dir.mkdir(parents=True, exist_ok=True)
    final.save(capture_volume_dir)
    final.save(charuco_dir)
    final.camera_array.to_toml(camera_array_path)
    final.camera_array.to_aniposelib_toml(workspace / "camera_array_aniposelib.toml")
    print(f"Saved capture volume to {capture_volume_dir}")
    print(f"Updated {camera_array_path}")


if __name__ == "__main__":
    main()
