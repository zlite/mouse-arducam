import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from images of a flat ArUco marker grid."
    )
    parser.add_argument("--frames-dir", type=Path, default=Path("calibration_frames_1280x720"))
    parser.add_argument("--camera", type=int, required=True, help="Camera index to use as the model camera.")
    parser.add_argument("--copy-to-cameras", type=int, nargs="*", default=[], help="Write the solved model to these camera IDs too.")
    parser.add_argument("--output", type=Path, default=Path("camera_intrinsics_1280x720_shared.npz"))
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--markers-x", type=int, default=7)
    parser.add_argument("--markers-y", type=int, default=5)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--id-order", choices=("row-major", "reverse-row-major"), default="row-major")
    parser.add_argument("--corner-shift", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--marker-mm", type=float, default=27.0)
    parser.add_argument("--separation-mm", type=float, default=7.0)
    parser.add_argument("--min-markers", type=int, default=8)
    return parser.parse_args()


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


def make_board(args):
    if not hasattr(cv2.aruco, args.dictionary):
        raise RuntimeError(f"OpenCV has no aruco dictionary named {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    marker_count = args.markers_x * args.markers_y
    if args.id_order == "reverse-row-major":
        ids = np.arange(args.first_id + marker_count - 1, args.first_id - 1, -1, dtype=np.int32)
    else:
        ids = np.arange(args.first_id, args.first_id + marker_count, dtype=np.int32)
    return cv2.aruco.GridBoard(
        (args.markers_x, args.markers_y),
        args.marker_mm / 1000.0,
        args.separation_mm / 1000.0,
        dictionary,
        ids,
    )


def marker_object_corners(marker_id, args):
    marker_count = args.markers_x * args.markers_y
    if args.id_order == "reverse-row-major":
        local_id = args.first_id + marker_count - 1 - int(marker_id)
    else:
        local_id = int(marker_id) - args.first_id
    if local_id < 0 or local_id >= marker_count:
        return None
    row = local_id // args.markers_x
    col = local_id % args.markers_x
    marker_m = args.marker_mm / 1000.0
    pitch_m = (args.marker_mm + args.separation_mm) / 1000.0
    x0 = col * pitch_m
    y0 = row * pitch_m
    corners = np.asarray(
        [
            [x0, y0, 0.0],
            [x0 + marker_m, y0, 0.0],
            [x0 + marker_m, y0 + marker_m, 0.0],
            [x0, y0 + marker_m, 0.0],
        ],
        dtype=np.float32,
    )
    return np.roll(corners, -args.corner_shift, axis=0)


def collect_points(args):
    detector = make_detector(args.dictionary)
    object_points = []
    image_points = []
    image_size = None
    pattern = f"set_*_cam_{args.camera}.png"
    paths = sorted(args.frames_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"No frames matched {args.frames_dir / pattern}")

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image_size = (image.shape[1], image.shape[0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            print(f"{path.name}: skipped, 0 markers")
            continue

        frame_object_points = []
        frame_image_points = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            world = marker_object_corners(int(marker_id), args)
            if world is None:
                continue
            frame_object_points.extend(world)
            frame_image_points.extend(marker_corners.reshape(4, 2).astype(np.float32))
        frame_object_points = np.asarray(frame_object_points, dtype=np.float32).reshape(-1, 3)
        frame_image_points = np.asarray(frame_image_points, dtype=np.float32).reshape(-1, 2)
        marker_count = len(frame_object_points) // 4
        if marker_count < args.min_markers:
            print(f"{path.name}: skipped, {marker_count} usable markers")
            continue
        object_points.append(frame_object_points)
        image_points.append(frame_image_points)
        print(f"{path.name}: accepted, {marker_count} usable markers")

    if image_size is None:
        raise RuntimeError(f"No readable frames found in {args.frames_dir}")
    if len(object_points) < 5:
        raise RuntimeError(f"Need at least 5 accepted views; found {len(object_points)}")
    return object_points, image_points, image_size


def main():
    args = parse_args()
    object_points, image_points, image_size = collect_points(args)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    output = {}
    camera_ids = sorted(set([args.camera] + args.copy_to_cameras))
    for camera_id in camera_ids:
        key = f"cam_{camera_id}"
        output[f"{key}_camera_matrix"] = camera_matrix
        output[f"{key}_dist_coeffs"] = dist_coeffs
        output[f"{key}_image_size"] = np.asarray(image_size, dtype=np.int32)

    np.savez(args.output, **output)
    summary = {
        "model_camera": args.camera,
        "copied_to_cameras": camera_ids,
        "source_frames": str(args.frames_dir),
        "dictionary": args.dictionary,
        "grid": {
            "markers_x": args.markers_x,
            "markers_y": args.markers_y,
            "first_id": args.first_id,
            "id_order": args.id_order,
            "corner_shift": args.corner_shift,
            "marker_m": args.marker_mm / 1000.0,
            "separation_m": args.separation_mm / 1000.0,
        },
        "valid_views": len(object_points),
        "image_size": list(image_size),
        "rms": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Camera {args.camera}: RMS={rms:.3f}, valid views={len(object_points)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
