import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from dshow_arducam_viewer import DShowCamera, find_format_index, fit_to_tile, resize_to_height


def parse_args():
    parser = argparse.ArgumentParser(description="First-pass multi-view mouse position fusion.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--poses", type=Path, default=Path("camera_poses.npz"))
    parser.add_argument(
        "--preview-2d",
        action="store_true",
        help="Run blob detection overlays without calibrated 3D fusion.",
    )
    parser.add_argument("--threshold", type=int, default=70, help="Blob threshold in grayscale.")
    parser.add_argument("--invert", action="store_true", help="Detect bright blobs instead of dark blobs.")
    parser.add_argument("--min-area", type=float, default=200.0)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--display-height", type=int, default=360)
    return parser.parse_args()


def load_projections(path, camera_ids):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run calibration first, or use --preview-2d to test mouse detection only."
        )
    data = np.load(path)
    projections = {}
    for camera_id in camera_ids:
        key = f"cam_{camera_id}_projection"
        if key not in data:
            raise RuntimeError(f"Missing projection matrix in {path}: {key}")
        projections[camera_id] = data[key]
    return projections


def detect_mouse_blob(frame, threshold, invert, min_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray, threshold, 255, mode)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    angle = None
    if len(contour) >= 5:
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
    return {
        "center": center,
        "contour": contour,
        "area": cv2.contourArea(contour),
        "angle": angle,
        "mask": mask,
    }


def triangulate(observations, projections):
    if len(observations) < 2:
        return None
    rows = []
    for camera_id, point in observations.items():
        projection = projections[camera_id]
        x, y = point
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-9:
        return None
    return homogeneous[:3] / homogeneous[3]


def annotate_frame(frame, camera_id, detection, xyz):
    annotated = frame.copy()
    if detection is not None:
        center = tuple(np.round(detection["center"]).astype(int))
        cv2.drawContours(annotated, [detection["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(annotated, center, 6, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"area {detection['area']:.0f}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    xyz_text = "XYZ: waiting"
    if xyz is not None:
        xyz_text = f"XYZ m: {xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}"
    cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 560), 48), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"Camera {camera_id} | {xyz_text}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def make_grid(frames, cols, display_height):
    tiles = [resize_to_height(frame, display_height) for frame in frames]
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), cols):
        row_tiles = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + cols]]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def main():
    args = parse_args()
    projections = None if args.preview_2d else load_projections(args.poses, args.cameras)
    cameras = []
    try:
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}")

        cv2.namedWindow("Mouse Fusion", cv2.WINDOW_NORMAL)
        while True:
            observations = {}
            detections = {}
            for camera in cameras:
                if camera.latest_frame is None:
                    continue
                detection = detect_mouse_blob(camera.latest_frame, args.threshold, args.invert, args.min_area)
                detections[camera.device_index] = detection
                if detection is not None:
                    observations[camera.device_index] = detection["center"]

            xyz = None if projections is None else triangulate(observations, projections)
            annotated_frames = []
            for camera in cameras:
                if camera.latest_frame is None:
                    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    annotated_frames.append(blank)
                else:
                    annotated_frames.append(
                        annotate_frame(camera.latest_frame, camera.device_index, detections.get(camera.device_index), xyz)
                    )

            cv2.imshow("Mouse Fusion", make_grid(annotated_frames, args.cols, args.display_height))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
