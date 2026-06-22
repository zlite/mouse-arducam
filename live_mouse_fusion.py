import argparse
import csv
import json
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
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--poses", type=Path, default=Path("camera_poses.npz"))
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--top-down", action="store_true", help="Show a synthetic top-down arena panel.")
    parser.add_argument("--top-down-size", type=int, default=420)
    parser.add_argument(
        "--position-method",
        choices=("floor", "triangulate"),
        default="floor",
        help="Estimate top-down position by intersecting detection rays with the floor, or by 3D triangulation.",
    )
    parser.add_argument("--floor-z", type=float, default=0.0, help="World z height used by --position-method floor.")
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Directory of empty-arena frames named like set_0000_cam_0.png for background subtraction.",
    )
    parser.add_argument(
        "--background-threshold",
        type=int,
        default=25,
        help="Pixel difference threshold used with --background-dir.",
    )
    parser.add_argument(
        "--background-thresholds",
        default=None,
        help="Optional per-camera background thresholds, e.g. 0:60,1:60,2:25,3:25.",
    )
    parser.add_argument(
        "--preview-2d",
        action="store_true",
        help="Run blob detection overlays without calibrated 3D fusion.",
    )
    parser.add_argument("--threshold", type=int, default=70, help="Blob threshold in grayscale.")
    parser.add_argument("--invert", action="store_true", help="Detect bright blobs instead of dark blobs.")
    parser.add_argument("--min-area", type=float, default=200.0)
    parser.add_argument("--candidate-count", type=int, default=6, help="Number of blob candidates to keep per camera.")
    parser.add_argument(
        "--morph-open",
        type=int,
        default=3,
        help="Morphological open kernel size for cleanup. Use odd values; 0 disables.",
    )
    parser.add_argument(
        "--morph-close",
        type=int,
        default=15,
        help="Morphological close kernel size for joining mouse fragments. Use odd values; 0 disables.",
    )
    parser.add_argument(
        "--morph-dilate",
        type=int,
        default=1,
        help="Dilation iterations after closing, to grow partial mouse blobs.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.08,
        help="Ignore contours larger than this fraction of the frame.",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=8,
        help="Ignore contours touching this many pixels from the image border.",
    )
    parser.add_argument(
        "--roi-y-min-ratio",
        type=float,
        default=0.42,
        help="Only detect below this fraction of the displayed image height.",
    )
    parser.add_argument(
        "--roi-y-max-ratio",
        type=float,
        default=1.0,
        help="Only detect above this fraction of the displayed image height.",
    )
    parser.add_argument(
        "--roi-top-reject-ratio",
        type=float,
        default=0.08,
        help="Reject contours whose bounding box starts this close to the top of the ROI.",
    )
    parser.add_argument("--roi-line-thickness", type=int, default=3)
    parser.add_argument(
        "--allow-outside-fallback",
        action="store_true",
        help="If all floor intersections land outside the arena, clamp and display their median anyway.",
    )
    parser.add_argument(
        "--floor-cluster-radius",
        type=float,
        default=0.08,
        help="Meters. Candidate floor hits within this radius can support the same mouse position.",
    )
    parser.add_argument(
        "--floor-candidate-margin",
        type=float,
        default=0.20,
        help="Meters outside the measured arena to allow while clustering floor candidates.",
    )
    parser.add_argument(
        "--hide-unselected-candidates",
        action="store_true",
        help="Only draw the geometry-selected candidate contour in each camera.",
    )
    parser.add_argument(
        "--manual-clicks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow clicking the mouse in camera tiles to override detections for that camera.",
    )
    parser.add_argument(
        "--manual-click-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a simple wall-based floor estimate if clicked rays miss the floor.",
    )
    parser.add_argument(
        "--manual-lock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep manual clicks as a permanent XYZ override instead of using them as a tracking seed.",
    )
    parser.add_argument(
        "--point-transform",
        choices=("auto", "normal", "flip_x", "flip_y", "flip_xy"),
        default="flip_xy",
        help="How displayed points map into the camera model. Datalogs indicate flip_xy for this rig.",
    )
    parser.add_argument(
        "--spatial-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the last fused top-down position to prefer blobs near the expected camera location.",
    )
    parser.add_argument(
        "--spatial-prior-radius",
        type=float,
        default=170.0,
        help="Pixel radius around the projected prior used to filter blob candidates.",
    )
    parser.add_argument(
        "--spatial-prior-min-keep",
        type=int,
        default=1,
        help="Keep this many nearest candidates if no blob falls inside the spatial-prior radius.",
    )
    parser.add_argument(
        "--projection-offsets",
        type=Path,
        default=Path("mouse_fusion_projection_offsets.json"),
        help="Optional per-camera pixel offsets learned from manual click datalogs.",
    )
    parser.add_argument(
        "--prior-local-threshold",
        type=int,
        default=10,
        help="Lower background-difference threshold inside the spatial-prior circle.",
    )
    parser.add_argument(
        "--prior-local-close",
        type=int,
        default=41,
        help="Close kernel for building one larger local blob inside the spatial-prior circle.",
    )
    parser.add_argument(
        "--prior-local-dilate",
        type=int,
        default=2,
        help="Dilation iterations for the local spatial-prior blob.",
    )
    parser.add_argument(
        "--template-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mouse templates cropped from manual clicks as the primary per-camera detector.",
    )
    parser.add_argument(
        "--template-size",
        type=int,
        default=120,
        help="Square template size in displayed pixels.",
    )
    parser.add_argument(
        "--template-anchor-y",
        type=float,
        default=0.72,
        help="Vertical fraction in the template corresponding to the clicked ground-contact point.",
    )
    parser.add_argument(
        "--template-search-radius",
        type=float,
        default=230.0,
        help="Pixels around the projected prior or previous match to search for template matches.",
    )
    parser.add_argument(
        "--template-min-score",
        type=float,
        default=0.30,
        help="Minimum normalized template-match score accepted as a detection.",
    )
    parser.add_argument(
        "--tracker-alpha",
        type=float,
        default=0.35,
        help="Low-pass smoothing factor for accepted XYZ updates. Higher follows motion faster.",
    )
    parser.add_argument(
        "--tracker-max-step",
        type=float,
        default=0.045,
        help="Meters. Reject automatic XYZ updates that jump farther than this from the filtered position.",
    )
    parser.add_argument(
        "--tracker-min-support",
        type=int,
        default=2,
        help="Minimum number of cameras needed before an automatic XYZ update is accepted.",
    )
    parser.add_argument(
        "--tracker-max-spread",
        type=float,
        default=0.075,
        help="Meters. Reject automatic XYZ updates when the supporting camera floor hits are this spread out.",
    )
    parser.add_argument(
        "--datalog",
        type=Path,
        default=Path("mouse_fusion_datalog.csv"),
        help="CSV file for manual click/fusion diagnostics. Use --datalog '' to disable.",
    )
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--display-height", type=int, default=360)
    args = parser.parse_args()
    if args.datalog == Path(""):
        args.datalog = None
    return args


def parse_camera_value_map(text, cast=int):
    values = {}
    if not text:
        return values
    for item in text.split(","):
        if not item.strip():
            continue
        camera_text, value_text = item.split(":", 1)
        values[int(camera_text.strip())] = cast(value_text.strip())
    return values


def load_projections(path, camera_ids):
    path = Path(path)
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


def load_camera_models(path, camera_ids):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run calibration first, or use --preview-2d to test mouse detection only."
        )
    data = np.load(path)
    models = {}
    for camera_id in camera_ids:
        key = f"cam_{camera_id}"
        required = {
            "camera_matrix": f"{key}_camera_matrix",
            "dist_coeffs": f"{key}_dist_coeffs",
            "rvec": f"{key}_rvec",
            "tvec": f"{key}_tvec",
            "camera_center": f"{key}_camera_center",
        }
        missing = [name for name in required.values() if name not in data]
        if missing:
            raise RuntimeError(f"Missing camera model entries in {path}: {missing}")
        rotation, _ = cv2.Rodrigues(data[required["rvec"]])
        models[camera_id] = {
            "camera_matrix": data[required["camera_matrix"]],
            "dist_coeffs": data[required["dist_coeffs"]],
            "rotation": rotation,
            "tvec": data[required["tvec"]].reshape(3, 1),
            "center": data[required["camera_center"]].reshape(3),
        }
    return models


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def rotate_point(point, width, height, degrees):
    x, y = point
    degrees %= 360
    if degrees == 90:
        return np.array([height - 1 - y, x], dtype=np.float32)
    if degrees == 180:
        return np.array([width - 1 - x, height - 1 - y], dtype=np.float32)
    if degrees == 270:
        return np.array([y, width - 1 - x], dtype=np.float32)
    return np.array([x, y], dtype=np.float32)


def rotate_detection(detection, width, height, degrees):
    if detection is None:
        return None
    rotated = dict(detection)
    contour = detection["contour"].reshape(-1, 2)
    rotated_contour = np.array(
        [rotate_point(point, width, height, degrees) for point in contour],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    rotated["center"] = rotate_point(detection["center"], width, height, degrees)
    if "contact" in detection:
        rotated["contact"] = rotate_point(detection["contact"], width, height, degrees)
    rotated["contour"] = rotated_contour
    return rotated


def unrotate_point(point, raw_width, raw_height, degrees):
    x, y = point
    degrees %= 360
    if degrees == 90:
        return np.array([y, raw_height - 1 - x], dtype=np.float32)
    if degrees == 180:
        return np.array([raw_width - 1 - x, raw_height - 1 - y], dtype=np.float32)
    if degrees == 270:
        return np.array([raw_width - 1 - y, x], dtype=np.float32)
    return np.array([x, y], dtype=np.float32)


def contour_touches_border(contour, width, height, margin):
    x, y, w, h = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def contour_score(contour, frame_height):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-9:
        return 0.0
    compactness = 4.0 * math.pi * area / (perimeter * perimeter)
    _x, y, _w, h = cv2.boundingRect(contour)
    center_y = (y + h / 2.0) / max(frame_height, 1)
    lower_weight = 0.4 + (center_y * center_y)
    return area * max(compactness, 0.05) * lower_weight


def load_backgrounds(path, camera_ids, rotation_degrees):
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Remove --background-dir, or capture empty-enclosure frames first:\n"
            f"  python capture_multicam_frames.py --cameras {' '.join(str(camera_id) for camera_id in camera_ids)} "
            f"--out-dir {path} --count 1 --interval 0.2 --warmup 1.0 --width 1280 --height 800 --format MJPG"
        )
    backgrounds = {}
    for camera_id in camera_ids:
        matches = sorted(path.glob(f"*_cam_{camera_id}.png"))
        if not matches:
            available = sorted(found.name for found in path.glob("*.png"))
            raise FileNotFoundError(
                f"No background frame for camera {camera_id} in {path}. "
                f"Expected a file matching *_cam_{camera_id}.png. Available PNGs: {available}"
            )
        grays = []
        for match in matches:
            image = cv2.imread(str(match))
            if image is None:
                raise RuntimeError(f"Could not read background frame: {match}")
            grays.append(cv2.cvtColor(rotate_frame(image, rotation_degrees), cv2.COLOR_BGR2GRAY))
        backgrounds[camera_id] = np.median(np.asarray(grays), axis=0).astype(np.uint8)
        print(f"Background cam {camera_id}: {len(matches)} frames from {path}", flush=True)
    return backgrounds


def detection_from_contour(contour, mask, roi):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    contour_points = contour.reshape(-1, 2)
    contact = contour_points[np.argmax(contour_points[:, 1])].astype(np.float32)
    angle = None
    if len(contour) >= 5:
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
    return {
        "center": center,
        "contact": contact,
        "contour": contour,
        "area": cv2.contourArea(contour),
        "angle": angle,
        "mask": mask,
        "roi": roi,
    }


def detect_mouse_candidates(
    frame,
    threshold,
    invert,
    min_area,
    max_area_ratio,
    border_margin,
    roi_y_min_ratio,
    roi_y_max_ratio,
    roi_top_reject_ratio,
    candidate_count,
    morph_open,
    morph_close,
    morph_dilate,
    background_gray=None,
    background_threshold=25,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if background_gray is not None:
        if background_gray.shape != gray.shape:
            background_gray = cv2.resize(background_gray, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(gray, background_gray)
        _, mask = cv2.threshold(diff, background_threshold, 255, cv2.THRESH_BINARY)
    else:
        mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(gray, threshold, 255, mode)
    height, width = frame.shape[:2]
    roi_y1 = int(np.clip(roi_y_min_ratio, 0.0, 1.0) * height)
    roi_y2 = int(np.clip(roi_y_max_ratio, 0.0, 1.0) * height)
    if roi_y2 < roi_y1:
        roi_y1, roi_y2 = roi_y2, roi_y1
    roi_y2 = max(roi_y1 + 1, roi_y2)
    roi_top_reject_px = int((roi_y2 - roi_y1) * max(0.0, roi_top_reject_ratio))
    mask[:roi_y1, :] = 0
    mask[roi_y2:, :] = 0
    if morph_open > 0:
        open_size = max(1, int(morph_open) | 1)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    if morph_close > 0:
        close_size = max(1, int(morph_close) | 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    if morph_dilate > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, dilate_kernel, iterations=int(morph_dilate))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = width * height * max_area_ratio
    contours = [
        contour
        for contour in contours
        if min_area <= cv2.contourArea(contour) <= max_area
        and not contour_touches_border(contour, width, height, border_margin)
        and cv2.boundingRect(contour)[1] > roi_y1 + roi_top_reject_px
    ]
    if not contours:
        return []

    contours = sorted(contours, key=lambda contour: contour_score(contour, height), reverse=True)
    candidates = []
    for contour in contours[: max(1, candidate_count)]:
        detection = detection_from_contour(contour, mask, (roi_y1, roi_y2))
        if detection is not None:
            candidates.append(detection)
    return candidates


def detect_prior_region_candidates(
    frame,
    background_gray,
    expected_point,
    radius_px,
    min_area,
    max_area_ratio,
    roi_y_min_ratio,
    roi_y_max_ratio,
    threshold,
    close_size,
    dilate_iterations,
):
    if background_gray is None or expected_point is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if background_gray.shape != gray.shape:
        background_gray = cv2.resize(background_gray, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(gray, background_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    height, width = frame.shape[:2]
    roi_y1 = int(np.clip(roi_y_min_ratio, 0.0, 1.0) * height)
    roi_y2 = int(np.clip(roi_y_max_ratio, 0.0, 1.0) * height)
    if roi_y2 < roi_y1:
        roi_y1, roi_y2 = roi_y2, roi_y1
    roi_y2 = max(roi_y1 + 1, roi_y2)
    mask[:roi_y1, :] = 0
    mask[roi_y2:, :] = 0

    local_mask = np.zeros_like(mask)
    center = tuple(np.round(expected_point).astype(int))
    cv2.circle(local_mask, center, int(max(8, radius_px)), 255, -1)
    mask = cv2.bitwise_and(mask, local_mask)

    if close_size > 0:
        size = max(1, int(close_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=int(dilate_iterations))
        mask = cv2.bitwise_and(mask, local_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = width * height * max_area_ratio
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        detection = detection_from_contour(contour, mask, (roi_y1, roi_y2))
        if detection is not None:
            detection["source"] = "prior_local"
            candidates.append(detection)
    return sorted(candidates, key=lambda candidate: candidate["area"], reverse=True)


def crop_with_padding(image, x1, y1, width, height):
    image_height, image_width = image.shape[:2]
    x2 = x1 + width
    y2 = y1 + height
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(image_width, x2)
    src_y2 = min(image_height, y2)
    patch = image[src_y1:src_y2, src_x1:src_x2]
    left = src_x1 - x1
    top = src_y1 - y1
    right = x2 - src_x2
    bottom = y2 - src_y2
    if any(value > 0 for value in (left, top, right, bottom)):
        patch = cv2.copyMakeBorder(patch, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return patch


def normalize_template_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.equalizeHist(gray)


def create_mouse_template(frame, contact_point, template_size, anchor_y):
    size = max(16, int(template_size))
    point = np.asarray(contact_point, dtype=np.float32)
    x1 = int(round(point[0] - size / 2))
    y1 = int(round(point[1] - size * anchor_y))
    patch = crop_with_padding(frame, x1, y1, size, size)
    return normalize_template_patch(patch)


def template_detection_from_match(top_left, size, anchor_y, score):
    x, y = top_left
    contour = np.array(
        [
            [[x, y]],
            [[x + size - 1, y]],
            [[x + size - 1, y + size - 1]],
            [[x, y + size - 1]],
        ],
        dtype=np.int32,
    )
    center = np.array([x + size / 2.0, y + size / 2.0], dtype=np.float32)
    contact = np.array([x + size / 2.0, y + size * anchor_y], dtype=np.float32)
    return {
        "center": center,
        "contact": contact,
        "contour": contour,
        "area": float(size * size),
        "angle": None,
        "mask": None,
        "roi": None,
        "score": float(score),
        "source": "template",
    }


def detect_template_candidate(
    frame,
    template,
    expected_point,
    previous_point,
    search_radius,
    min_score,
    roi_y_min_ratio,
    roi_y_max_ratio,
    anchor_y,
):
    if template is None:
        return None
    gray = normalize_template_patch(frame)
    template_height, template_width = template.shape[:2]
    image_height, image_width = gray.shape[:2]
    if template_width >= image_width or template_height >= image_height:
        return None

    target = expected_point if expected_point is not None else previous_point
    if target is None:
        return None
    target = np.asarray(target, dtype=np.float32)
    radius = int(max(search_radius, template_width))
    center_x = int(round(target[0]))
    center_y = int(round(target[1] - template_height * anchor_y + template_height / 2.0))

    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(image_width, center_x + radius)
    y2 = min(image_height, center_y + radius)

    roi_top = int(np.clip(roi_y_min_ratio, 0.0, 1.0) * image_height)
    roi_bottom = int(np.clip(roi_y_max_ratio, 0.0, 1.0) * image_height)
    if roi_bottom < roi_top:
        roi_top, roi_bottom = roi_bottom, roi_top
    y1 = max(y1, roi_top - int(template_height * anchor_y))
    y2 = min(y2, roi_bottom + int(template_height * (1.0 - anchor_y)))

    if x2 - x1 < template_width or y2 - y1 < template_height:
        return None

    search = gray[y1:y2, x1:x2]
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _min_value, max_value, _min_location, max_location = cv2.minMaxLoc(result)
    if max_value < min_score:
        return None
    top_left = (x1 + max_location[0], y1 + max_location[1])
    return template_detection_from_match(top_left, template_width, anchor_y, max_value)


def detect_mouse_blob(*args, **kwargs):
    candidates = detect_mouse_candidates(*args, **kwargs)
    return candidates[0] if candidates else None


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


def point_inside_arena(point, geometry, margin=0.02):
    if geometry is None:
        return True
    width_m = float(geometry.get("measurements", {}).get("arena_width_m", 0.29))
    length_m = float(geometry.get("measurements", {}).get("arena_length_m", 0.38))
    return (
        -width_m / 2 - margin <= point[0] <= width_m / 2 + margin
        and -length_m / 2 - margin <= point[1] <= length_m / 2 + margin
    )


def floor_intersection(camera_id, point, camera_models, floor_z):
    model = camera_models.get(camera_id)
    if model is None:
        return None
    center, ray_world = pixel_ray_world(model, point)
    if abs(ray_world[2]) < 1e-9:
        return None
    scale = (floor_z - center[2]) / ray_world[2]
    if scale <= 0:
        return None
    return center + scale * ray_world


def pixel_ray_world(model, point):
    pixel = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(
        pixel,
        model["camera_matrix"],
        model["dist_coeffs"],
    ).reshape(2)
    ray_camera = np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)
    ray_world = model["rotation"].T @ ray_camera
    center = model["center"]
    ray_world /= max(np.linalg.norm(ray_world), 1e-12)
    return center, ray_world


def raw_point_variants(display_point, raw_width, raw_height, rotation_degrees):
    base = unrotate_point(display_point, raw_width, raw_height, rotation_degrees)
    x, y = base
    return [
        ("normal", np.array([x, y], dtype=np.float32)),
        ("flip_x", np.array([raw_width - 1 - x, y], dtype=np.float32)),
        ("flip_y", np.array([x, raw_height - 1 - y], dtype=np.float32)),
        ("flip_xy", np.array([raw_width - 1 - x, raw_height - 1 - y], dtype=np.float32)),
    ]


def model_point_from_display(display_point, raw_width, raw_height, rotation_degrees, point_transform):
    variants = dict(raw_point_variants(display_point, raw_width, raw_height, rotation_degrees))
    return variants[point_transform]


def display_point_from_model(model_point, raw_width, raw_height, rotation_degrees, point_transform):
    x, y = np.asarray(model_point, dtype=np.float32)
    if point_transform == "normal":
        base = np.array([x, y], dtype=np.float32)
    elif point_transform == "flip_x":
        base = np.array([raw_width - 1 - x, y], dtype=np.float32)
    elif point_transform == "flip_y":
        base = np.array([x, raw_height - 1 - y], dtype=np.float32)
    elif point_transform == "flip_xy":
        base = np.array([raw_width - 1 - x, raw_height - 1 - y], dtype=np.float32)
    else:
        raise ValueError(f"Unknown point transform: {point_transform}")
    return rotate_point(base, raw_width, raw_height, rotation_degrees)


def project_world_to_display(
    camera_id,
    xyz,
    camera_models,
    raw_shape,
    rotation_degrees,
    point_transform,
    projection_offsets=None,
):
    if xyz is None or camera_id not in camera_models or raw_shape is None:
        return None
    model = camera_models[camera_id]
    rvec, _ = cv2.Rodrigues(model["rotation"])
    image_points, _ = cv2.projectPoints(
        np.asarray(xyz, dtype=np.float64).reshape(1, 1, 3),
        rvec,
        model["tvec"],
        model["camera_matrix"],
        model["dist_coeffs"],
    )
    raw_height, raw_width = raw_shape[:2]
    point = image_points.reshape(2)
    if not (-raw_width <= point[0] <= raw_width * 2 and -raw_height <= point[1] <= raw_height * 2):
        return None
    display_point = display_point_from_model(point, raw_width, raw_height, rotation_degrees, point_transform)
    if projection_offsets:
        display_point = display_point + projection_offsets.get(camera_id, 0.0)
    return display_point


def apply_spatial_prior(candidates, expected_point, radius_px, min_keep):
    if expected_point is None or not candidates:
        return candidates
    expected = np.asarray(expected_point, dtype=np.float32)
    ranked = []
    for candidate in candidates:
        distance = float(np.linalg.norm(candidate["contact"] - expected))
        ranked.append((distance, candidate))
    ranked.sort(key=lambda item: item[0])
    nearby = [candidate for distance, candidate in ranked if distance <= radius_px]
    if nearby:
        return sorted(
            nearby,
            key=lambda candidate: (
                -candidate["area"],
                float(np.linalg.norm(candidate["contact"] - expected)),
            ),
        )
    keep = max(0, min_keep)
    nearest = [candidate for _distance, candidate in ranked[:keep]]
    return sorted(nearest, key=lambda candidate: -candidate["area"])


def estimate_manual_floor_position(
    manual_points,
    camera_models,
    geometry,
    floor_z,
    raw_shapes,
    rotation_degrees,
    point_transform,
):
    if not manual_points:
        return None, {
            "observations": 0,
            "inside": 0,
            "outside": 0,
            "variant": "none",
            "fallback": False,
        }

    variant_points = {}
    variant_names = ("normal", "flip_x", "flip_y", "flip_xy") if point_transform == "auto" else (point_transform,)
    for camera_id, point in manual_points.items():
        shape = raw_shapes.get(camera_id)
        if shape is None or camera_id not in camera_models:
            continue
        raw_height, raw_width = shape[:2]
        variant_points[camera_id] = dict(raw_point_variants(point, raw_width, raw_height, rotation_degrees))

    if not variant_points:
        return None, {
            "observations": len(manual_points),
            "inside": 0,
            "outside": 0,
            "variant": "none",
            "fallback": False,
        }

    best = None
    for variant_name in variant_names:
        points = []
        outside = 0
        for camera_id, variants in variant_points.items():
            floor_point = floor_intersection(camera_id, variants[variant_name], camera_models, floor_z)
            if floor_point is None:
                continue
            if point_inside_arena(floor_point, geometry):
                points.append(floor_point)
            else:
                outside += 1
        if not points:
            continue
        xyz = np.median(np.asarray(points), axis=0)
        spread = float(np.mean([np.linalg.norm(point[:2] - xyz[:2]) for point in points]))
        score = (len(points), -spread, -outside)
        if best is None or score > best["score"]:
            best = {
                "xyz": xyz,
                "score": score,
                "inside": len(points),
                "outside": outside,
                "variant": variant_name,
                "spread": spread,
            }

    if best is None:
        return None, {
            "observations": len(variant_points),
            "inside": 0,
            "outside": 0,
            "variant": "none",
            "fallback": False,
        }

    return best["xyz"], {
        "observations": len(variant_points),
        "inside": best["inside"],
        "outside": best["outside"],
        "variant": best["variant"],
        "spread": best["spread"],
        "fallback": False,
    }


def estimate_floor_position(observations, camera_models, geometry, floor_z, allow_outside_fallback=False):
    points = []
    rejected_points = []
    for camera_id, point in observations.items():
        floor_point = floor_intersection(camera_id, point, camera_models, floor_z)
        if floor_point is None:
            continue
        if point_inside_arena(floor_point, geometry):
            points.append(floor_point)
        else:
            rejected_points.append(floor_point)

    if points:
        return np.median(np.asarray(points), axis=0), {
            "observations": len(observations),
            "inside": len(points),
            "outside": len(rejected_points),
            "fallback": False,
        }
    if rejected_points and allow_outside_fallback:
        return np.median(np.asarray(rejected_points), axis=0), {
            "observations": len(observations),
            "inside": 0,
            "outside": len(rejected_points),
            "fallback": True,
        }
    return None, {
        "observations": len(observations),
        "inside": 0,
        "outside": len(rejected_points),
        "fallback": False,
    }


def estimate_manual_click_fallback(display_points, geometry):
    if not display_points or geometry is None:
        return None, {"observations": len(display_points), "fallback": False}

    width_m = float(geometry.get("measurements", {}).get("arena_width_m", 0.29))
    length_m = float(geometry.get("measurements", {}).get("arena_length_m", 0.38))
    estimates = []
    for camera_id, point in display_points.items():
        position_name = None
        for name, spec in geometry.get("cameras", {}).items():
            if int(spec["camera_index"]) == camera_id:
                position_name = name
                break
        if position_name is None:
            continue

        px, py = point
        # The fallback is intentionally rough: use the clicked horizontal image
        # coordinate as progress across the enclosure, and the known camera row
        # as the along-length estimate. It is only a visible/manual fallback.
        if position_name.startswith("left"):
            x = -width_m / 2 + np.clip(px, 0.0, 1.0) * width_m
        elif position_name.startswith("right"):
            x = width_m / 2 - np.clip(px, 0.0, 1.0) * width_m
        else:
            x = 0.0

        if position_name.endswith("_1"):
            y = -length_m / 2 + 0.25 * length_m
        elif position_name.endswith("_2"):
            y = -length_m / 2 + 0.75 * length_m
        else:
            y = -length_m / 2 + np.clip(py, 0.0, 1.0) * length_m
        estimates.append([x, y, 0.0])

    if not estimates:
        return None, {"observations": len(display_points), "fallback": False}
    return np.median(np.asarray(estimates, dtype=np.float64), axis=0), {
        "observations": len(estimates),
        "fallback": True,
    }


def choose_floor_consistent_detections(
    candidate_sets,
    camera_models,
    geometry,
    floor_z,
    raw_shapes,
    rotation_degrees,
    point_transform,
    cluster_radius,
    candidate_margin,
):
    floor_candidates = []
    for camera_id, candidates in candidate_sets.items():
        raw_height, raw_width = raw_shapes[camera_id][:2]
        for detection in candidates:
            raw_contact = model_point_from_display(
                detection["contact"],
                raw_width,
                raw_height,
                rotation_degrees,
                point_transform,
            )
            floor_point = floor_intersection(camera_id, raw_contact, camera_models, floor_z)
            if floor_point is None or not point_inside_arena(floor_point, geometry, margin=candidate_margin):
                continue
            floor_candidates.append(
                {
                    "camera_id": camera_id,
                    "detection": detection,
                    "floor_point": floor_point,
                }
            )

    if not floor_candidates:
        return {}, None, {
            "observations": sum(bool(candidates) for candidates in candidate_sets.values()),
            "inside": 0,
            "outside": 0,
            "support": 0,
            "fallback": False,
        }

    best_group = []
    best_score = None
    for seed in floor_candidates:
        grouped_by_camera = {}
        for candidate in floor_candidates:
            distance = np.linalg.norm(candidate["floor_point"][:2] - seed["floor_point"][:2])
            if distance > cluster_radius:
                continue
            camera_id = candidate["camera_id"]
            previous = grouped_by_camera.get(camera_id)
            if previous is None or distance < previous["distance"]:
                grouped_by_camera[camera_id] = {**candidate, "distance": distance}
        group = list(grouped_by_camera.values())
        if not group:
            continue
        support = len(group)
        spread = float(np.mean([item["distance"] for item in group]))
        area_score = float(np.mean([item["detection"]["area"] for item in group]))
        score = (support, -spread, area_score)
        if best_score is None or score > best_score:
            best_score = score
            best_group = group

    selected_detections = {item["camera_id"]: item["detection"] for item in best_group}
    xyz = np.median(np.asarray([item["floor_point"] for item in best_group]), axis=0)
    xyz_spread = float(np.mean([np.linalg.norm(item["floor_point"][:2] - xyz[:2]) for item in best_group]))
    return selected_detections, xyz, {
        "observations": sum(bool(candidates) for candidates in candidate_sets.values()),
        "inside": len(floor_candidates),
        "outside": 0,
        "support": len(best_group),
        "spread": xyz_spread,
        "fallback": False,
    }


def annotate_frame(
    frame,
    camera_id,
    camera_label,
    detection,
    candidates,
    manual_point,
    expected_point,
    expected_radius,
    xyz,
    roi=None,
    roi_line_thickness=3,
    clamped=False,
    position_status=None,
    hide_unselected_candidates=False,
):
    annotated = frame.copy()
    if roi is not None:
        roi_y1, roi_y2 = roi
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1] - 1, roi_y1), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
        cv2.line(
            annotated,
            (0, roi_y1),
            (annotated.shape[1] - 1, roi_y1),
            (0, 255, 255),
            roi_line_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "ROI",
            (12, max(18, roi_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if candidates and not hide_unselected_candidates:
        for candidate in candidates:
            if detection is not None and candidate is detection:
                continue
            cv2.drawContours(annotated, [candidate["contour"]], -1, (0, 180, 0), 1)

    if detection is not None:
        center = tuple(np.round(detection["center"]).astype(int))
        contact = tuple(np.round(detection["contact"]).astype(int))
        cv2.drawContours(annotated, [detection["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(annotated, center, 6, (0, 0, 255), -1)
        cv2.circle(annotated, contact, 5, (255, 0, 0), -1)
        cv2.putText(
            annotated,
            (
                f"tmpl {detection.get('score', 0.0):.2f}"
                if detection.get("source") == "template"
                else f"{detection.get('source', 'blob')} area {detection['area']:.0f}"
            ),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if manual_point is not None:
        manual_center = tuple(np.round(manual_point).astype(int))
        cv2.drawMarker(
            annotated,
            manual_center,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(annotated, manual_center, 10, (255, 0, 255), 2, cv2.LINE_AA)
    if expected_point is not None:
        expected_center = tuple(np.round(expected_point).astype(int))
        cv2.circle(annotated, expected_center, int(max(4, expected_radius)), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(
            annotated,
            expected_center,
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    xyz_text = "XYZ: waiting"
    if xyz is not None:
        suffix = " clamped" if clamped else ""
        xyz_text = f"XYZ m: {xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}{suffix}"
    elif position_status:
        xyz_text = f"XYZ waiting | {position_status}"
    cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 560), 48), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"{camera_label} (cam {camera_id}) | {xyz_text}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def camera_position_labels(geometry):
    labels = {}
    if geometry is None:
        return labels
    for position_name, spec in geometry.get("cameras", {}).items():
        labels[int(spec["camera_index"])] = position_name
    return labels


def load_geometry(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_projection_offsets(path):
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    offsets = {}
    for camera_id, correction in data.get("corrections_px", {}).items():
        offsets[int(camera_id)] = np.array(
            [float(correction.get("dx", 0.0)), float(correction.get("dy", 0.0))],
            dtype=np.float32,
        )
    if offsets:
        print(f"Loaded projection offsets from {path}: {sorted(offsets)}", flush=True)
    return offsets


def clamp_xyz_to_arena(xyz, geometry):
    if xyz is None or geometry is None:
        return xyz, False
    width_m = float(geometry.get("measurements", {}).get("arena_width_m", 0.29))
    length_m = float(geometry.get("measurements", {}).get("arena_length_m", 0.38))
    clamped = np.array(xyz, dtype=np.float64).copy()
    before = clamped.copy()
    clamped[0] = np.clip(clamped[0], -width_m / 2, width_m / 2)
    clamped[1] = np.clip(clamped[1], -length_m / 2, length_m / 2)
    return clamped, not np.allclose(before[:2], clamped[:2])


def make_top_down_panel(xyz, poses_path, geometry, size, position_method, position_status=None):
    panel = np.full((size, size, 3), 245, dtype=np.uint8)
    margin = 34
    if geometry is None:
        cv2.putText(panel, "No geometry file", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        return panel

    width_m = float(geometry.get("measurements", {}).get("arena_width_m", 0.29))
    length_m = float(geometry.get("measurements", {}).get("arena_length_m", 0.38))
    scale = min((size - 2 * margin) / width_m, (size - 2 * margin) / length_m)
    cx = size // 2
    cy = size // 2

    def world_to_panel(x, y):
        px = int(round(cx + x * scale))
        py = int(round(cy - y * scale))
        return px, py

    left, top = world_to_panel(-width_m / 2, length_m / 2)
    right, bottom = world_to_panel(width_m / 2, -length_m / 2)
    cv2.rectangle(panel, (left, top), (right, bottom), (30, 30, 30), 2)
    cv2.line(panel, world_to_panel(0, -length_m / 2), world_to_panel(0, length_m / 2), (210, 210, 210), 1)
    cv2.line(panel, world_to_panel(-width_m / 2, 0), world_to_panel(width_m / 2, 0), (210, 210, 210), 1)

    for position_name, spec in geometry.get("cameras", {}).items():
        x, y, _z = spec["center_m"]
        lx, ly, _lz = spec["look_at_m"]
        p = world_to_panel(x, y)
        q = world_to_panel(lx, ly)
        cv2.circle(panel, p, 5, (70, 70, 70), -1)
        cv2.line(panel, p, q, (120, 120, 120), 1)
        cv2.putText(panel, position_name, (p[0] + 7, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)

    display_xyz, was_clamped = clamp_xyz_to_arena(xyz, geometry)
    if display_xyz is not None:
        x, y, z = display_xyz
        mouse = world_to_panel(float(x), float(y))
        cv2.circle(panel, mouse, 8, (0, 0, 255), -1)
        suffix = " clamped" if was_clamped else ""
        cv2.putText(
            panel,
            f"x {x:+.3f} y {y:+.3f} z {z:+.3f} m{suffix}",
            (12, size - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    else:
        waiting_text = "XYZ waiting"
        if position_status:
            waiting_text += f" | {position_status}"
        cv2.putText(panel, waiting_text, (12, size - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

    cv2.putText(panel, "Top-down", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, poses_path.name, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(panel, position_method, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    return panel


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


def make_grid_with_layout(items, cols, display_height):
    tiles = []
    for item in items:
        frame = item["frame"]
        resized = resize_to_height(frame, display_height)
        tiles.append(
            {
                **item,
                "tile": resized,
                "source_width": frame.shape[1],
                "source_height": frame.shape[0],
            }
        )

    tile_height = max(item["tile"].shape[0] for item in tiles)
    tile_width = max(item["tile"].shape[1] for item in tiles)
    rows = []
    layout = []
    for start in range(0, len(tiles), cols):
        row_tiles = []
        row_index = start // cols
        for col_index, item in enumerate(tiles[start : start + cols]):
            fitted = fit_to_tile(item["tile"], tile_width, tile_height)
            row_tiles.append(fitted)

            resized_height, resized_width = item["tile"].shape[:2]
            scale = min(tile_width / resized_width, tile_height / resized_height)
            fitted_width = max(1, int(resized_width * scale))
            fitted_height = max(1, int(resized_height * scale))
            pad_x = (tile_width - fitted_width) // 2
            pad_y = (tile_height - fitted_height) // 2
            global_x = col_index * tile_width
            global_y = row_index * tile_height
            layout.append(
                {
                    "camera_id": item["camera_id"],
                    "rect": (
                        global_x + pad_x,
                        global_y + pad_y,
                        global_x + pad_x + fitted_width,
                        global_y + pad_y + fitted_height,
                    ),
                    "source_width": item["source_width"],
                    "source_height": item["source_height"],
                }
            )
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows), layout


def append_side_panel(grid, panel):
    panel = fit_to_tile(panel, panel.shape[1], grid.shape[0])
    return np.hstack([grid, panel])


def open_datalog(path, camera_ids):
    if path is None:
        return None, None
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8")
    fieldnames = [
        "timestamp",
        "event",
        "clicked_cameras",
        "xyz_x_m",
        "xyz_y_m",
        "xyz_z_m",
        "clamped",
        "position_status",
        "manual_variant",
        "manual_spread_m",
        "manual_inside",
        "manual_outside",
    ]
    for camera_id in camera_ids:
        fieldnames.extend(
            [
                f"cam_{camera_id}_display_x",
                f"cam_{camera_id}_display_y",
                f"cam_{camera_id}_norm_x",
                f"cam_{camera_id}_norm_y",
            ]
        )
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        handle.flush()
    print(f"Writing fusion datalog: {path}", flush=True)
    return handle, writer


def write_manual_datalog_row(
    writer,
    event_id,
    manual_points,
    display_shapes,
    xyz,
    clamped,
    position_status,
    floor_status,
    camera_ids,
):
    if writer is None:
        return
    row = {
        "timestamp": f"{time.time():.6f}",
        "event": event_id,
        "clicked_cameras": " ".join(str(camera_id) for camera_id in sorted(manual_points)),
        "xyz_x_m": "" if xyz is None else f"{xyz[0]:.6f}",
        "xyz_y_m": "" if xyz is None else f"{xyz[1]:.6f}",
        "xyz_z_m": "" if xyz is None else f"{xyz[2]:.6f}",
        "clamped": int(bool(clamped)),
        "position_status": position_status or "",
        "manual_variant": "" if floor_status is None else floor_status.get("variant", ""),
        "manual_spread_m": "" if floor_status is None else f"{floor_status.get('spread', float('nan')):.6f}",
        "manual_inside": "" if floor_status is None else floor_status.get("inside", ""),
        "manual_outside": "" if floor_status is None else floor_status.get("outside", ""),
    }
    for camera_id in camera_ids:
        point = manual_points.get(camera_id)
        shape = display_shapes.get(camera_id)
        if point is None or shape is None:
            row[f"cam_{camera_id}_display_x"] = ""
            row[f"cam_{camera_id}_display_y"] = ""
            row[f"cam_{camera_id}_norm_x"] = ""
            row[f"cam_{camera_id}_norm_y"] = ""
            continue
        height, width = shape[:2]
        row[f"cam_{camera_id}_display_x"] = f"{point[0]:.2f}"
        row[f"cam_{camera_id}_display_y"] = f"{point[1]:.2f}"
        row[f"cam_{camera_id}_norm_x"] = f"{point[0] / max(width - 1, 1):.6f}"
        row[f"cam_{camera_id}_norm_y"] = f"{point[1] / max(height - 1, 1):.6f}"
    writer.writerow(row)


def main():
    args = parse_args()
    geometry = load_geometry(args.geometry) if args.top_down else None
    position_labels = camera_position_labels(geometry)
    background_thresholds = parse_camera_value_map(args.background_thresholds, int)
    projections = None
    camera_models = None
    if not args.preview_2d:
        if args.position_method == "triangulate":
            projections = load_projections(args.poses, args.cameras)
        else:
            camera_models = load_camera_models(args.poses, args.cameras)
    backgrounds = load_backgrounds(args.background_dir, args.cameras, args.rotation)
    projection_offsets = load_projection_offsets(args.projection_offsets)
    cameras = []
    manual_points = {}
    mouse_templates = {}
    template_last_points = {}
    latest_display_frames = {}
    manual_revision = 0
    logged_manual_revision = 0
    manual_seed_pending = False
    grid_layout = []
    spatial_prior_xyz = None
    filtered_xyz = None
    datalog_handle, datalog_writer = open_datalog(args.datalog, args.cameras)

    def handle_mouse(event, x, y, flags, userdata):
        nonlocal manual_revision, manual_seed_pending
        if not args.manual_clicks or event != cv2.EVENT_LBUTTONDOWN:
            return
        for item in grid_layout:
            x1, y1, x2, y2 = item["rect"]
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            camera_id = item["camera_id"]
            px = (x - x1) / (x2 - x1)
            py = (y - y1) / (y2 - y1)
            manual_points[camera_id] = np.array(
                [px * item["source_width"], py * item["source_height"]],
                dtype=np.float32,
            )
            if args.template_tracking and camera_id in latest_display_frames:
                mouse_templates[camera_id] = create_mouse_template(
                    latest_display_frames[camera_id],
                    manual_points[camera_id],
                    args.template_size,
                    args.template_anchor_y,
                )
                template_last_points[camera_id] = manual_points[camera_id].copy()
            manual_revision += 1
            manual_seed_pending = True
            print(
                f"Manual mouse point cam {camera_id}: "
                f"{manual_points[camera_id][0]:.1f}, {manual_points[camera_id][1]:.1f}"
                f"{' template saved' if camera_id in mouse_templates else ''}",
                flush=True,
            )
            break

    try:
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}")

        cv2.namedWindow("Mouse Fusion", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mouse Fusion", handle_mouse)
        while True:
            observations = {}
            manual_observations = {}
            detections = {}
            candidate_sets = {}
            raw_shapes = {}
            display_shapes = {}
            expected_points = {}
            position_status = None
            floor_status = None
            for camera in cameras:
                if camera.latest_frame is None:
                    continue
                raw_shapes[camera.device_index] = camera.latest_frame.shape
                display_frame = rotate_frame(camera.latest_frame, args.rotation)
                latest_display_frames[camera.device_index] = display_frame
                display_shapes[camera.device_index] = display_frame.shape
                point_transform = args.point_transform if args.point_transform != "auto" else "flip_xy"
                expected_point = None
                if args.spatial_prior and spatial_prior_xyz is not None and camera_models is not None:
                    expected_point = project_world_to_display(
                        camera.device_index,
                        spatial_prior_xyz,
                        camera_models,
                        camera.latest_frame.shape,
                        args.rotation,
                        point_transform,
                        projection_offsets,
                    )
                expected_points[camera.device_index] = expected_point
                candidates = detect_mouse_candidates(
                    display_frame,
                    args.threshold,
                    args.invert,
                    args.min_area,
                    args.max_area_ratio,
                    args.border_margin,
                    args.roi_y_min_ratio,
                    args.roi_y_max_ratio,
                    args.roi_top_reject_ratio,
                    args.candidate_count,
                    args.morph_open,
                    args.morph_close,
                    args.morph_dilate,
                    backgrounds.get(camera.device_index),
                    background_thresholds.get(camera.device_index, args.background_threshold),
                )
                local_candidates = detect_prior_region_candidates(
                    display_frame,
                    backgrounds.get(camera.device_index),
                    expected_point,
                    args.spatial_prior_radius,
                    args.min_area,
                    args.max_area_ratio,
                    args.roi_y_min_ratio,
                    args.roi_y_max_ratio,
                    args.prior_local_threshold,
                    args.prior_local_close,
                    args.prior_local_dilate,
                )
                if local_candidates:
                    candidates = local_candidates + candidates
                template_candidate = None
                if args.template_tracking:
                    template_candidate = detect_template_candidate(
                        display_frame,
                        mouse_templates.get(camera.device_index),
                        expected_point,
                        template_last_points.get(camera.device_index),
                        args.template_search_radius,
                        args.template_min_score,
                        args.roi_y_min_ratio,
                        args.roi_y_max_ratio,
                        args.template_anchor_y,
                    )
                    if template_candidate is not None:
                        candidates = [template_candidate] + candidates
                candidates = apply_spatial_prior(
                    candidates,
                    expected_point,
                    args.spatial_prior_radius,
                    args.spatial_prior_min_keep,
                )
                candidate_sets[camera.device_index] = candidates
                detection = candidates[0] if candidates else None
                detections[camera.device_index] = detection
                if detection is not None:
                    observations[camera.device_index] = model_point_from_display(
                        detection["contact"],
                        camera.latest_frame.shape[1],
                        camera.latest_frame.shape[0],
                        args.rotation,
                        args.point_transform if args.point_transform != "auto" else "flip_xy",
                    )
                if camera.device_index in manual_points:
                    manual_observations[camera.device_index] = model_point_from_display(
                        manual_points[camera.device_index],
                        camera.latest_frame.shape[1],
                        camera.latest_frame.shape[0],
                        args.rotation,
                        args.point_transform if args.point_transform != "auto" else "flip_xy",
                    )

            xyz = None
            use_manual_seed = False
            if not args.preview_2d:
                use_manual_seed = manual_observations and args.position_method == "floor" and (
                    args.manual_lock or manual_seed_pending
                )
                if use_manual_seed:
                    xyz, floor_status = estimate_manual_floor_position(
                        manual_points,
                        camera_models,
                        geometry,
                        args.floor_z,
                        raw_shapes,
                        args.rotation,
                        args.point_transform,
                    )
                    position_status = (
                        f"{floor_status['inside']}/{floor_status['observations']} manual floor hits"
                        f" ({floor_status['outside']} outside, {floor_status['variant']})"
                    )
                    if args.manual_click_fallback and (xyz is None or floor_status["inside"] == 0):
                        manual_display_points = {}
                        for camera_id, point in manual_points.items():
                            shape = display_shapes.get(camera_id)
                            if shape is None:
                                continue
                            height, width = shape[:2]
                            manual_display_points[camera_id] = np.array(
                                [
                                    point[0] / max(width - 1, 1),
                                    point[1] / max(height - 1, 1),
                                ],
                                dtype=np.float32,
                            )
                        fallback_xyz, fallback_status = estimate_manual_click_fallback(
                            manual_display_points,
                            geometry,
                        )
                        if fallback_xyz is not None:
                            xyz = fallback_xyz
                            floor_status = {
                                **fallback_status,
                                "inside": "",
                                "outside": "",
                                "variant": "fallback",
                                "spread": float("nan"),
                            }
                            position_status = (
                                f"{fallback_status['observations']} manual click fallback "
                                f"(click floor contact)"
                            )
                elif args.position_method == "triangulate":
                    xyz = triangulate(observations, projections)
                    position_status = f"{len(observations)} obs"
                else:
                    detections, xyz, floor_status = choose_floor_consistent_detections(
                        candidate_sets,
                        camera_models,
                        geometry,
                        args.floor_z,
                        raw_shapes,
                        args.rotation,
                        args.point_transform if args.point_transform != "auto" else "flip_xy",
                        args.floor_cluster_radius,
                        args.floor_candidate_margin,
                    )
                    position_status = (
                        f"{floor_status['support']} cam support | "
                        f"{floor_status['inside']} floor candidates | "
                        f"{floor_status['observations']} cams with blobs | "
                        f"spread {floor_status.get('spread', 0.0):.3f} m"
                    )
                    if xyz is None and args.allow_outside_fallback:
                        xyz, floor_status = estimate_floor_position(
                            observations,
                            camera_models,
                            geometry,
                            args.floor_z,
                            args.allow_outside_fallback,
                        )
                        fallback = " fallback" if floor_status["fallback"] else ""
                        position_status = (
                            f"{floor_status['inside']}/{floor_status['observations']} floor hits"
                            f" ({floor_status['outside']} outside){fallback}"
                        )
            accepted_xyz = xyz
            if xyz is not None and not use_manual_seed and args.position_method == "floor":
                support = 0 if floor_status is None else int(floor_status.get("support", 0) or 0)
                spread = float("inf") if floor_status is None else float(floor_status.get("spread", 0.0) or 0.0)
                if support < args.tracker_min_support:
                    accepted_xyz = None
                    position_status = f"{position_status} | hold: support {support}<{args.tracker_min_support}"
                elif spread > args.tracker_max_spread:
                    accepted_xyz = None
                    position_status = f"{position_status} | hold: spread {spread:.3f}>{args.tracker_max_spread:.3f}"
                elif filtered_xyz is not None:
                    step = float(np.linalg.norm(np.asarray(xyz)[:2] - filtered_xyz[:2]))
                    if step > args.tracker_max_step:
                        accepted_xyz = None
                        position_status = f"{position_status} | hold: jump {step:.3f}>{args.tracker_max_step:.3f}"

            if accepted_xyz is not None:
                accepted_xyz = np.asarray(accepted_xyz, dtype=np.float64)
                if use_manual_seed or filtered_xyz is None:
                    filtered_xyz = accepted_xyz
                else:
                    alpha = float(np.clip(args.tracker_alpha, 0.0, 1.0))
                    filtered_xyz = (1.0 - alpha) * filtered_xyz + alpha * accepted_xyz
                if not use_manual_seed:
                    for camera_id, detection in detections.items():
                        if detection is not None and detection.get("source") == "template":
                            template_last_points[camera_id] = detection["contact"].copy()

            display_xyz, xyz_clamped = clamp_xyz_to_arena(filtered_xyz, geometry)
            if display_xyz is not None:
                spatial_prior_xyz = display_xyz
                if manual_seed_pending and not args.manual_lock:
                    manual_seed_pending = False
            if manual_points and manual_revision != logged_manual_revision:
                write_manual_datalog_row(
                    datalog_writer,
                    manual_revision,
                    manual_points,
                    display_shapes,
                    display_xyz,
                    xyz_clamped,
                    position_status,
                    floor_status,
                    args.cameras,
                )
                if datalog_handle is not None:
                    datalog_handle.flush()
                logged_manual_revision = manual_revision
            annotated_items = []
            for camera in cameras:
                if camera.latest_frame is None:
                    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    annotated_items.append({"camera_id": camera.device_index, "frame": blank})
                else:
                    frame = camera.latest_frame
                    display_frame = rotate_frame(frame, args.rotation)
                    annotated = annotate_frame(
                        display_frame,
                        camera.device_index,
                        position_labels.get(camera.device_index, f"Camera {camera.device_index}"),
                        detections.get(camera.device_index),
                        candidate_sets.get(camera.device_index, []),
                        manual_points.get(camera.device_index),
                        expected_points.get(camera.device_index),
                        args.spatial_prior_radius,
                        display_xyz,
                        (
                            int(np.clip(args.roi_y_min_ratio, 0.0, 1.0) * display_frame.shape[0]),
                            int(np.clip(args.roi_y_max_ratio, 0.0, 1.0) * display_frame.shape[0]),
                        ),
                        args.roi_line_thickness,
                        xyz_clamped,
                        position_status,
                        args.hide_unselected_candidates,
                    )
                    annotated_items.append({"camera_id": camera.device_index, "frame": annotated})

            grid, grid_layout = make_grid_with_layout(annotated_items, args.cols, args.display_height)
            if args.top_down:
                top_down = make_top_down_panel(
                    xyz,
                    args.poses,
                    geometry,
                    args.top_down_size,
                    args.position_method,
                    position_status,
                )
                grid = append_side_panel(grid, top_down)
            cv2.imshow("Mouse Fusion", grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                manual_points.clear()
                mouse_templates.clear()
                template_last_points.clear()
                manual_revision += 1
                manual_seed_pending = False
                spatial_prior_xyz = None
                filtered_xyz = None
                print("Cleared manual mouse points", flush=True)
    finally:
        for camera in cameras:
            camera.stop()
        if datalog_handle is not None:
            datalog_handle.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
