import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

import cv2
import numpy as np

from dshow_arducam_viewer import DShowCamera, fit_to_tile, get_formats, resize_to_height
from live_gods_eye_fusion import (
    POINT_TRANSFORMS,
    apply_single_pitch_offset,
    display_points_from_model,
    load_camera_models,
    load_camera_tuning,
    rotate_frame,
)


WINDOW_NAME = "Side View Fusion"
SIDE_NAMES = ("left", "right")
ARUCO_DICT = "DICT_6X6_250"
CONFIG_PATH = Path("dual_arducam_viewer_config.json")
POSITION_KEYS = ("left_1", "left_2", "right_1", "right_2")
POSITION_SHORT = {
    "left_1": "L1",
    "left_2": "L2",
    "right_1": "R1",
    "right_2": "R2",
}
ASSIGN_KEYS = {
    ord("1"): "left_1",
    ord("2"): "left_2",
    ord("3"): "right_1",
    ord("4"): "right_2",
}


def log(message):
    try:
        print(message, flush=True)
    except OSError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Experimental left/right side-view fusion from paired ArduCams.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--allow-format-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the best available same-format mode if a camera lacks the requested resolution.",
    )
    parser.add_argument("--scan", action="store_true", help="List DirectShow camera devices and exit.")
    parser.add_argument("--list-formats", action="store_true", help="List advertised DirectShow formats for --cameras and exit.")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--poses", type=Path, default=Path("camera_poses_manual.npz"))
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--camera-tuning", type=Path, default=Path("gods_eye_camera_tuning.json"))
    parser.add_argument("--no-camera-tuning", action="store_true")
    parser.add_argument("--point-transform", choices=POINT_TRANSFORMS, default="flip_xy")
    parser.add_argument("--output-width", type=int, default=760)
    parser.add_argument("--output-height", type=int, default=320)
    parser.add_argument("--max-z-mm", type=int, default=160)
    parser.add_argument(
        "--plane-y-mm",
        type=int,
        default=0,
        help="Virtual side curtain offset across the enclosure width. 0 is the centerline.",
    )
    parser.add_argument("--blend-sigma-mm", type=int, default=95)
    parser.add_argument("--min-depth-mm", type=int, default=10)
    parser.add_argument("--max-view-angle-deg", type=int, default=82)
    parser.add_argument("--show-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-height", type=int, default=210)
    parser.add_argument("--blend", choices=("average", "strongest"), default="average")
    parser.add_argument("--stitch-mask", choices=("full", "floor"), default="full")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. 0 runs until q/Esc.")
    parser.add_argument("--startup-timeout", type=float, default=3.0, help="Seconds to wait for initial camera frames.")
    parser.add_argument(
        "--stitch-calibration-frames",
        type=int,
        default=10,
        help="ArUco mode: estimate each side-pair stitch from this many good frames, then freeze it.",
    )
    parser.add_argument(
        "--live-stitch",
        action="store_true",
        help="Keep updating the pair stitch every frame. Useful for debugging, but usually visibly wobbly.",
    )
    parser.add_argument(
        "--mode",
        choices=("aruco", "plane"),
        default="aruco",
        help="aruco stitches each side pair in image space; plane projects both cameras onto a virtual side curtain.",
    )
    return parser.parse_args()


def list_devices():
    from dshow_arducam_viewer import list_devices as _list_devices

    return _list_devices()


def print_formats(camera_ids):
    devices = list_devices()
    for camera_id in camera_ids:
        name = devices[camera_id] if 0 <= camera_id < len(devices) else "(missing device)"
        log(f"\nDevice {camera_id}: {name}")
        try:
            formats = get_formats(camera_id)
        except Exception as exc:
            log(f"  Could not list formats: {exc}")
            continue
        for fmt in sorted(formats, key=lambda item: (item["media_type_str"], item["width"], item["height"])):
            log(
                f"  {fmt['index']:3d} {fmt['media_type_str']:4s} "
                f"{fmt['width']}x{fmt['height']} "
                f"fps={fmt.get('max_framerate', 0):.1f}-{fmt.get('min_framerate', 0):.1f}"
            )


def find_format(device_index, subtype, width, height, allow_fallback=True):
    subtype = subtype.upper()
    formats = get_formats(device_index)
    exact = [
        fmt
        for fmt in formats
        if fmt["media_type_str"].upper() == subtype
        and fmt["width"] == width
        and fmt["height"] == height
    ]
    if exact:
        return exact[0], False

    if not allow_fallback:
        raise RuntimeError(f"Camera {device_index} has no {subtype} {width}x{height} DirectShow format.")

    same_format = [fmt for fmt in formats if fmt["media_type_str"].upper() == subtype]
    if not same_format:
        raise RuntimeError(f"Camera {device_index} has no {subtype} DirectShow format.")

    under_request = [fmt for fmt in same_format if fmt["width"] <= width and fmt["height"] <= height]
    candidates = under_request or same_format
    selected = max(candidates, key=lambda fmt: (fmt["width"] * fmt["height"], fmt["width"], fmt["height"]))
    return selected, True


def load_geometry(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_position_assignments(config_path, geometry):
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        positions = data.get("positions", {})
        if positions:
            return {position: int(camera_id) for position, camera_id in positions.items() if position in POSITION_KEYS}
    return {
        position_name: int(spec["camera_index"])
        for position_name, spec in geometry.get("cameras", {}).items()
        if position_name in POSITION_KEYS
    }


def save_position_assignments(config_path, args, position_to_camera):
    payload = {
        "cameras": sorted(set(int(camera_id) for camera_id in position_to_camera.values())),
        "positions": {position: int(position_to_camera[position]) for position in POSITION_KEYS if position in position_to_camera},
        "width": args.width,
        "height": args.height,
        "format": args.format,
        "rotation": args.rotation,
        "cols": 2,
        "display_height": 400,
        "allow_format_fallback": args.allow_format_fallback,
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"Saved camera assignments -> {config_path}")


def camera_position_labels(position_to_camera):
    return {
        int(camera_id): position_name
        for position_name, camera_id in position_to_camera.items()
    }


def side_camera_ids(position_to_camera):
    return {
        "left": [position_to_camera[name] for name in ("left_1", "left_2") if name in position_to_camera],
        "right": [position_to_camera[name] for name in ("right_1", "right_2") if name in position_to_camera],
    }


def make_side_world_grid(geometry, output_width, output_height, plane_y_m, max_z_m):
    length = float(geometry["measurements"]["arena_length_m"])
    # New convention for this view: horizontal is enclosure X/long axis.
    long_x = np.linspace(-length / 2.0, length / 2.0, output_width, dtype=np.float32)
    z = np.linspace(max_z_m, 0.0, output_height, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(long_x, z)

    # Camera pose files still use the old convention:
    # old x = short axis, old y = long axis, z = up.
    world_old = np.column_stack(
        [
            np.full(grid_x.size, plane_y_m, dtype=np.float32),
            grid_x.reshape(-1),
            grid_z.reshape(-1),
        ]
    )
    return world_old, grid_x, grid_z


def precompute_side_maps(
    camera_models,
    camera_ids,
    geometry,
    output_width,
    output_height,
    plane_y_m,
    max_z_m,
    source_shape,
    rotation_degrees,
    point_transform,
    tuning,
    min_depth_m,
    max_view_angle_deg,
    blend_sigma_m,
):
    raw_height, raw_width = source_shape[:2]
    world_points, grid_x, _grid_z = make_side_world_grid(
        geometry,
        output_width,
        output_height,
        plane_y_m,
        max_z_m,
    )
    maps = {}
    sigma = max(blend_sigma_m, 1e-6)
    for camera_id in camera_ids:
        model = camera_models[camera_id]
        camera_tuning = tuning.get(camera_id, {})
        adjusted_model = apply_single_pitch_offset(model, float(camera_tuning.get("pitch_deg", 0.0)))
        camera_transform = camera_tuning.get("point_transform", point_transform)
        image_points, _ = cv2.projectPoints(
            world_points.reshape(-1, 1, 3).astype(np.float64),
            adjusted_model["rvec"],
            adjusted_model["tvec"],
            adjusted_model["camera_matrix"],
            adjusted_model["dist_coeffs"],
        )
        display_points = display_points_from_model(
            image_points.reshape(-1, 2),
            raw_width,
            raw_height,
            rotation_degrees,
            camera_transform,
        )
        camera_points = (adjusted_model["rotation"] @ world_points.T + adjusted_model["tvec"]).T
        camera_depth = camera_points[:, 2]
        lateral = np.linalg.norm(camera_points[:, :2], axis=1)
        max_lateral = np.tan(np.deg2rad(max_view_angle_deg)) * np.maximum(camera_depth, 1e-9)

        map_x = display_points[:, 0].reshape(output_height, output_width).astype(np.float32)
        map_y = display_points[:, 1].reshape(output_height, output_width).astype(np.float32)
        in_bounds = (
            (map_x >= 0)
            & (map_x <= raw_width - 1)
            & (map_y >= 0)
            & (map_y <= raw_height - 1)
            & (camera_depth.reshape(output_height, output_width) > min_depth_m)
            & (lateral.reshape(output_height, output_width) <= max_lateral.reshape(output_height, output_width))
        )

        camera_long_x = float(camera_models[camera_id]["center"][1])
        ownership_weight = np.exp(-0.5 * np.square((grid_x - camera_long_x) / sigma)).astype(np.float32)
        maps[camera_id] = {
            "map_x": map_x,
            "map_y": map_y,
            "valid": in_bounds.astype(np.float32) * ownership_weight,
        }
        log(
            f"Precomputed side map for cam {camera_id}: {100.0 * in_bounds.mean():.1f}% visible",
        )
    return maps


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 0.5
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_aruco_points(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    points = {}
    if ids is None:
        return points
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        points[int(marker_id)] = marker_corners.reshape(4, 2).astype(np.float32)
    return points


def estimate_pair_homography(reference_frame, moving_frame, detector):
    ref_points = detect_aruco_points(reference_frame, detector)
    moving_points = detect_aruco_points(moving_frame, detector)
    common_ids = sorted(set(ref_points) & set(moving_points))
    if len(common_ids) < 2:
        return None, len(common_ids), 0, None

    src = []
    dst = []
    for marker_id in common_ids:
        src.extend(moving_points[marker_id])
        dst.extend(ref_points[marker_id])
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    return homography, len(common_ids), inliers, src


def smooth_homography(previous, current, alpha=0.85):
    if current is None:
        return previous
    if previous is None:
        return current
    blended = alpha * previous + (1.0 - alpha) * current
    if abs(blended[2, 2]) > 1e-9:
        blended /= blended[2, 2]
    return blended


def median_homography(samples):
    if not samples:
        return None
    normalized = []
    for homography in samples:
        if homography is None:
            continue
        h = np.asarray(homography, dtype=np.float64)
        if abs(h[2, 2]) > 1e-9:
            h = h / h[2, 2]
        normalized.append(h)
    if not normalized:
        return None
    median = np.median(np.stack(normalized, axis=0), axis=0)
    if abs(median[2, 2]) > 1e-9:
        median = median / median[2, 2]
    return median


def is_reasonable_homography(homography, frame_shape, max_scale=4.0):
    if homography is None:
        return False
    height, width = frame_shape[:2]
    points = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(projected)):
        return False
    area = abs(cv2.contourArea(projected.astype(np.float32)))
    source_area = float(width * height)
    if area < source_area / (max_scale * max_scale) or area > source_area * max_scale * max_scale:
        return False
    return True


def floor_hull_mask(shape, points, padding_px=24):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if points is None or len(points) < 3:
        mask[:, :] = 255
        return mask
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    if padding_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding_px * 2 + 1, padding_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def warp_pair_panorama(
    reference_frame,
    moving_frame,
    homography,
    moving_floor_points=None,
    blend_mode="average",
    mask_mode="full",
):
    h_ref, w_ref = reference_frame.shape[:2]
    h_mov, w_mov = moving_frame.shape[:2]
    out_w = min(1800, max(w_ref, int(round(w_ref * 1.45))))
    out_h = min(900, h_ref)
    x_offset = int(round((out_w - w_ref) * 0.5))
    y_offset = int(round((out_h - h_ref) * 0.5))
    translate = np.array([[1.0, 0.0, x_offset], [0.0, 1.0, y_offset], [0.0, 0.0, 1.0]], dtype=np.float64)

    ref_canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = max(0, x_offset)
    y0 = max(0, y_offset)
    x1 = min(x0 + w_ref, out_w)
    y1 = min(y0 + h_ref, out_h)
    ref_canvas[y0:y1, x0:x1] = reference_frame[: y1 - y0, : x1 - x0]

    warped_moving = cv2.warpPerspective(
        moving_frame,
        translate @ homography,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ref_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    ref_mask[y0:y1, x0:x1] = 255
    moving_source_mask = (
        floor_hull_mask(moving_frame.shape, moving_floor_points)
        if mask_mode == "floor"
        else np.full((h_mov, w_mov), 255, dtype=np.uint8)
    )
    moving_mask = cv2.warpPerspective(
        moving_source_mask,
        translate @ homography,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
    )

    ref_valid = ref_mask > 0
    mov_valid = moving_mask > 0
    both = ref_valid & mov_valid
    only_mov = mov_valid & ~ref_valid
    fused = ref_canvas.copy()
    fused[only_mov] = warped_moving[only_mov]
    if np.any(both):
        if blend_mode == "strongest":
            fused[both] = np.maximum(ref_canvas[both], warped_moving[both])
        else:
            fused[both] = (
                0.5 * ref_canvas[both].astype(np.float32)
                + 0.5 * warped_moving[both].astype(np.float32)
            ).astype(np.uint8)
    content = (ref_mask > 0) | (moving_mask > 0)
    if np.any(content):
        ys, xs = np.where(content)
        pad = 12
        crop_x0 = max(0, int(xs.min()) - pad)
        crop_y0 = max(0, int(ys.min()) - pad)
        crop_x1 = min(out_w, int(xs.max()) + pad + 1)
        crop_y1 = min(out_h, int(ys.max()) + pad + 1)
        fused = fused[crop_y0:crop_y1, crop_x0:crop_x1]
    return fused


def make_aruco_panorama(
    display_frames,
    pair_ids,
    detector,
    previous_homography,
    label,
    blend_mode,
    mask_mode,
    stitch_state,
    calibration_frames,
    live_stitch,
):
    if len(pair_ids) < 2:
        return np.zeros((320, 760, 3), dtype=np.uint8), previous_homography, "missing pair"
    ref_id, moving_id = pair_ids[0], pair_ids[1]
    reference = display_frames.get(ref_id)
    moving = display_frames.get(moving_id)
    if reference is None or moving is None:
        return np.zeros((320, 760, 3), dtype=np.uint8), previous_homography, "waiting for frames"
    homography, tags, inliers, moving_floor_points = estimate_pair_homography(reference, moving, detector)
    if homography is not None and inliers >= 8:
        stitch_state["samples"].append(homography)
        stitch_state["latest_floor_points"] = moving_floor_points
        if len(stitch_state["samples"]) >= calibration_frames and stitch_state.get("locked") is None:
            stitch_state["locked"] = median_homography(stitch_state["samples"])
            stitch_state["locked_floor_points"] = moving_floor_points

    if live_stitch:
        active_homography = smooth_homography(previous_homography, homography, alpha=0.95)
        active_floor_points = moving_floor_points
    else:
        active_homography = stitch_state.get("locked")
        active_floor_points = stitch_state.get("locked_floor_points")

    if active_homography is None or active_floor_points is None:
        fallback = np.hstack([resize_to_height(reference, 320), resize_to_height(moving, 320)])
        samples = len(stitch_state["samples"])
        return fallback, previous_homography, f"{label} calibrating stitch {samples}/{calibration_frames} tags {tags}"

    pano = warp_pair_panorama(reference, moving, active_homography, active_floor_points, blend_mode, mask_mode)
    samples = len(stitch_state["samples"])
    lock_label = "live" if live_stitch else ("locked" if stitch_state.get("locked") is not None else f"cal {samples}/{calibration_frames}")
    status = f"{label} ArUco panorama cam {ref_id}+{moving_id} | {lock_label} | {mask_mode} | tags {tags} | inliers {inliers}"
    return pano, active_homography, status


def fuse_side(display_frames, camera_maps, camera_ids, blend_mode):
    first_map = next(iter(camera_maps.values()))
    output_shape = first_map["map_x"].shape
    if blend_mode == "strongest":
        fused = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
        best_weight = np.zeros(output_shape, dtype=np.float32)
    else:
        accum = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)

    for camera_id in camera_ids:
        frame = display_frames.get(camera_id)
        maps = camera_maps.get(camera_id)
        if frame is None or maps is None:
            continue
        warped = cv2.remap(
            frame,
            maps["map_x"],
            maps["map_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        weight = maps["valid"]
        if blend_mode == "strongest":
            take = weight > best_weight
            fused[take] = warped[take]
            best_weight[take] = weight[take]
        else:
            accum += warped.astype(np.float32) * weight[:, :, None]
            weights += weight

    if blend_mode == "strongest":
        return fused
    fused = np.zeros_like(accum, dtype=np.uint8)
    valid = weights > 1e-6
    fused[valid] = np.clip(accum[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return fused


def draw_label(frame, label):
    shown = frame.copy()
    cv2.rectangle(shown, (8, 8), (min(shown.shape[1] - 8, 560), 48), (0, 0, 0), -1)
    cv2.putText(shown, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return shown


def draw_side_guides(frame, geometry, max_z_m):
    shown = frame.copy()
    height, width = shown.shape[:2]
    length = float(geometry["measurements"]["arena_length_m"])
    for frac in (0.25, 0.5, 0.75):
        x = int(round(frac * (width - 1)))
        cv2.line(shown, (x, 0), (x, height - 1), (70, 70, 70), 1)
    floor_y = height - 1
    cv2.line(shown, (0, floor_y), (width - 1, floor_y), (110, 110, 110), 1)
    cv2.putText(shown, f"near {-length/2:.2f}m", (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.putText(shown, f"far +{length/2:.2f}m", (width - 112, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    cv2.putText(shown, f"z {max_z_m:.2f}m", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    return shown


def ordered_source_camera_ids(camera_ids, position_to_camera):
    ordered = []
    for position in POSITION_KEYS:
        camera_id = position_to_camera.get(position)
        if camera_id in camera_ids and camera_id not in ordered:
            ordered.append(camera_id)
    for camera_id in camera_ids:
        if camera_id not in ordered:
            ordered.append(camera_id)
    return ordered


def make_source_grid(
    display_frames,
    labels,
    camera_ids,
    source_height,
    selected_camera_id=None,
    position_to_camera=None,
    blend_mode="average",
    view_mode="aruco",
    mask_mode="full",
):
    if position_to_camera is not None:
        camera_ids = ordered_source_camera_ids(camera_ids, position_to_camera)
    tiles = []
    for camera_id in camera_ids:
        frame = display_frames.get(camera_id)
        if frame is None:
            frame = np.zeros((source_height, int(source_height * 1.6), 3), dtype=np.uint8)
        tile = resize_to_height(frame, source_height)
        position = labels.get(camera_id)
        label = f"{position or 'unassigned'} cam {camera_id}"
        cv2.rectangle(tile, (8, 8), (min(tile.shape[1] - 8, 320), 44), (0, 0, 0), -1)
        cv2.putText(tile, label, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if camera_id == selected_camera_id:
            cv2.rectangle(tile, (2, 2), (tile.shape[1] - 3, tile.shape[0] - 3), (0, 255, 0), 4)
        tiles.append(tile)

    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    rects = []
    for start in range(0, len(tiles), 2):
        row = []
        for offset, tile in enumerate(tiles[start : start + 2]):
            fitted = fit_to_tile(tile, tile_width, tile_height)
            row.append(fitted)
            camera_id = camera_ids[start + offset]
            row_index = len(rows)
            col_index = offset
            rects.append(
                {
                    "camera_id": camera_id,
                    "rect": (
                        col_index * tile_width,
                        row_index * tile_height,
                        (col_index + 1) * tile_width,
                        (row_index + 1) * tile_height,
                    ),
                }
            )
        if len(row) < 2:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    footer_height = 82
    footer = np.zeros((footer_height, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        footer,
        "click tile, press 1=L1 2=L2 3=R1 4=R2 | s save | r recalibrate",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    button_specs = [
        ("blend", "Average", "average"),
        ("blend", "Strong", "strongest"),
        ("mode", "ArUco", "aruco"),
        ("mode", "Plane", "plane"),
        ("mask", "Full", "full"),
        ("mask", "Floor", "floor"),
    ]
    x = 10
    y = 38
    for action, label, value in button_specs:
        is_active = (
            (action == "blend" and blend_mode == value)
            or (action == "mode" and view_mode == value)
            or (action == "mask" and mask_mode == value)
        )
        width = max(68, 14 + len(label) * 10)
        color = (42, 130, 42) if is_active else (42, 42, 42)
        border = (160, 230, 160) if is_active else (155, 155, 155)
        cv2.rectangle(footer, (x, y), (x + width, y + 30), color, -1)
        cv2.rectangle(footer, (x, y), (x + width, y + 30), border, 1)
        cv2.putText(
            footer,
            label,
            (x + 7, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rects.append(
            {
                "action": action,
                "value": value,
                "rect": (x, grid.shape[0] + y, x + width, grid.shape[0] + y + 30),
            }
        )
        x += width + 8
    return np.vstack([grid, footer]), rects


def stack_view(left_view, right_view, sources=None):
    composite_width = max(left_view.shape[1], right_view.shape[1])
    left_view = fit_to_tile(left_view, composite_width, left_view.shape[0])
    right_view = fit_to_tile(right_view, composite_width, right_view.shape[0])
    composites = np.vstack([left_view, right_view])
    if sources is None:
        return composites, (0, 0)
    height = max(composites.shape[0], sources.shape[0])
    left = fit_to_tile(composites, composites.shape[1], height)
    source_y = max(0, (height - sources.shape[0]) // 2)
    right = np.zeros((height, sources.shape[1], 3), dtype=np.uint8)
    right[source_y : source_y + sources.shape[0], :] = sources
    return np.hstack([left, right]), (left.shape[1], source_y)


def main():
    args = parse_args()
    if args.scan:
        for index, device in enumerate(list_devices()):
            log(f"{index}: {device}")
        return
    if args.list_formats:
        print_formats(args.cameras)
        return

    geometry = load_geometry(args.geometry)
    position_to_camera = load_position_assignments(args.config, geometry)
    log(
        "Loaded assignments: "
        + ", ".join(
            f"{POSITION_SHORT[position]}=cam {position_to_camera[position]}"
            for position in POSITION_KEYS
            if position in position_to_camera
        )
    )
    labels = camera_position_labels(position_to_camera)
    side_ids = side_camera_ids(position_to_camera)
    camera_models = load_camera_models(args.poses, args.cameras)
    tuning = load_camera_tuning(args.camera_tuning, args.no_camera_tuning)
    if args.no_camera_tuning:
        tuning = {}

    cameras = []
    writer = None
    selected_camera_id = None
    source_rects = []
    source_offset = (0, 0)
    ui_state = {
        "blend": args.blend,
        "mode": args.mode,
        "mask": args.stitch_mask,
    }

    def reset_stitch_state():
        return {
            "left": {"samples": [], "locked": None, "latest_floor_points": None, "locked_floor_points": None},
            "right": {"samples": [], "locked": None, "latest_floor_points": None, "locked_floor_points": None},
        }

    def handle_mouse(event, x, y, _flags, _userdata):
        nonlocal selected_camera_id
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        offset_x, offset_y = source_offset
        local_x = x - offset_x
        local_y = y - offset_y
        for entry in source_rects:
            x1, y1, x2, y2 = entry["rect"]
            if x1 <= local_x < x2 and y1 <= local_y < y2:
                action = entry.get("action")
                if action in ui_state:
                    ui_state[action] = entry["value"]
                    log(f"{action}: {entry['value']}")
                    return
                selected_camera_id = entry["camera_id"]
                log(f"Selected camera {selected_camera_id}")
                return

    try:
        for camera_id in args.cameras:
            camera_format, used_fallback = find_format(
                camera_id,
                args.format,
                args.width,
                args.height,
                args.allow_format_fallback,
            )
            suffix = " fallback" if used_fallback else ""
            log(
                f"Starting camera {camera_id}: {camera_format['media_type_str'].upper()} "
                f"{camera_format['width']}x{camera_format['height']}{suffix}"
            )
            camera = DShowCamera(camera_id, camera_format["index"])
            camera.start()
            cameras.append(camera)
            log(f"Started camera {camera_id}")

        startup_deadline = time.perf_counter() + args.startup_timeout
        while time.perf_counter() < startup_deadline and not all(camera.has_frame() for camera in cameras):
            time.sleep(0.02)
        for camera in cameras:
            if not camera.has_frame():
                log(f"Warning: camera {camera.device_index} has not delivered an initial frame yet.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
        cv2.createTrackbar("Y+145", WINDOW_NAME, int(args.plane_y_mm + 145), 290, lambda _value: None)
        cv2.createTrackbar("Z mm", WINDOW_NAME, int(args.max_z_mm), 300, lambda _value: None)
        cv2.createTrackbar("Blend mm", WINDOW_NAME, int(args.blend_sigma_mm), 250, lambda _value: None)
        cv2.createTrackbar("Depth", WINDOW_NAME, int(args.min_depth_mm), 120, lambda _value: None)
        cv2.createTrackbar("Angle", WINDOW_NAME, int(args.max_view_angle_deg), 89, lambda _value: None)

        last_params = None
        side_maps = {}
        detector = make_aruco_detector()
        pair_homographies = {"left": None, "right": None}
        stitch_states = reset_stitch_state()
        source_shape = (args.height, args.width, 3)
        frame_times = []
        stop_at = time.perf_counter() + args.duration if args.duration > 0 else None

        while True:
            display_frames = {}
            for camera in cameras:
                if camera.latest_frame is not None:
                    display_frames[camera.device_index] = rotate_frame(camera.latest_frame, args.rotation)

            plane_y_mm = cv2.getTrackbarPos("Y+145", WINDOW_NAME) - 145
            max_z_mm = max(20, cv2.getTrackbarPos("Z mm", WINDOW_NAME))
            blend_sigma_mm = max(5, cv2.getTrackbarPos("Blend mm", WINDOW_NAME))
            min_depth_mm = cv2.getTrackbarPos("Depth", WINDOW_NAME)
            max_angle_deg = max(5, cv2.getTrackbarPos("Angle", WINDOW_NAME))
            blend_mode = ui_state["blend"]
            view_mode = ui_state["mode"]
            mask_mode = ui_state["mask"]
            params = (plane_y_mm, max_z_mm, blend_sigma_mm, min_depth_mm, max_angle_deg)
            if view_mode == "plane" and params != last_params:
                side_maps = {}
                for side_name in SIDE_NAMES:
                    side_maps[side_name] = precompute_side_maps(
                        camera_models,
                        side_ids[side_name],
                        geometry,
                        args.output_width,
                        args.output_height,
                        plane_y_mm / 1000.0,
                        max_z_mm / 1000.0,
                        source_shape,
                        args.rotation,
                        args.point_transform,
                        tuning,
                        min_depth_mm / 1000.0,
                        float(max_angle_deg),
                        blend_sigma_mm / 1000.0,
                    )
                last_params = params

            now = time.perf_counter()
            frame_times.append(now)
            frame_times = [value for value in frame_times if now - value <= 1.0]
            fps = len(frame_times) / max(now - frame_times[0], 1e-6) if len(frame_times) > 1 else 0.0

            if view_mode == "aruco":
                left, pair_homographies["left"], left_status = make_aruco_panorama(
                    display_frames,
                    side_ids["left"],
                    detector,
                    pair_homographies["left"],
                    "left",
                    blend_mode,
                    mask_mode,
                    stitch_states["left"],
                    args.stitch_calibration_frames,
                    args.live_stitch,
                )
                right, pair_homographies["right"], right_status = make_aruco_panorama(
                    display_frames,
                    side_ids["right"],
                    detector,
                    pair_homographies["right"],
                    "right",
                    blend_mode,
                    mask_mode,
                    stitch_states["right"],
                    args.stitch_calibration_frames,
                    args.live_stitch,
                )
                left = draw_label(left, f"{left_status} | {blend_mode} | {fps:.1f} FPS")
                right = draw_label(right, f"{right_status} | {blend_mode}")
            else:
                left = fuse_side(display_frames, side_maps["left"], side_ids["left"], blend_mode)
                right = fuse_side(display_frames, side_maps["right"], side_ids["right"], blend_mode)
                left = draw_side_guides(left, geometry, max_z_mm / 1000.0)
                right = draw_side_guides(right, geometry, max_z_mm / 1000.0)
                left = draw_label(left, f"left side curtain | {blend_mode} | y {plane_y_mm:+d}mm | {fps:.1f} FPS")
                right = draw_label(right, f"right side curtain | {blend_mode} | y {plane_y_mm:+d}mm")

            sources = None
            if args.show_sources:
                sources, source_rects = make_source_grid(
                    display_frames,
                    labels,
                    args.cameras,
                    args.source_height,
                    selected_camera_id,
                    position_to_camera,
                    blend_mode,
                    view_mode,
                    mask_mode,
                )
            view, source_offset = stack_view(left, right, sources)

            if args.record is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.record), fourcc, args.fps, (view.shape[1], view.shape[0]))
            if writer is not None:
                writer.write(view)

            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in ASSIGN_KEYS and selected_camera_id is not None:
                position = ASSIGN_KEYS[key]
                for existing_position, camera_id in list(position_to_camera.items()):
                    if camera_id == selected_camera_id and existing_position != position:
                        del position_to_camera[existing_position]
                position_to_camera[position] = selected_camera_id
                labels = camera_position_labels(position_to_camera)
                side_ids = side_camera_ids(position_to_camera)
                pair_homographies = {"left": None, "right": None}
                stitch_states = reset_stitch_state()
                log(
                    "Assignments: "
                    + ", ".join(
                        f"{POSITION_SHORT[position]}=cam {position_to_camera[position]}"
                        for position in POSITION_KEYS
                        if position in position_to_camera
                    )
                )
            if key == ord("s"):
                save_position_assignments(args.config, args, position_to_camera)
            if key == ord("r"):
                pair_homographies = {"left": None, "right": None}
                stitch_states = reset_stitch_state()
                log("Reset side-view stitch calibration.")
            if stop_at is not None and time.perf_counter() >= stop_at:
                break
    finally:
        if writer is not None:
            writer.release()
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = traceback.format_exc()
        Path("side_view_fusion_error.log").write_text(message, encoding="utf-8")
        log(message)
        raise
