import argparse
import json
import os
import time
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

import cv2
import numpy as np

from dshow_arducam_viewer import DShowCamera, find_format_index, fit_to_tile, resize_to_height


cv2.ocl.setUseOpenCL(False)


POINT_TRANSFORMS = ("normal", "flip_x", "flip_y", "flip_xy")


def parse_args():
    parser = argparse.ArgumentParser(description="Live floor-plane bird's-eye fusion from the four Arducams.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--poses", type=Path, default=Path("camera_poses_manual.npz"))
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--camera-tuning", type=Path, default=Path("gods_eye_camera_tuning.json"))
    parser.add_argument("--no-camera-tuning", action="store_true")
    parser.add_argument(
        "--floor-warp",
        type=Path,
        default=Path("floor_checkerboard_warps.json"),
        help="Optional checkerboard-derived empirical floor warp file.",
    )
    parser.add_argument("--no-floor-warp", action="store_true", help="Ignore checkerboard-derived floor warps.")
    parser.add_argument(
        "--manual-hull-margin-mm",
        type=float,
        default=15.0,
        help="For manual floor warps, only trust the clicked floor-point hull plus this margin.",
    )
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument(
        "--point-transform",
        choices=POINT_TRANSFORMS,
        default="flip_xy",
        help="Transform from raw camera model pixels into displayed/rotated frames.",
    )
    parser.add_argument(
        "--projection-offsets",
        type=Path,
        default=None,
        help="Optional tracker-derived display-pixel offsets. Disabled by default for floor warping.",
    )
    parser.add_argument("--no-projection-offsets", action="store_true")
    parser.add_argument("--output-width", type=int, default=520)
    parser.add_argument("--pixels-per-meter", type=float, default=1400.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument(
        "--blend",
        choices=("average", "nearest"),
        default="average",
        help="Average all valid camera warps, or pick the physically nearest side camera per output pixel.",
    )
    parser.add_argument(
        "--debug-camera",
        type=int,
        default=None,
        help="Start with only one camera enabled in the fused view, useful for diagnosing bad poses/transforms.",
    )
    parser.add_argument(
        "--start-disabled",
        action="store_true",
        help="Start with all camera contributions disabled; turn them on one at a time with the use-cam sliders.",
    )
    parser.add_argument(
        "--min-camera-depth",
        type=float,
        default=0.01,
        help="Meters in front of the camera required before a floor point is sampled.",
    )
    parser.add_argument(
        "--max-view-angle-deg",
        type=float,
        default=82.0,
        help="Reject floor points farther off-axis than this angle.",
    )
    parser.add_argument(
        "--pitch-offset-deg",
        type=float,
        default=0.0,
        help="Diagnostic camera pitch offset around each camera's local x axis. Try +/-10..35 if the floor grid is high/low.",
    )
    parser.add_argument(
        "--interactive-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show OpenCV sliders for pitch/depth/view-angle diagnostics.",
    )
    parser.add_argument("--show-sources", action="store_true", help="Show camera source tiles next to the fused view.")
    parser.add_argument(
        "--source-floor-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the projected arena floor outline/grid on the source camera tiles.",
    )
    parser.add_argument("--source-height", type=int, default=220)
    parser.add_argument("--record", type=Path, default=None, help="Optional MP4 path to record the fused top-down view.")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def load_geometry(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_camera_models(path, camera_ids):
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
        missing = [value for value in required.values() if value not in data]
        if missing:
            raise RuntimeError(f"Missing camera model entries in {path}: {missing}")
        rotation, _ = cv2.Rodrigues(data[required["rvec"]])
        models[camera_id] = {
            "camera_matrix": data[required["camera_matrix"]],
            "dist_coeffs": data[required["dist_coeffs"]],
            "rotation": rotation,
            "rvec": data[required["rvec"]],
            "tvec": data[required["tvec"]].reshape(3, 1),
            "center": data[required["camera_center"]].reshape(3),
        }
    return models


def apply_pitch_offset(camera_models, pitch_degrees):
    if abs(pitch_degrees) < 1e-9:
        return camera_models
    angle = np.deg2rad(pitch_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )
    adjusted = {}
    for camera_id, model in camera_models.items():
        new_model = dict(model)
        rotation = pitch @ model["rotation"]
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = -rotation @ model["center"].reshape(3, 1)
        new_model["rotation"] = rotation
        new_model["rvec"] = rvec
        new_model["tvec"] = tvec
        adjusted[camera_id] = new_model
    print(f"Applied diagnostic pitch offset: {pitch_degrees:+.1f} deg", flush=True)
    return adjusted


def apply_single_pitch_offset(model, pitch_degrees):
    if abs(pitch_degrees) < 1e-9:
        return dict(model)
    angle = np.deg2rad(pitch_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )
    adjusted = dict(model)
    rotation = pitch @ model["rotation"]
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ model["center"].reshape(3, 1)
    adjusted["rotation"] = rotation
    adjusted["rvec"] = rvec
    adjusted["tvec"] = tvec
    return adjusted


def camera_position_labels(geometry):
    labels = {}
    for position_name, spec in geometry.get("cameras", {}).items():
        labels[int(spec["camera_index"])] = position_name
    return labels


def load_projection_offsets(path, disabled):
    if disabled or path is None or not path.exists():
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


def load_camera_tuning(path, disabled):
    if disabled or path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    tuning = {}
    for camera_id, spec in data.get("cameras", {}).items():
        tuning[int(camera_id)] = {
            "pitch_deg": float(spec.get("pitch_deg", 0.0)),
            "min_depth_m": float(spec.get("min_depth_m", 0.0)),
            "max_view_angle_deg": float(spec.get("max_view_angle_deg", 82.0)),
            "point_transform": spec.get("point_transform", "flip_xy"),
        }
    if tuning:
        print(f"Loaded camera tuning from {path}: {sorted(tuning)}", flush=True)
    return tuning


def load_floor_warps(path, disabled):
    if disabled or path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    warps = {}
    for camera_id, spec in data.get("cameras", {}).items():
        homography = np.asarray(spec.get("homography_world_to_image"), dtype=np.float64)
        if homography.shape != (3, 3):
            continue
        warps[int(camera_id)] = {
            "homography_world_to_image": homography,
            "rms_px": float(spec.get("rms_px", float("nan"))),
            "points": int(spec.get("points", 0)),
            "inliers": int(spec.get("inliers", 0)),
            "manual_world_points": np.asarray(
                [pair["world_m"] for pair in spec.get("manual_pairs", [])],
                dtype=np.float32,
            ),
        }
    if warps:
        details = ", ".join(
            f"cam {camera_id}: {warps[camera_id]['rms_px']:.2f}px"
            for camera_id in sorted(warps)
        )
        print(f"Loaded checkerboard floor warps from {path}: {details}", flush=True)
    return warps


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def rotate_points(points, width, height, degrees):
    x = points[:, 0]
    y = points[:, 1]
    degrees %= 360
    if degrees == 90:
        return np.column_stack([height - 1 - y, x])
    if degrees == 180:
        return np.column_stack([width - 1 - x, height - 1 - y])
    if degrees == 270:
        return np.column_stack([y, width - 1 - x])
    return points.copy()


def display_points_from_model(points, width, height, rotation_degrees, point_transform):
    x = points[:, 0]
    y = points[:, 1]
    if point_transform == "normal":
        base = points.copy()
    elif point_transform == "flip_x":
        base = np.column_stack([width - 1 - x, y])
    elif point_transform == "flip_y":
        base = np.column_stack([x, height - 1 - y])
    elif point_transform == "flip_xy":
        base = np.column_stack([width - 1 - x, height - 1 - y])
    else:
        raise ValueError(f"Unknown point transform: {point_transform}")
    base = np.nan_to_num(base, nan=-1e6, posinf=1e6, neginf=-1e6)
    base = np.clip(base, -1e6, 1e6)
    return rotate_points(base.astype(np.float32), width, height, rotation_degrees)


def make_world_grid(geometry, output_width, pixels_per_meter, floor_z):
    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])
    output_height = max(1, int(round(arena_length * pixels_per_meter)))
    output_width = max(1, int(output_width))
    xs = np.linspace(-arena_width / 2.0, arena_width / 2.0, output_width, dtype=np.float32)
    ys = np.linspace(arena_length / 2.0, -arena_length / 2.0, output_height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    world = np.column_stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.full(grid_x.size, floor_z, dtype=np.float32),
        ]
    )
    return world, (output_height, output_width), arena_width, arena_length


def precompute_camera_maps(
    camera_models,
    camera_ids,
    world_points,
    output_shape,
    source_shape,
    rotation_degrees,
    point_transform,
    projection_offsets,
    min_camera_depth,
    max_view_angle_deg,
):
    raw_height, raw_width = source_shape[:2]
    output_height, output_width = output_shape
    maps = {}
    for camera_id in camera_ids:
        model = camera_models[camera_id]
        image_points, _ = cv2.projectPoints(
            world_points.reshape(-1, 1, 3).astype(np.float64),
            model["rvec"],
            model["tvec"],
            model["camera_matrix"],
            model["dist_coeffs"],
        )
        display_points = display_points_from_model(
            image_points.reshape(-1, 2),
            raw_width,
            raw_height,
            rotation_degrees,
            point_transform,
        )
        if camera_id in projection_offsets:
            display_points += projection_offsets[camera_id]

        camera_points = (model["rotation"] @ world_points.T + model["tvec"]).T
        camera_depth = camera_points[:, 2]
        lateral = np.linalg.norm(camera_points[:, :2], axis=1)
        max_lateral = np.tan(np.deg2rad(max_view_angle_deg)) * np.maximum(camera_depth, 1e-9)

        map_x = display_points[:, 0].reshape(output_height, output_width).astype(np.float32)
        map_y = display_points[:, 1].reshape(output_height, output_width).astype(np.float32)
        valid = (
            (map_x >= 0)
            & (map_x <= raw_width - 1)
            & (map_y >= 0)
            & (map_y <= raw_height - 1)
            & (camera_depth.reshape(output_height, output_width) > min_camera_depth)
            & (lateral.reshape(output_height, output_width) <= max_lateral.reshape(output_height, output_width))
        )
        maps[camera_id] = {"map_x": map_x, "map_y": map_y, "valid": valid.astype(np.float32)}
        print(
            f"Precomputed floor map for cam {camera_id}: "
            f"{100.0 * valid.mean():.1f}% valid coverage (not an alignment score)",
            flush=True,
        )
    return maps


def manual_world_hull_mask(warp, world_points, output_shape, margin_m):
    manual_world = warp.get("manual_world_points")
    if manual_world is None or len(manual_world) < 3:
        return np.ones(output_shape, dtype=bool)

    output_height, output_width = output_shape
    x_min = float(np.min(world_points[:, 0]))
    x_max = float(np.max(world_points[:, 0]))
    y_min = float(np.min(world_points[:, 1]))
    y_max = float(np.max(world_points[:, 1]))
    px = (manual_world[:, 0] - x_min) / max(x_max - x_min, 1e-9) * (output_width - 1)
    py = (y_max - manual_world[:, 1]) / max(y_max - y_min, 1e-9) * (output_height - 1)
    hull_points = np.column_stack([px, py]).astype(np.float32)
    hull = cv2.convexHull(hull_points).reshape(-1, 2)

    mask = np.zeros(output_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    if margin_m > 0:
        pixels_per_meter = 0.5 * (
            output_width / max(x_max - x_min, 1e-9)
            + output_height / max(y_max - y_min, 1e-9)
        )
        radius = max(1, int(round(margin_m * pixels_per_meter)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def precompute_empirical_floor_maps(floor_warps, camera_ids, world_points, output_shape, source_shape, hull_margin_m=0.015):
    raw_height, raw_width = source_shape[:2]
    output_height, output_width = output_shape
    xy = world_points[:, :2].astype(np.float32).reshape(-1, 1, 2)
    maps = {}
    for camera_id in camera_ids:
        warp = floor_warps.get(camera_id)
        if warp is None:
            continue
        image_points = cv2.perspectiveTransform(xy, warp["homography_world_to_image"]).reshape(-1, 2)
        map_x = image_points[:, 0].reshape(output_height, output_width).astype(np.float32)
        map_y = image_points[:, 1].reshape(output_height, output_width).astype(np.float32)
        hull_mask = manual_world_hull_mask(warp, world_points, output_shape, hull_margin_m)
        valid = (
            (map_x >= 0)
            & (map_x <= raw_width - 1)
            & (map_y >= 0)
            & (map_y <= raw_height - 1)
            & hull_mask
        )
        maps[camera_id] = {"map_x": map_x, "map_y": map_y, "valid": valid.astype(np.float32)}
        print(
            f"Precomputed checkerboard floor map for cam {camera_id}: "
            f"{100.0 * valid.mean():.1f}% trusted coverage, fit {warp['rms_px']:.2f}px",
            flush=True,
        )
    return maps


def project_world_points_to_display(
    camera_id,
    points,
    camera_models,
    source_shape,
    rotation_degrees,
    point_transform,
    projection_offsets,
):
    raw_height, raw_width = source_shape[:2]
    model = camera_models[camera_id]
    image_points, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 3),
        model["rvec"],
        model["tvec"],
        model["camera_matrix"],
        model["dist_coeffs"],
    )
    display_points = display_points_from_model(
        image_points.reshape(-1, 2),
        raw_width,
        raw_height,
        rotation_degrees,
        point_transform,
    )
    if camera_id in projection_offsets:
        display_points += projection_offsets[camera_id]
    camera_points = (model["rotation"] @ np.asarray(points, dtype=np.float64).reshape(-1, 3).T + model["tvec"]).T
    return display_points, camera_points


def floor_overlay_points(geometry, floor_z):
    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])
    xs = np.linspace(-arena_width / 2.0, arena_width / 2.0, 5)
    ys = np.linspace(-arena_length / 2.0, arena_length / 2.0, 5)
    lines = []
    for x in xs:
        y_line = np.linspace(-arena_length / 2.0, arena_length / 2.0, 60)
        lines.append(np.column_stack([np.full_like(y_line, x), y_line, np.full_like(y_line, floor_z)]))
    for y in ys:
        x_line = np.linspace(-arena_width / 2.0, arena_width / 2.0, 60)
        lines.append(np.column_stack([x_line, np.full_like(x_line, y), np.full_like(x_line, floor_z)]))
    return lines


def draw_source_floor_overlay(
    frame,
    camera_id,
    camera_models,
    geometry,
    source_shape,
    rotation_degrees,
    point_transform,
    projection_offsets,
    floor_z,
    min_camera_depth=0.01,
    max_view_angle_deg=82.0,
):
    frame = frame.copy()
    for line in floor_overlay_points(geometry, floor_z):
        points, camera_points = project_world_points_to_display(
            camera_id,
            line,
            camera_models,
            source_shape,
            rotation_degrees,
            point_transform,
            projection_offsets,
        )
        if not np.all(np.isfinite(points)):
            continue
        depth = camera_points[:, 2]
        lateral = np.linalg.norm(camera_points[:, :2], axis=1)
        max_lateral = np.tan(np.deg2rad(max_view_angle_deg)) * np.maximum(depth, 1e-9)
        valid = (depth > min_camera_depth) & (lateral <= max_lateral)
        for index in range(len(points) - 1):
            if not (valid[index] and valid[index + 1]):
                continue
            p1 = tuple(np.round(points[index]).astype(int))
            p2 = tuple(np.round(points[index + 1]).astype(int))
            cv2.line(frame, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_empirical_source_floor_overlay(frame, camera_id, floor_warps, geometry, floor_z):
    frame = frame.copy()
    warp = floor_warps.get(camera_id)
    if warp is None:
        return frame
    height, width = frame.shape[:2]
    for line in floor_overlay_points(geometry, floor_z):
        xy = line[:, :2].astype(np.float32).reshape(-1, 1, 2)
        points = cv2.perspectiveTransform(xy, warp["homography_world_to_image"]).reshape(-1, 2)
        finite = np.all(np.isfinite(points), axis=1)
        in_bounds = (
            finite
            & (points[:, 0] >= 0)
            & (points[:, 0] <= width - 1)
            & (points[:, 1] >= 0)
            & (points[:, 1] <= height - 1)
        )
        for index in range(len(points) - 1):
            if not (in_bounds[index] and in_bounds[index + 1]):
                continue
            p1 = tuple(np.round(points[index]).astype(int))
            p2 = tuple(np.round(points[index + 1]).astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (8, 42), (min(width - 8, 250), 68), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"manual warp {warp['rms_px']:.1f}px",
        (16, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame


def nearest_camera_masks(camera_models, camera_ids, world_points, output_shape, valid_maps):
    centers = np.asarray([camera_models[camera_id]["center"] for camera_id in camera_ids], dtype=np.float64)
    xy = world_points[:, :2].astype(np.float64)
    distances = []
    for center in centers:
        distances.append(np.linalg.norm(xy - center[:2], axis=1))
    order = np.argsort(np.asarray(distances), axis=0)
    masks = {camera_id: np.zeros(output_shape, dtype=np.float32) for camera_id in camera_ids}
    flat_valid = {camera_id: valid_maps[camera_id].reshape(-1) > 0 for camera_id in camera_ids}
    for pixel_index in range(world_points.shape[0]):
        for camera_rank in order[:, pixel_index]:
            camera_id = camera_ids[int(camera_rank)]
            if flat_valid[camera_id][pixel_index]:
                masks[camera_id].reshape(-1)[pixel_index] = 1.0
                break
    return masks


def draw_arena_overlay(frame, geometry, labels, fps, subtitle="floor fusion"):
    overlay = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)
    cv2.line(overlay, (width // 2, 0), (width // 2, height - 1), (90, 90, 90), 1)
    cv2.line(overlay, (0, height // 2), (width - 1, height // 2), (90, 90, 90), 1)

    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])

    def world_to_panel(x, y):
        px = int(round((x + arena_width / 2.0) / arena_width * (width - 1)))
        py = int(round((arena_length / 2.0 - y) / arena_length * (height - 1)))
        return px, py

    for name, spec in geometry.get("cameras", {}).items():
        x, y, _z = spec["center_m"]
        px, py = world_to_panel(x, y)
        cv2.circle(overlay, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(overlay, name, (px + 7, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.rectangle(overlay, (8, 8), (min(width - 8, 455), 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"God's-eye {subtitle} | {fps:4.1f} FPS",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)


def fuse_top_down(display_frames, camera_maps, camera_ids, blend, nearest_masks=None, debug_camera=None):
    first_map = next(iter(camera_maps.values()))
    output_shape = first_map["map_x"].shape
    accum = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.float32)
    weights = np.zeros(output_shape, dtype=np.float32)

    for camera_id in camera_ids:
        if debug_camera is not None and camera_id != debug_camera:
            continue
        frame = display_frames.get(camera_id)
        if frame is None:
            continue
        maps = camera_maps[camera_id]
        warped = cv2.remap(
            frame,
            maps["map_x"],
            maps["map_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if debug_camera is not None:
            weight = maps["valid"]
        elif blend == "nearest" and nearest_masks is not None:
            weight = nearest_masks[camera_id] * maps["valid"]
        else:
            weight = maps["valid"]
        accum += warped.astype(np.float32) * weight[:, :, None]
        weights += weight

    fused = np.zeros_like(accum, dtype=np.uint8)
    valid = weights > 1e-6
    fused[valid] = np.clip(accum[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    return fused


def make_source_grid(
    display_frames,
    labels,
    cameras,
    source_height,
    camera_models=None,
    geometry=None,
    source_shape=None,
    rotation_degrees=180,
    point_transform="normal",
    projection_offsets=None,
    floor_z=0.0,
    draw_floor_overlay=True,
    min_camera_depth=0.01,
    max_view_angle_deg=82.0,
    camera_tuning=None,
    active_camera_ids=None,
    floor_warps=None,
):
    active_camera_ids = set(active_camera_ids or [])
    tiles = []
    for camera in cameras:
        is_active = camera.device_index in active_camera_ids
        frame = display_frames.get(camera.device_index)
        if frame is None:
            frame = np.zeros((source_height, int(source_height * 16 / 9), 3), dtype=np.uint8)
        else:
            if draw_floor_overlay and geometry is not None and camera.device_index in (floor_warps or {}):
                frame = draw_empirical_source_floor_overlay(
                    frame,
                    camera.device_index,
                    floor_warps,
                    geometry,
                    floor_z,
                )
            elif draw_floor_overlay and camera_models is not None and geometry is not None and source_shape is not None:
                tuning = (camera_tuning or {}).get(camera.device_index, {})
                camera_transform = tuning.get("point_transform", point_transform)
                camera_min_depth = float(tuning.get("min_depth_m", min_camera_depth))
                camera_max_angle = float(tuning.get("max_view_angle_deg", max_view_angle_deg))
                frame = draw_source_floor_overlay(
                    frame,
                    camera.device_index,
                    camera_models,
                    geometry,
                    source_shape,
                    rotation_degrees,
                    camera_transform,
                    projection_offsets or {},
                    floor_z,
                    camera_min_depth,
                    camera_max_angle,
                )
            frame = resize_to_height(frame, source_height)
        label = labels.get(camera.device_index, f"cam_{camera.device_index}")
        label_color = (255, 255, 255) if is_active else (150, 150, 150)
        border_color = (0, 220, 0) if is_active else (90, 90, 90)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, 2)
        cv2.rectangle(frame, (6, 6), (min(frame.shape[1] - 6, 250), 32), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{label} cam {camera.device_index}{'' if is_active else ' off'}",
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            label_color,
            1,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), 2):
        row = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + 2]]
        if len(row) < 2:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    args = parse_args()
    geometry = load_geometry(args.geometry)
    base_camera_models = load_camera_models(args.poses, args.cameras)
    labels = camera_position_labels(geometry)
    projection_offsets = load_projection_offsets(args.projection_offsets, args.no_projection_offsets)
    camera_tuning = load_camera_tuning(args.camera_tuning, args.no_camera_tuning)
    floor_warps = load_floor_warps(args.floor_warp, args.no_floor_warp)
    world_points, output_shape, _arena_width, _arena_length = make_world_grid(
        geometry,
        args.output_width,
        args.pixels_per_meter,
        args.floor_z,
    )

    def rebuild_maps(pitch_offset_deg, min_camera_depth, max_view_angle_deg, point_transform):
        active_models = {}
        maps = {}
        for camera_id in args.cameras:
            tuning = camera_tuning.get(camera_id, {})
            camera_pitch = float(tuning.get("pitch_deg", 0.0)) + pitch_offset_deg
            camera_transform = tuning.get("point_transform", point_transform)
            camera_min_depth = float(tuning.get("min_depth_m", min_camera_depth))
            camera_max_angle = float(tuning.get("max_view_angle_deg", max_view_angle_deg))
            active_models[camera_id] = apply_single_pitch_offset(base_camera_models[camera_id], camera_pitch)
            if camera_id in floor_warps:
                maps.update(
                    precompute_empirical_floor_maps(
                        floor_warps,
                        [camera_id],
                        world_points,
                        output_shape,
                        (args.height, args.width, 3),
                        args.manual_hull_margin_mm / 1000.0,
                    )
                )
            else:
                maps.update(
                    precompute_camera_maps(
                        active_models,
                        [camera_id],
                        world_points,
                        output_shape,
                        (args.height, args.width, 3),
                        args.rotation,
                        camera_transform,
                        projection_offsets,
                        camera_min_depth,
                        camera_max_angle,
                    )
                )
        masks = None
        if args.blend == "nearest":
            masks = nearest_camera_masks(
                active_models,
                args.cameras,
                world_points,
                output_shape,
                {camera_id: maps[camera_id]["valid"] for camera_id in args.cameras},
            )
        return active_models, maps, masks

    current_pitch = float(args.pitch_offset_deg)
    current_min_depth = float(args.min_camera_depth)
    current_max_angle = float(args.max_view_angle_deg)
    current_point_transform = args.point_transform
    camera_models, camera_maps, nearest_masks = rebuild_maps(
        current_pitch,
        current_min_depth,
        current_max_angle,
        current_point_transform,
    )

    cameras = []
    writer = None
    try:
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        if args.record is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.record), fourcc, args.fps, (output_shape[1], output_shape[0]))
            print(f"Recording fused view to {args.record}", flush=True)

        cv2.namedWindow("God's Eye Fusion", cv2.WINDOW_NORMAL)
        if args.interactive_controls:
            cv2.createTrackbar("pitch +45", "God's Eye Fusion", int(round(current_pitch + 45.0)), 90, lambda _value: None)
            cv2.createTrackbar("min depth mm", "God's Eye Fusion", int(round(current_min_depth * 1000.0)), 200, lambda _value: None)
            cv2.createTrackbar("max angle deg", "God's Eye Fusion", int(round(current_max_angle)), 120, lambda _value: None)
            cv2.createTrackbar("transform", "God's Eye Fusion", POINT_TRANSFORMS.index(current_point_transform), len(POINT_TRANSFORMS) - 1, lambda _value: None)
            for camera_id in args.cameras:
                if args.start_disabled:
                    enabled = 0
                elif args.debug_camera is not None:
                    enabled = 1 if camera_id == args.debug_camera else 0
                else:
                    enabled = 1
                cv2.createTrackbar(f"use cam {camera_id}", "God's Eye Fusion", enabled, 1, lambda _value: None)
        initial_enabled = {}
        for camera_id in args.cameras:
            if args.start_disabled:
                initial_enabled[camera_id] = False
            elif args.debug_camera is not None:
                initial_enabled[camera_id] = camera_id == args.debug_camera
            else:
                initial_enabled[camera_id] = True
        started_at = time.perf_counter()
        frames = 0
        while True:
            if args.interactive_controls:
                active_camera_ids = [
                    camera_id
                    for camera_id in args.cameras
                    if cv2.getTrackbarPos(f"use cam {camera_id}", "God's Eye Fusion") > 0
                ]
            else:
                active_camera_ids = [camera_id for camera_id in args.cameras if initial_enabled[camera_id]]

            if args.interactive_controls:
                slider_pitch = cv2.getTrackbarPos("pitch +45", "God's Eye Fusion") - 45.0
                slider_min_depth = cv2.getTrackbarPos("min depth mm", "God's Eye Fusion") / 1000.0
                slider_max_angle = max(1.0, float(cv2.getTrackbarPos("max angle deg", "God's Eye Fusion")))
                slider_transform = POINT_TRANSFORMS[
                    min(cv2.getTrackbarPos("transform", "God's Eye Fusion"), len(POINT_TRANSFORMS) - 1)
                ]
                if (
                    abs(slider_pitch - current_pitch) > 1e-9
                    or abs(slider_min_depth - current_min_depth) > 1e-9
                    or abs(slider_max_angle - current_max_angle) > 1e-9
                    or slider_transform != current_point_transform
                ):
                    current_pitch = slider_pitch
                    current_min_depth = slider_min_depth
                    current_max_angle = slider_max_angle
                    current_point_transform = slider_transform
                    camera_models, camera_maps, nearest_masks = rebuild_maps(
                        current_pitch,
                        current_min_depth,
                        current_max_angle,
                        current_point_transform,
                    )

            display_frames = {}
            for camera in cameras:
                if camera.latest_frame is not None:
                    display_frames[camera.device_index] = rotate_frame(camera.latest_frame, args.rotation)

            fused = fuse_top_down(
                display_frames,
                camera_maps,
                active_camera_ids,
                args.blend,
                nearest_masks,
                None,
            )
            frames += 1
            fps = frames / max(time.perf_counter() - started_at, 1e-9)
            active_text = ",".join(str(camera_id) for camera_id in active_camera_ids) if active_camera_ids else "none"
            subtitle = f"{args.blend} floor fusion cams {active_text}"
            subtitle = (
                f"{subtitle} pitch {current_pitch:+.0f} depth {current_min_depth*1000:.0f}mm "
                f"angle {current_max_angle:.0f} {current_point_transform}"
            )
            fused_overlay = draw_arena_overlay(fused, geometry, labels, fps, subtitle)

            shown = fused_overlay
            if args.show_sources:
                source_grid = make_source_grid(
                    display_frames,
                    labels,
                    cameras,
                    args.source_height,
                    camera_models,
                    geometry,
                    (args.height, args.width, 3),
                    args.rotation,
                    current_point_transform,
                    projection_offsets,
                    args.floor_z,
                    args.source_floor_overlay,
                    current_min_depth,
                    current_max_angle,
                    camera_tuning,
                    active_camera_ids,
                    floor_warps,
                )
                shown = np.hstack([fit_to_tile(fused_overlay, fused_overlay.shape[1], source_grid.shape[0]), source_grid])

            cv2.imshow("God's Eye Fusion", shown)
            if writer is not None:
                writer.write(fused_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                transform_index = key - ord("0")
                current_point_transform = POINT_TRANSFORMS[transform_index]
                if args.interactive_controls:
                    cv2.setTrackbarPos("transform", "God's Eye Fusion", transform_index)
                camera_models, camera_maps, nearest_masks = rebuild_maps(
                    current_pitch,
                    current_min_depth,
                    current_max_angle,
                    current_point_transform,
                )
    finally:
        for camera in cameras:
            camera.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
