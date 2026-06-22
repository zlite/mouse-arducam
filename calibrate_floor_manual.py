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


WINDOW = "Manual Floor Warp Calibration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually match camera pixels to top-down floor points and save empirical floor warps."
    )
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--geometry", type=Path, default=Path("manual_rig_geometry.json"))
    parser.add_argument("--output", type=Path, default=Path("floor_checkerboard_warps.json"))
    parser.add_argument("--source-height", type=int, default=620)
    parser.add_argument("--map-width", type=int, default=520)
    parser.add_argument(
        "--grid-mm",
        type=float,
        default=0.0,
        help="Optional top-down snap/grid spacing in mm. Use checker square size, or 0 for no snapping.",
    )
    parser.add_argument("--grid-origin-x-mm", type=float, default=0.0)
    parser.add_argument("--grid-origin-y-mm", type=float, default=0.0)
    parser.add_argument("--checker-cols", type=int, default=0, help="Optional checkerboard square columns to draw.")
    parser.add_argument("--checker-rows", type=int, default=0, help="Optional checkerboard square rows to draw.")
    parser.add_argument(
        "--checker-center-x-mm",
        type=float,
        default=0.0,
        help="Checkerboard center x position in arena coordinates.",
    )
    parser.add_argument(
        "--checker-center-y-mm",
        type=float,
        default=0.0,
        help="Checkerboard center y position in arena coordinates.",
    )
    return parser.parse_args()


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def load_geometry(path):
    return json.loads(path.read_text(encoding="utf-8"))


def camera_position_labels(geometry):
    labels = {}
    for position_name, spec in geometry.get("cameras", {}).items():
        labels[int(spec["camera_index"])] = position_name
    return labels


def load_existing_calibration(path):
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    calibrations = data.get("cameras", {}).copy()
    pairs = {}
    for camera_id, spec in calibrations.items():
        pair_list = spec.get("manual_pairs", [])
        if pair_list:
            pairs[int(camera_id)] = [
                {
                    "image": np.asarray(pair["image_px"], dtype=np.float32),
                    "world": np.asarray(pair["world_m"], dtype=np.float32),
                }
                for pair in pair_list
            ]
    if calibrations:
        print(f"Loaded existing floor warp entries from {path}: {sorted(calibrations)}", flush=True)
    return calibrations, pairs


def fit_homography(pairs):
    if len(pairs) < 4:
        return None, None, None
    world = np.asarray([pair["world"] for pair in pairs], dtype=np.float32)
    image = np.asarray([pair["image"] for pair in pairs], dtype=np.float32)
    homography, inlier_mask = cv2.findHomography(world, image, cv2.RANSAC, 4.0)
    if homography is None:
        return None, None, None
    projected = cv2.perspectiveTransform(world.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - image, axis=1)
    inliers = inlier_mask.reshape(-1).astype(bool) if inlier_mask is not None else np.ones(len(errors), dtype=bool)
    rms = float(np.sqrt(np.mean(np.square(errors[inliers])))) if np.any(inliers) else float("nan")
    return homography, inliers, rms


def save_calibration(path, args, geometry, calibrations, all_pairs):
    payload = {
        "type": "manual_floor_homography",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry": {
            "arena_width_m": float(geometry["measurements"]["arena_width_m"]),
            "arena_length_m": float(geometry["measurements"]["arena_length_m"]),
        },
        "source": {
            "width": args.width,
            "height": args.height,
            "rotation": args.rotation,
            "format": args.format,
        },
        "manual_grid": {
            "grid_m": args.grid_mm / 1000.0,
            "origin_m": [args.grid_origin_x_mm / 1000.0, args.grid_origin_y_mm / 1000.0],
            "checker_squares": [args.checker_cols, args.checker_rows],
            "checker_center_m": [args.checker_center_x_mm / 1000.0, args.checker_center_y_mm / 1000.0],
        },
        "cameras": calibrations,
    }
    for camera_id, pairs in all_pairs.items():
        if str(camera_id) in payload["cameras"]:
            payload["cameras"][str(camera_id)]["manual_pairs"] = [
                {
                    "image_px": pair["image"].astype(float).tolist(),
                    "world_m": pair["world"].astype(float).tolist(),
                }
                for pair in pairs
            ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def world_to_panel(point, geometry, panel_rect):
    x, y = point
    left, top, width, height = panel_rect
    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])
    px = left + int(round((x + arena_width / 2.0) / arena_width * (width - 1)))
    py = top + int(round((arena_length / 2.0 - y) / arena_length * (height - 1)))
    return px, py


def panel_to_world(x, y, geometry, panel_rect):
    left, top, width, height = panel_rect
    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])
    wx = ((x - left) / max(width - 1, 1)) * arena_width - arena_width / 2.0
    wy = arena_length / 2.0 - ((y - top) / max(height - 1, 1)) * arena_length
    return np.asarray([wx, wy], dtype=np.float32)


def checker_bounds(args):
    grid_m = args.grid_mm / 1000.0
    if grid_m <= 0 or args.checker_cols <= 0 or args.checker_rows <= 0:
        return None
    center_x = args.checker_center_x_mm / 1000.0
    center_y = args.checker_center_y_mm / 1000.0
    width = args.checker_cols * grid_m
    height = args.checker_rows * grid_m
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
        grid_m,
    )


def snap_world(point, args):
    grid_m = args.grid_mm / 1000.0
    if grid_m <= 0:
        return point
    bounds = checker_bounds(args)
    if bounds is not None:
        x_min, y_min, x_max, y_max, _grid_m = bounds
        snapped = np.asarray(
            [
                round((point[0] - x_min) / grid_m) * grid_m + x_min,
                round((point[1] - y_min) / grid_m) * grid_m + y_min,
            ],
            dtype=np.float32,
        )
        snapped[0] = np.clip(snapped[0], x_min, x_max)
        snapped[1] = np.clip(snapped[1], y_min, y_max)
        return snapped
    origin = np.asarray([args.grid_origin_x_mm / 1000.0, args.grid_origin_y_mm / 1000.0], dtype=np.float32)
    return np.round((point - origin) / grid_m) * grid_m + origin


def draw_checkerboard(panel, geometry, args, panel_rect):
    bounds = checker_bounds(args)
    if bounds is None:
        return False
    x_min, y_min, _x_max, _y_max, grid_m = bounds
    for row in range(args.checker_rows):
        for col in range(args.checker_cols):
            x0 = x_min + col * grid_m
            x1 = x0 + grid_m
            y0 = y_min + row * grid_m
            y1 = y0 + grid_m
            p0 = world_to_panel((x0, y1), geometry, panel_rect)
            p1 = world_to_panel((x1, y0), geometry, panel_rect)
            color = (235, 235, 235) if (row + col) % 2 else (35, 35, 35)
            cv2.rectangle(panel, p0, p1, color, -1)
            cv2.rectangle(panel, p0, p1, (150, 150, 150), 1)
    for col in range(args.checker_cols + 1):
        x = x_min + col * grid_m
        p0 = world_to_panel((x, y_min), geometry, panel_rect)
        p1 = world_to_panel((x, y_min + args.checker_rows * grid_m), geometry, panel_rect)
        cv2.line(panel, p0, p1, (80, 80, 80), 1)
    for row in range(args.checker_rows + 1):
        y = y_min + row * grid_m
        p0 = world_to_panel((x_min, y), geometry, panel_rect)
        p1 = world_to_panel((x_min + args.checker_cols * grid_m, y), geometry, panel_rect)
        cv2.line(panel, p0, p1, (80, 80, 80), 1)
    return True


def draw_topdown(geometry, args, panel_width, active_pairs, pending_world=None):
    arena_width = float(geometry["measurements"]["arena_width_m"])
    arena_length = float(geometry["measurements"]["arena_length_m"])
    panel_height = max(1, int(round(panel_width * arena_length / arena_width)))
    panel = np.full((panel_height, panel_width, 3), 245, dtype=np.uint8)
    panel_rect = (0, 0, panel_width, panel_height)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), (20, 20, 20), 2)
    cv2.line(panel, (panel_width // 2, 0), (panel_width // 2, panel_height - 1), (190, 190, 190), 1)
    cv2.line(panel, (0, panel_height // 2), (panel_width - 1, panel_height // 2), (190, 190, 190), 1)

    drew_checkerboard = draw_checkerboard(panel, geometry, args, panel_rect)

    grid_m = args.grid_mm / 1000.0
    if grid_m > 0 and not drew_checkerboard:
        origin_x = args.grid_origin_x_mm / 1000.0
        origin_y = args.grid_origin_y_mm / 1000.0
        x = origin_x
        while x > -arena_width / 2.0:
            x -= grid_m
        while x <= arena_width / 2.0:
            px, _py = world_to_panel((x, 0.0), geometry, panel_rect)
            cv2.line(panel, (px, 0), (px, panel_height - 1), (215, 215, 215), 1)
            x += grid_m
        y = origin_y
        while y > -arena_length / 2.0:
            y -= grid_m
        while y <= arena_length / 2.0:
            _px, py = world_to_panel((0.0, y), geometry, panel_rect)
            cv2.line(panel, (0, py), (panel_width - 1, py), (215, 215, 215), 1)
            y += grid_m

    for index, pair in enumerate(active_pairs, start=1):
        px, py = world_to_panel(pair["world"], geometry, panel_rect)
        cv2.circle(panel, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(panel, str(index), (px + 7, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if pending_world is not None:
        px, py = world_to_panel(pending_world, geometry, panel_rect)
        cv2.drawMarker(panel, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

    cv2.rectangle(panel, (8, 8), (min(panel_width - 8, 420), 38), (0, 0, 0), -1)
    cv2.putText(panel, "top-down: click matching floor point", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def draw_source(frame, camera_id, label, pairs, pending_image, homography_rms, source_height):
    source = resize_to_height(frame, source_height)
    scale = source.shape[0] / frame.shape[0]
    for index, pair in enumerate(pairs, start=1):
        px, py = np.round(pair["image"] * scale).astype(int)
        cv2.circle(source, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(source, str(index), (px + 7, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if pending_image is not None:
        px, py = np.round(pending_image * scale).astype(int)
        cv2.drawMarker(source, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)

    status = f"{label} cam {camera_id} | pairs {len(pairs)}"
    if homography_rms is not None:
        status += f" | fit {homography_rms:.2f}px"
    cv2.rectangle(source, (8, 8), (min(source.shape[1] - 8, 620), 42), (0, 0, 0), -1)
    cv2.putText(source, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(source, (8, source.shape[0] - 38), (min(source.shape[1] - 8, 900), source.shape[0] - 8), (0, 0, 0), -1)
    cv2.putText(
        source,
        "click camera point, then top-down point | s save | u undo | c clear | q quit",
        (18, source.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return source, scale


def make_calibration_entry(homography, inliers, rms, pairs):
    return {
        "homography_world_to_image": homography.tolist(),
        "rms_px": rms,
        "inliers": int(np.count_nonzero(inliers)),
        "points": int(len(pairs)),
        "method": "manual_click_homography",
    }


def top_left_tile(frame, width, height):
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    tile[: frame.shape[0], : frame.shape[1]] = frame
    return tile


def main():
    args = parse_args()
    geometry = load_geometry(args.geometry)
    labels = camera_position_labels(geometry)
    calibrations, saved_pairs = load_existing_calibration(args.output)
    pairs_by_camera = {camera_id: saved_pairs.get(camera_id, []) for camera_id in args.cameras}
    pending_image = None
    current_layout = {}
    selected_index = 0
    cameras = []

    def on_mouse(event, x, y, _flags, _userdata):
        nonlocal pending_image
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        camera_id = args.cameras[selected_index]
        if event == cv2.EVENT_RBUTTONDOWN:
            if pending_image is not None:
                pending_image = None
            elif pairs_by_camera[camera_id]:
                pairs_by_camera[camera_id].pop()
            return

        source_rect = current_layout.get("source_rect")
        map_rect = current_layout.get("map_rect")
        source_scale = current_layout.get("source_scale")
        if source_rect is None or map_rect is None or source_scale is None:
            return
        sx, sy, sw, sh = source_rect
        mx, my, mw, mh = map_rect
        if sx <= x < sx + sw and sy <= y < sy + sh:
            pending_image = np.asarray([(x - sx) / source_scale, (y - sy) / source_scale], dtype=np.float32)
            return
        if mx <= x < mx + mw and my <= y < my + mh and pending_image is not None:
            world = panel_to_world(x - mx, y - my, geometry, (0, 0, mw, mh))
            world = snap_world(world, args)
            pairs_by_camera[camera_id].append({"image": pending_image.copy(), "world": world})
            pending_image = None

    try:
        for camera_id in args.cameras:
            format_index = find_format_index(camera_id, args.format, args.width, args.height)
            camera = DShowCamera(camera_id, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {camera_id}: {args.format.upper()} {args.width}x{args.height}", flush=True)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("camera", WINDOW, 0, len(args.cameras) - 1, lambda value: None)
        cv2.setMouseCallback(WINDOW, on_mouse)
        print(
            "Click a floor/checker intersection in the camera view, then its matching top-down location. "
            "Press s to save, u to undo, c to clear current camera.",
            flush=True,
        )

        previous_selected_index = selected_index
        while True:
            selected_index = min(cv2.getTrackbarPos("camera", WINDOW), len(args.cameras) - 1)
            if selected_index != previous_selected_index:
                pending_image = None
                previous_selected_index = selected_index
            camera_id = args.cameras[selected_index]
            camera = next(camera for camera in cameras if camera.device_index == camera_id)
            frame = camera.latest_frame
            if frame is None:
                source = np.zeros((args.source_height, int(args.source_height * 16 / 9), 3), dtype=np.uint8)
                source_scale = 1.0
                cv2.putText(source, f"cam {camera_id} waiting", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                display = rotate_frame(frame, args.rotation)
                homography, inliers, rms = fit_homography(pairs_by_camera[camera_id])
                source, source_scale = draw_source(
                    display,
                    camera_id,
                    labels.get(camera_id, f"cam_{camera_id}"),
                    pairs_by_camera[camera_id],
                    pending_image,
                    rms,
                    args.source_height,
                )

            topdown = draw_topdown(geometry, args, args.map_width, pairs_by_camera[camera_id])
            height = max(source.shape[0], topdown.shape[0])
            source_tile = top_left_tile(source, source.shape[1], height)
            topdown_tile = top_left_tile(topdown, topdown.shape[1], height)
            shown = np.hstack([source_tile, topdown_tile])

            current_layout["source_rect"] = (0, 0, source.shape[1], source.shape[0])
            current_layout["map_rect"] = (source_tile.shape[1], 0, topdown.shape[1], topdown.shape[0])
            current_layout["source_scale"] = source_scale

            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if pending_image is not None:
                    pending_image = None
                elif pairs_by_camera[camera_id]:
                    pairs_by_camera[camera_id].pop()
            elif key == ord("c"):
                pairs_by_camera[camera_id].clear()
                pending_image = None
            elif key == ord("s"):
                saved_any = False
                for save_camera_id, pairs in pairs_by_camera.items():
                    homography, inliers, rms = fit_homography(pairs)
                    if homography is None:
                        print(f"cam {save_camera_id}: need at least 4 good point pairs before saving", flush=True)
                        continue
                    calibrations[str(save_camera_id)] = make_calibration_entry(homography, inliers, rms, pairs)
                    saved_any = True
                    print(
                        f"cam {save_camera_id}: saved {len(pairs)} pairs, "
                        f"{int(np.count_nonzero(inliers))} inliers, fit {rms:.2f}px",
                        flush=True,
                    )
                if saved_any:
                    save_calibration(args.output, args, geometry, calibrations, pairs_by_camera)
                    print(f"Saved manual floor warps -> {args.output}", flush=True)
            elif key in (ord("n"), ord("]")):
                selected_index = (selected_index + 1) % len(args.cameras)
                cv2.setTrackbarPos("camera", WINDOW, selected_index)
                pending_image = None
            elif key in (ord("p"), ord("[")):
                selected_index = (selected_index - 1) % len(args.cameras)
                cv2.setTrackbarPos("camera", WINDOW, selected_index)
                pending_image = None
    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
