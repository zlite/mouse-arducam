import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect_hand_keypoints import HAND_CONNECTIONS, LANDMARK_NAMES, require_hand_model, require_mediapipe
from dshow_arducam_viewer import fit_to_tile, resize_to_height


WINDOW_NAME = "Keypoint Event Review"
POSITION_ORDER = ("left_1", "left_2", "right_1", "right_2")
STATE_PATH = Path(".generate_keypoint_events_state.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and review multi-camera keypoints from recorded event folders.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Session folder or root folder containing event_* folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("keypoint_events"))
    parser.add_argument("--keypoint-run-dir", type=Path, default=None, help="Existing keypoint_events/run_* folder to replay.")
    parser.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--choose-folder", action="store_true", help="Open a folder picker for the input directory.")
    parser.add_argument("--no-generate", action="store_true", help="Skip keypoint generation and only replay existing output.")
    parser.add_argument("--no-replay", action="store_true", help="Generate keypoints without opening the replay UI.")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--inference-width", type=int, default=640)
    parser.add_argument("--min-detection-confidence", type=float, default=0.25)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.25)
    parser.add_argument("--display-height", type=int, default=260)
    parser.add_argument("--group-window-sec", type=float, default=1.5)
    return parser.parse_args()


def choose_folder():
    import tkinter as tk
    from tkinter import filedialog

    initial_dir = Path.cwd()
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            remembered = Path(state.get("last_input_dir", ""))
            if remembered.exists():
                initial_dir = remembered
        except Exception:
            pass
    elif Path("mouse_events").exists():
        initial_dir = Path("mouse_events").resolve()

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="Choose recorded event/session folder", initialdir=str(initial_dir))
    root.destroy()
    if not selected:
        return None
    selected_path = Path(selected)
    STATE_PATH.write_text(json.dumps({"last_input_dir": str(selected_path)}, indent=2), encoding="utf-8")
    return selected_path


def has_event_clip(path):
    return path.is_dir() and any(path.glob("*.mp4")) and any(path.glob("*_frames.csv"))


def find_event_dirs(input_dir):
    if has_event_clip(input_dir):
        return [input_dir]

    direct = sorted([path for path in input_dir.glob("event_*") if has_event_clip(path)])
    if direct:
        return direct

    nested = []
    for pattern in ("session_*/event_*", "20*/event_*", "*/event_*"):
        nested.extend([path for path in input_dir.glob(pattern) if has_event_clip(path)])

    event_dirs = []
    seen = set()
    for path in sorted(nested):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        event_dirs.append(path)
    return event_dirs


def load_clip(event_dir):
    metadata_path = event_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    videos = sorted(event_dir.glob("*.mp4"))
    csvs = sorted(event_dir.glob("*_frames.csv"))
    if not videos or not csvs:
        return None
    with csvs[0].open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    if not frame_rows:
        return None
    try:
        start_perf = float(frame_rows[0]["perf_time"])
    except Exception:
        start_perf = float(metadata.get("started_at_unix", 0.0))
    return {
        "event_dir": event_dir,
        "metadata": metadata,
        "video": videos[0],
        "frames_csv": csvs[0],
        "frame_rows": frame_rows,
        "start_perf": start_perf,
        "camera_id": int(metadata.get("camera_id", -1)),
        "label": metadata.get("label", event_dir.name),
    }


def group_clips(clips, group_window_sec):
    clips = sorted(clips, key=lambda clip: clip["start_perf"])
    groups = []
    for clip in clips:
        if not groups or abs(clip["start_perf"] - groups[-1]["start_perf"]) > group_window_sec:
            groups.append({"start_perf": clip["start_perf"], "clips": [clip]})
        else:
            groups[-1]["clips"].append(clip)
    for index, group in enumerate(groups):
        labels = "_".join(sorted(str(clip["label"]) for clip in group["clips"]))
        group["name"] = f"event_group_{index:03d}_{labels}"
    return groups


def groups_from_summary(run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No summary.json found in {run_dir}")
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = []
    for entry in summaries:
        clips = []
        for clip_summary in entry.get("clips", []):
            video_path = Path(clip_summary.get("video", ""))
            if not video_path.exists():
                continue
            clip = load_clip(video_path.parent)
            if clip is not None:
                clips.append(clip)
        if clips:
            groups.append(
                {
                    "start_perf": min(clip["start_perf"] for clip in clips),
                    "clips": clips,
                    "name": entry.get("group", f"event_group_{len(groups):03d}"),
                    "keypoints_csv": run_dir / entry.get("keypoints_csv", f"{entry.get('group', '')}_keypoints.csv"),
                }
            )
    return groups


def resize_for_inference(frame, target_width):
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame
    scale = target_width / float(frame.shape[1])
    return cv2.resize(frame, (target_width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)


def make_keypoint_fields():
    fields = ["group", "camera_id", "label", "frame_index", "unix_time", "perf_time", "hand_index", "handedness", "hand_score"]
    for name in LANDMARK_NAMES:
        fields.extend([f"{name}_x_px", f"{name}_y_px", f"{name}_z_rel", f"{name}_x_norm", f"{name}_y_norm"])
    return fields


def blank_keypoints(row):
    for name in LANDMARK_NAMES:
        row[f"{name}_x_px"] = ""
        row[f"{name}_y_px"] = ""
        row[f"{name}_z_rel"] = ""
        row[f"{name}_x_norm"] = ""
        row[f"{name}_y_norm"] = ""


def add_keypoints(row, landmarks, source_width, source_height, inference_width, inference_height):
    scale_x = source_width / float(inference_width)
    scale_y = source_height / float(inference_height)
    for name, landmark in zip(LANDMARK_NAMES, landmarks):
        row[f"{name}_x_px"] = f"{landmark.x * inference_width * scale_x:.2f}"
        row[f"{name}_y_px"] = f"{landmark.y * inference_height * scale_y:.2f}"
        row[f"{name}_z_rel"] = f"{landmark.z:.6f}"
        row[f"{name}_x_norm"] = f"{landmark.x:.6f}"
        row[f"{name}_y_norm"] = f"{landmark.y:.6f}"


def build_landmarker(mp, args):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, landmarks, inference_width, inference_height):
    if not landmarks:
        return frame
    height, width = frame.shape[:2]
    scale_x = width / float(inference_width)
    scale_y = height / float(inference_height)
    for hand in landmarks:
        points = [(int(round(lm.x * inference_width * scale_x)), int(round(lm.y * inference_height * scale_y))) for lm in hand]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
        for point_index, point in enumerate(points):
            color = (0, 0, 255) if point_index in (4, 8, 12, 16, 20) else (255, 255, 0)
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
    return frame


def generate_group(mp, group, run_dir, args):
    keypoint_path = run_dir / f"{group['name']}_keypoints.csv"
    summary = {"group": group["name"], "clips": []}
    fields = make_keypoint_fields()

    with keypoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip in sorted(group["clips"], key=lambda item: POSITION_ORDER.index(item["label"]) if item["label"] in POSITION_ORDER else 99):
            cap = cv2.VideoCapture(str(clip["video"]))
            if not cap.isOpened():
                continue
            source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            detected_frames = 0
            processed_frames = 0
            base_perf = None
            last_ms = -1
            started = time.perf_counter()
            with build_landmarker(mp, args) as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok or processed_frames >= len(clip["frame_rows"]):
                        break
                    frame_row = clip["frame_rows"][processed_frames]
                    perf = float(frame_row.get("perf_time") or processed_frames / 30.0)
                    if base_perf is None:
                        base_perf = perf
                    timestamp_ms = int(round((perf - base_perf) * 1000.0))
                    if timestamp_ms <= last_ms:
                        timestamp_ms = last_ms + 1
                    last_ms = timestamp_ms

                    inference = resize_for_inference(frame, args.inference_width)
                    rgb = cv2.cvtColor(inference, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                    hands = result.hand_landmarks or []
                    handedness = result.handedness or []
                    if hands:
                        detected_frames += 1
                    for hand_index, landmarks in enumerate(hands):
                        category_name = ""
                        category_score = ""
                        if hand_index < len(handedness) and handedness[hand_index]:
                            category_name = handedness[hand_index][0].category_name
                            category_score = f"{handedness[hand_index][0].score:.6f}"
                        row = {
                            "group": group["name"],
                            "camera_id": clip["camera_id"],
                            "label": clip["label"],
                            "frame_index": frame_row.get("frame_index", processed_frames),
                            "unix_time": frame_row.get("unix_time", ""),
                            "perf_time": frame_row.get("perf_time", ""),
                            "hand_index": hand_index,
                            "handedness": category_name,
                            "hand_score": category_score,
                        }
                        blank_keypoints(row)
                        add_keypoints(row, landmarks, source_width, source_height, inference.shape[1], inference.shape[0])
                        writer.writerow(row)
                    processed_frames += 1
            cap.release()
            elapsed = max(time.perf_counter() - started, 1e-9)
            summary["clips"].append(
                {
                    "label": clip["label"],
                    "camera_id": clip["camera_id"],
                    "video": str(clip["video"]),
                    "frames": processed_frames,
                    "detected_frames": detected_frames,
                    "processing_fps": processed_frames / elapsed,
                }
            )
            print(f"{group['name']} {clip['label']} cam {clip['camera_id']}: {detected_frames}/{processed_frames} frames with keypoints", flush=True)

    summary["keypoints_csv"] = keypoint_path.name
    return summary


def load_keypoints(csv_path):
    by_camera_frame = {}
    if not csv_path.exists():
        return by_camera_frame
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["camera_id"]), int(row["frame_index"]))
            points = []
            for name in LANDMARK_NAMES:
                x = row.get(f"{name}_x_px", "")
                y = row.get(f"{name}_y_px", "")
                if x == "" or y == "":
                    points.append(None)
                else:
                    points.append((float(x), float(y)))
            by_camera_frame.setdefault(key, []).append(points)
    return by_camera_frame


def draw_csv_points(frame, hands):
    for points in hands:
        for start, end in HAND_CONNECTIONS:
            if points[start] is not None and points[end] is not None:
                cv2.line(frame, tuple(map(int, points[start])), tuple(map(int, points[end])), (0, 255, 0), 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            if point is None:
                continue
            color = (0, 0, 255) if index in (4, 8, 12, 16, 20) else (255, 255, 0)
            cv2.circle(frame, tuple(map(int, point)), 4, color, -1, cv2.LINE_AA)
    return frame


def make_replay_grid(frames, display_height):
    order = sorted(frames, key=lambda item: POSITION_ORDER.index(item[0]) if item[0] in POSITION_ORDER else 99)
    tiles = []
    for label, camera_id, frame, frame_index in order:
        frame = frame.copy()
        cv2.rectangle(frame, (4, 4), (min(frame.shape[1] - 4, 420), 36), (0, 0, 0), -1)
        cv2.putText(frame, f"{label} cam {camera_id} frame {frame_index}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(resize_to_height(frame, display_height))
    if not tiles:
        return None
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), 2):
        row = [fit_to_tile(tile, tile_w, tile_h) for tile in tiles[start : start + 2]]
        if len(row) == 1:
            row.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def replay_group(group, keypoint_csv, args):
    keypoints = load_keypoints(keypoint_csv)
    caps = []
    try:
        for clip in group["clips"]:
            cap = cv2.VideoCapture(str(clip["video"]))
            if cap.isOpened():
                caps.append((clip, cap))
        if not caps:
            return
        paused = False
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frames = []
            for clip, cap in caps:
                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frame = cap.read()
                if not ok:
                    return
                hands = keypoints.get((clip["camera_id"], frame_index), [])
                draw_csv_points(frame, hands)
                frames.append((clip["label"], clip["camera_id"], frame, frame_index))
            grid = make_replay_grid(frames, args.display_height)
            if grid is not None:
                cv2.imshow(WINDOW_NAME, grid)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                for _clip, cap in caps:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    finally:
        for _clip, cap in caps:
            cap.release()
        cv2.destroyWindow(WINDOW_NAME)


def main():
    args = parse_args()
    if args.keypoint_run_dir is not None:
        args.keypoint_run_dir = args.keypoint_run_dir.resolve()
    if args.choose_folder or (args.input_dir is None and args.keypoint_run_dir is None):
        args.input_dir = choose_folder()
    if args.input_dir is None and args.keypoint_run_dir is None:
        raise SystemExit("No input folder selected.")

    if args.keypoint_run_dir is not None and args.no_generate:
        groups = groups_from_summary(args.keypoint_run_dir)
        run_dir = args.keypoint_run_dir
        print(f"Selected keypoint run: {run_dir}", flush=True)
        print(f"Loaded {len(groups)} replay event(s).", flush=True)
    else:
        args.input_dir = args.input_dir.resolve()
        require_hand_model(args.hand_model)
        mp = require_mediapipe()

        event_dirs = find_event_dirs(args.input_dir)
        print(f"Selected input: {args.input_dir}", flush=True)
        print(f"Found {len(event_dirs)} camera clip folder(s).", flush=True)

        clips = [load_clip(path) for path in event_dirs]
        clips = [clip for clip in clips if clip is not None]
        if not clips:
            raise SystemExit(
                f"No event clips found under {args.input_dir}\n"
                "Choose the timestamped session folder, the parent mouse_events folder, "
                "or one single event_* camera folder."
            )
        groups = group_clips(clips, args.group_window_sec)
        run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        if not args.no_generate:
            for group in groups:
                summaries.append(generate_group(mp, group, run_dir, args))
            (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print(f"Wrote keypoint run: {run_dir}", flush=True)

    print(f"Grouped into {len(groups)} event(s).", flush=True)
    for group in groups:
        labels = ", ".join(f"{clip['label']} cam {clip['camera_id']}" for clip in group["clips"])
        print(f"  {group['name']}: {labels}", flush=True)

    if not args.no_replay:
        for group in groups:
            keypoint_csv = group.get("keypoints_csv", run_dir / f"{group['name']}_keypoints.csv")
            replay_group(group, keypoint_csv, args)


if __name__ == "__main__":
    main()
