import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline hand/finger keypoint detection on saved mouse event clips."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("mouse_events"))
    parser.add_argument("--output-dir", type=Path, default=Path("hand_keypoints"))
    parser.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument(
        "--selection",
        choices=("latest", "all"),
        default="latest",
        help="Process the latest event batch or every event in the input directory.",
    )
    parser.add_argument(
        "--latest-window-sec",
        type=float,
        default=30.0,
        help="When --selection latest, include event dirs modified within this many seconds of the newest event.",
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.45)
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1),
        default=1,
        help="0 is faster, 1 is usually more accurate.",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=640,
        help="Resize frames to this width for detection. 0 keeps original size.",
    )
    parser.add_argument("--no-annotated-video", action="store_true")
    parser.add_argument("--annotated-fps", type=float, default=30.0)
    return parser.parse_args()


def require_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe is not installed. Install it with:\n"
            "  python -m pip install mediapipe\n"
            "Then rerun this script."
        ) from exc
    return mp


def require_hand_model(path):
    if not path.exists():
        raise SystemExit(
            f"Hand model not found: {path}\n"
            "Download the MediaPipe hand landmarker model to that path, or pass --hand-model."
        )


def event_dirs(input_dir, selection, latest_window_sec):
    events = sorted(
        [path for path in input_dir.glob("event_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if selection == "all" or not events:
        return list(reversed(events))
    newest = events[0].stat().st_mtime
    selected = [
        path
        for path in events
        if newest - path.stat().st_mtime <= latest_window_sec
    ]
    return list(reversed(selected))


def load_event_files(event_dir):
    metadata_path = event_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    mp4s = sorted(event_dir.glob("*.mp4"))
    csvs = sorted(event_dir.glob("*_frames.csv"))
    if not mp4s or not csvs:
        return None

    with csvs[0].open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))

    return {
        "event_dir": event_dir,
        "metadata": metadata,
        "video_path": mp4s[0],
        "frames_csv": csvs[0],
        "frame_rows": frame_rows,
    }


def make_output_fields():
    fields = [
        "event",
        "camera_id",
        "label",
        "frame_index",
        "unix_time",
        "perf_time",
        "hand_index",
        "handedness",
        "hand_score",
    ]
    for name in LANDMARK_NAMES:
        fields.extend(
            [
                f"{name}_x_px",
                f"{name}_y_px",
                f"{name}_z_rel",
                f"{name}_x_norm",
                f"{name}_y_norm",
            ]
        )
    return fields


def resize_for_inference(frame, target_width):
    if target_width <= 0:
        return frame, 1.0
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0
    scale = target_width / float(width)
    resized = cv2.resize(
        frame,
        (target_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def blank_landmark_fields(row):
    for name in LANDMARK_NAMES:
        row[f"{name}_x_px"] = ""
        row[f"{name}_y_px"] = ""
        row[f"{name}_z_rel"] = ""
        row[f"{name}_x_norm"] = ""
        row[f"{name}_y_norm"] = ""


def add_landmarks(row, landmarks, width, height):
    for name, landmark in zip(LANDMARK_NAMES, landmarks):
        row[f"{name}_x_px"] = f"{landmark.x * width:.2f}"
        row[f"{name}_y_px"] = f"{landmark.y * height:.2f}"
        row[f"{name}_z_rel"] = f"{landmark.z:.6f}"
        row[f"{name}_x_norm"] = f"{landmark.x:.6f}"
        row[f"{name}_y_norm"] = f"{landmark.y:.6f}"


def draw_hand_overlay(frame, hand_landmarks, source_width, source_height, inference_width, inference_height):
    if not hand_landmarks:
        return frame

    scale_x = source_width / float(inference_width)
    scale_y = source_height / float(inference_height)
    for landmarks in hand_landmarks:
        points = []
        for landmark in landmarks:
            points.append((int(round(landmark.x * inference_width * scale_x)), int(round(landmark.y * inference_height * scale_y))))
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            color = (0, 0, 255) if index in (4, 8, 12, 16, 20) else (255, 255, 0)
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
    return frame


def process_event(mp, event, output_dir, args):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    metadata = event["metadata"]
    camera_id = metadata.get("camera_id", "")
    label = metadata.get("label", "")
    event_name = event["event_dir"].name

    cap = cv2.VideoCapture(str(event["video_path"]))
    if not cap.isOpened():
        print(f"Could not open {event['video_path']}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = event["frame_rows"]

    out_csv = output_dir / f"{event_name}_hand_keypoints.csv"
    annotated_path = output_dir / f"{event_name}_hand_keypoints.mp4"
    writer = None
    if not args.no_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, args.annotated_fps, (width, height))

    fields = make_output_fields()
    detected_frames = 0
    detected_hands = 0
    processed = 0
    started = time.perf_counter()
    base_perf_time = None
    last_timestamp_ms = -1

    base_options = mp_python.BaseOptions(model_asset_path=str(args.hand_model))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    with vision.HandLandmarker.create_from_options(options) as hands, out_csv.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if processed >= len(rows):
                break

            frame_row = rows[processed]
            try:
                perf_time = float(frame_row.get("perf_time") or 0.0)
            except ValueError:
                perf_time = processed / max(args.annotated_fps, 1e-9)
            if base_perf_time is None:
                base_perf_time = perf_time
            timestamp_ms = int(round((perf_time - base_perf_time) * 1000.0))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            inference_frame, scale = resize_for_inference(frame, args.inference_width)
            rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hands.detect_for_video(mp_image, timestamp_ms)

            hand_landmarks = results.hand_landmarks or []
            handednesses = results.handedness or []
            hand_count = len(hand_landmarks)
            if hand_count:
                detected_frames += 1
                detected_hands += hand_count

            for hand_index, landmarks in enumerate(hand_landmarks):
                handedness = ""
                score = ""
                if hand_index < len(handednesses) and handednesses[hand_index]:
                    category = handednesses[hand_index][0]
                    handedness = category.category_name
                    score = f"{category.score:.6f}"
                out_row = {
                    "event": event_name,
                    "camera_id": camera_id,
                    "label": label,
                    "frame_index": frame_row.get("frame_index", processed),
                    "unix_time": frame_row.get("unix_time", ""),
                    "perf_time": frame_row.get("perf_time", ""),
                    "hand_index": hand_index,
                    "handedness": handedness,
                    "hand_score": score,
                }
                blank_landmark_fields(out_row)
                add_landmarks(out_row, landmarks, width, height)
                csv_writer.writerow(out_row)

            if writer is not None:
                annotated = frame.copy()
                annotated = draw_hand_overlay(
                    annotated,
                    hand_landmarks,
                    width,
                    height,
                    inference_frame.shape[1],
                    inference_frame.shape[0],
                )
                text = f"{label} cam {camera_id} | hands {hand_count}"
                cv2.rectangle(annotated, (6, 6), (min(width - 8, 430), 42), (0, 0, 0), -1)
                cv2.putText(annotated, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(annotated)

            processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "event": event_name,
        "camera_id": camera_id,
        "label": label,
        "video_frames": frame_count,
        "timestamp_rows": len(rows),
        "processed_frames": processed,
        "detected_frames": detected_frames,
        "detected_hands": detected_hands,
        "processing_fps": processed / elapsed,
        "keypoints_csv": out_csv.name,
        "annotated_video": annotated_path.name if writer is not None else "",
    }


def main():
    args = parse_args()
    mp = require_mediapipe()
    require_hand_model(args.hand_model)

    selected = event_dirs(args.input_dir, args.selection, args.latest_window_sec)
    event_files = [load_event_files(event_dir) for event_dir in selected]
    event_files = [event for event in event_files if event is not None]
    if not event_files:
        raise SystemExit(f"No event videos found in {args.input_dir}")

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for event in event_files:
        print(f"Processing {event['event_dir'].name}...", flush=True)
        summary = process_event(mp, event, run_dir, args)
        if summary is not None:
            summaries.append(summary)
            print(
                f"  {summary['label']} cam {summary['camera_id']}: "
                f"{summary['detected_frames']}/{summary['processed_frames']} frames with hands, "
                f"{summary['processing_fps']:.1f} FPS",
                flush=True,
            )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
