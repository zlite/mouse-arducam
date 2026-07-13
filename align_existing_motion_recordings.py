#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


CAMERA_IDS = {
    "front_1": 0,
    "front_2": 1,
    "back_1": 2,
    "back_2": 3,
    "side_1": 4,
    "side_2": 5,
    "top_1": 8,
    "top_2": 9,
    "top_3": 10,
    "top_4": 11,
}


@dataclass
class Segment:
    role: str
    video_path: Path
    rows: list
    times: list
    frames: list


def parse_args():
    parser = argparse.ArgumentParser(description="Align legacy motion clips by frame timestamps")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-cameras", type=int, default=4)
    return parser.parse_args()


def load_segment(csv_path):
    video_path = csv_path.with_suffix(".mp4")
    if not video_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "unix_time" not in rows[0]:
        return None

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < len(rows):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    count = min(len(rows), len(frames))
    if count == 0:
        return None
    rows = rows[:count]
    frames = frames[:count]
    return Segment(
        role=csv_path.parent.name,
        video_path=video_path,
        rows=rows,
        times=[float(row["unix_time"]) for row in rows],
        frames=frames,
    )


def packet_at(segments, target):
    candidates = []
    for segment in segments:
        if target < segment.times[0] or target > segment.times[-1]:
            continue
        position = bisect.bisect_left(segment.times, target)
        for index in (position - 1, position):
            if 0 <= index < len(segment.times):
                candidates.append((abs(segment.times[index] - target), segment, index))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def true_runs(values):
    runs = []
    start = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def main():
    args = parse_args()
    segments = []
    for csv_path in sorted(args.source.glob("*/*.csv")):
        segment = load_segment(csv_path)
        if segment is not None:
            segments.append(segment)
    if not segments:
        raise SystemExit(f"No timestamped MP4 recordings found under {args.source}")

    by_role = defaultdict(list)
    for segment in segments:
        by_role[segment.role].append(segment)

    period = 1.0 / args.fps
    first_tick = math.ceil(min(segment.times[0] for segment in segments) * args.fps) / args.fps
    last_tick = math.floor(max(segment.times[-1] for segment in segments) * args.fps) / args.fps
    frame_count = int(round((last_tick - first_tick) * args.fps)) + 1
    targets = [first_tick + index * period for index in range(frame_count)]

    args.output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    availability = []
    role_packets = {}
    for role, role_segments in by_role.items():
        role_packets[role] = [packet_at(role_segments, target) for target in targets]

    for sync_index, target in enumerate(targets):
        visible_roles = [role for role, packets in role_packets.items() if packets[sync_index] is not None]
        availability.append(len(visible_roles))
        all_rows.append(
            {
                "sync_index": sync_index,
                "target_unix_time": f"{target:.6f}",
                "available_camera_count": len(visible_roles),
                "available_roles": ";".join(sorted(visible_roles)),
                "meets_minimum": int(len(visible_roles) >= args.min_cameras),
            }
        )

    timestamp_fields = list(all_rows[0])
    with (args.output / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timestamp_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for fallback_id, role in enumerate(sorted(by_role), start=100):
        camera_id = CAMERA_IDS.get(role, fallback_id)
        sample_frame = by_role[role][0].frames[0]
        height, width = sample_frame.shape[:2]
        video_path = args.output / f"cam_{camera_id}.mp4"
        csv_path = args.output / f"cam_{camera_id}.csv"
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {video_path}")
        black = sample_frame * 0
        fields = [
            "sync_index",
            "target_unix_time",
            "source_video",
            "source_frame_index",
            "capture_unix_time",
            "time_error_sec",
            "available",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fields)
            csv_writer.writeheader()
            for sync_index, (target, packet) in enumerate(zip(targets, role_packets[role])):
                if packet is None:
                    writer.write(black)
                    row = {
                        "sync_index": sync_index,
                        "target_unix_time": f"{target:.6f}",
                        "source_video": "",
                        "source_frame_index": "",
                        "capture_unix_time": "",
                        "time_error_sec": "",
                        "available": 0,
                    }
                else:
                    error, segment, source_index = packet
                    writer.write(segment.frames[source_index])
                    row = {
                        "sync_index": sync_index,
                        "target_unix_time": f"{target:.6f}",
                        "source_video": segment.video_path.name,
                        "source_frame_index": source_index,
                        "capture_unix_time": f"{segment.times[source_index]:.6f}",
                        "time_error_sec": f"{error:.6f}",
                        "available": 1,
                    }
                csv_writer.writerow(row)
        writer.release()
        metadata = {
            "video": video_path.name,
            "role": role,
            "frames": frame_count,
            "fps": args.fps,
            "start_time": datetime.fromtimestamp(first_tick).astimezone().isoformat(timespec="milliseconds"),
        }
        (args.output / f"cam_{camera_id}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    qualifying_runs = true_runs([count >= args.min_cameras for count in availability])
    summary = {
        "source": str(args.source.resolve()),
        "fps": args.fps,
        "frames": frame_count,
        "duration_sec": frame_count / args.fps,
        "start_time": datetime.fromtimestamp(first_tick).astimezone().isoformat(timespec="milliseconds"),
        "end_time": datetime.fromtimestamp(last_tick).astimezone().isoformat(timespec="milliseconds"),
        "camera_roles": sorted(by_role),
        "maximum_concurrent_cameras": max(availability),
        "minimum_required_cameras": args.min_cameras,
        "qualifying_intervals": [
            {
                "start_frame": start,
                "end_frame_exclusive": end,
                "start_unix_time": targets[start],
                "end_unix_time": targets[end - 1] + period,
            }
            for start, end in qualifying_runs
        ],
        "note": "Legacy source videos are 10 FPS; repeated frames create the 30 FPS timeline.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
