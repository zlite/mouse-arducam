import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

from dshow_arducam_viewer import fit_to_tile, resize_to_height


TIMER_RE = re.compile(r"\d+(?:[.:]\d+)?")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one-frame timer sync statistics from a timer_sync_capture.py session."
    )
    parser.add_argument("session", nargs="?", type=Path, default=None, help="Session folder. Defaults to newest timer_sync_captures/*.")
    parser.add_argument("--captures-dir", type=Path, default=Path("timer_sync_captures"))
    parser.add_argument("--target-time", type=float, default=None, help="Elapsed seconds to sample. Defaults to common overlap midpoint.")
    parser.add_argument(
        "--four-500ms",
        action="store_true",
        help="Analyze four frames centered in four 500 ms bins: 0.25, 0.75, 1.25, and 1.75 s.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Timer ROI as x,y,w,h. Values <= 1 are treated as fractions of image width/height.",
    )
    parser.add_argument("--display-height", type=int, default=220)
    parser.add_argument("--output-prefix", default="timer_sync_one_frame")
    parser.add_argument("--grid-ms", type=float, default=10.0, help="Scatter plot grid spacing in milliseconds.")
    parser.add_argument("--x-range-ms", type=float, default=100.0, help="Scatter plot X-axis span in milliseconds.")
    parser.add_argument("--y-range-ms", type=float, default=100.0, help="Scatter plot Y-axis span in milliseconds.")
    parser.add_argument("--tesseract-config", default="--psm 7 -c tessedit_char_whitelist=0123456789.:")
    return parser.parse_args()


def newest_session(captures_dir):
    sessions = [path for path in captures_dir.glob("*") if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"No capture sessions found in {captures_dir}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def read_frame_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["written_index"] = int(row["written_index"])
            row["source_frame_index"] = int(row["source_frame_index"])
            row["elapsed_s"] = float(row["elapsed_s"])
            row["elapsed_ns"] = int(row["elapsed_ns"])
            rows.append(row)
    return rows


def camera_id_from_csv(path):
    match = re.match(r"cam_(\d+)_frames\.csv", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_session_rows(session_dir):
    cameras = {}
    for csv_path in sorted(session_dir.glob("cam_*_frames.csv")):
        camera_id = camera_id_from_csv(csv_path)
        if camera_id is None:
            continue
        rows = read_frame_csv(csv_path)
        cameras[camera_id] = rows
    if not cameras:
        raise RuntimeError(f"No cam_*_frames.csv files with rows found in {session_dir}")
    return cameras


def choose_target_time(cameras, explicit_target):
    if explicit_target is not None:
        return explicit_target
    first = max(rows[0]["elapsed_s"] for rows in cameras.values())
    last = min(rows[-1]["elapsed_s"] for rows in cameras.values())
    if first <= last:
        return (first + last) / 2.0
    return np.median([rows[len(rows) // 2]["elapsed_s"] for rows in cameras.values()])


def nearest_row(rows, target_time):
    return min(rows, key=lambda row: abs(row["elapsed_s"] - target_time))


def parse_roi(roi_text, shape):
    if not roi_text:
        return None
    height, width = shape[:2]
    values = [float(part.strip()) for part in roi_text.split(",")]
    if len(values) != 4:
        raise ValueError("--roi must have four values: x,y,w,h")
    x, y, w, h = values
    if max(abs(v) for v in values) <= 1.0:
        x *= width
        w *= width
        y *= height
        h *= height
    x = max(0, min(width - 1, int(round(x))))
    y = max(0, min(height - 1, int(round(y))))
    w = max(1, min(width - x, int(round(w))))
    h = max(1, min(height - y, int(round(h))))
    return x, y, w, h


def auto_timer_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    search = np.zeros_like(gray)
    x0 = int(width * 0.15)
    x1 = int(width * 0.90)
    y0 = int(height * 0.05)
    y1 = int(height * 0.55)
    search[y0:y1, x0:x1] = gray[y0:y1, x0:x1]

    threshold = max(145, int(np.percentile(search[search > 0], 98)) if np.any(search > 0) else 180)
    mask = cv2.inRange(search, threshold, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    components = []
    for index in range(1, num):
        x, y, w, h, area = stats[index]
        cx, cy = centroids[index]
        if 6 <= area <= 2000 and 3 <= w <= 80 and 3 <= h <= 60:
            components.append((x, y, w, h, area, cx, cy))
    if not components:
        return None

    best = None
    for seed in components:
        _sx, _sy, _sw, _sh, _area, seed_x, seed_y = seed
        cluster = [
            component
            for component in components
            if abs(component[5] - seed_x) < width * 0.10 and abs(component[6] - seed_y) < height * 0.055
        ]
        if len(cluster) < 2:
            continue

        x_min = min(component[0] for component in cluster)
        y_min = min(component[1] for component in cluster)
        x_max = max(component[0] + component[2] for component in cluster)
        y_max = max(component[1] + component[3] for component in cluster)
        box_width = x_max - x_min
        box_height = y_max - y_min
        aspect = box_width / max(box_height, 1)
        if not (20 <= box_width <= 220 and 8 <= box_height <= 90 and 1.0 <= aspect <= 8.0):
            continue

        pad_x = max(8, int(box_width * 0.45))
        pad_y = max(8, int(box_height * 0.80))
        x = max(0, x_min - pad_x)
        y = max(0, y_min - pad_y)
        w = min(width - x, box_width + 2 * pad_x)
        h = min(height - y, box_height + 2 * pad_y)
        crop_mean = float(gray[y : y + h, x : x + w].mean())
        bright_area = sum(component[4] for component in cluster)
        center_penalty = np.linalg.norm(
            np.asarray([x + w / 2.0, y + h / 2.0]) - np.asarray([width * 0.55, height * 0.25])
        ) / max(width, height)
        score = len(cluster) * 20.0 + bright_area / 25.0 - max(0.0, crop_mean - 115.0) * 1.2 - center_penalty * 20.0
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))

    if best is None:
        return None
    return best[1]


def preprocess_timer_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def try_tesseract(binary, config):
    try:
        import pytesseract
    except Exception:
        return None, None
    text = pytesseract.image_to_string(binary, config=config)
    match = TIMER_RE.search(text.replace(" ", ""))
    if not match:
        return None, text.strip()
    return match.group(0).replace(":", "."), text.strip()


def infer_seven_segment(binary):
    # Lightweight fallback tuned for the millisecond timer's four large seven-segment digits.
    digit_patterns = {
        (1, 1, 1, 0, 1, 1, 1): "0",
        (0, 0, 1, 0, 0, 1, 0): "1",
        (1, 0, 1, 1, 1, 0, 1): "2",
        (1, 0, 1, 1, 0, 1, 1): "3",
        (0, 1, 1, 1, 0, 1, 0): "4",
        (1, 1, 0, 1, 0, 1, 1): "5",
        (1, 1, 0, 1, 1, 1, 1): "6",
        (1, 0, 1, 0, 0, 1, 0): "7",
        (1, 1, 1, 1, 1, 1, 1): "8",
        (1, 1, 1, 1, 0, 1, 1): "9",
    }

    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for index in range(1, num):
        x, y, w, h, area = stats[index]
        aspect = w / max(h, 1)
        if area > 450 and h > 45 and y > binary.shape[0] * 0.32 and aspect < 0.98:
            boxes.append([x, y, x + w, y + h, area])
    if not boxes:
        return None

    boxes.sort(key=lambda box: box[0])
    digit_boxes = []
    for box in boxes:
        if not digit_boxes or box[0] - digit_boxes[-1][2] > 2:
            digit_boxes.append(box[:])
        else:
            digit_boxes[-1][0] = min(digit_boxes[-1][0], box[0])
            digit_boxes[-1][1] = min(digit_boxes[-1][1], box[1])
            digit_boxes[-1][2] = max(digit_boxes[-1][2], box[2])
            digit_boxes[-1][3] = max(digit_boxes[-1][3], box[3])
            digit_boxes[-1][4] += box[4]

    # The indicator LED is another bright blob to the right of the display. The first four groups are the digits.
    digit_boxes = digit_boxes[:4]
    if len(digit_boxes) != 4:
        return None

    def classify(box):
        x1, y1, x2, y2, _area = box
        x_pad = 5
        y_pad = 6
        slot = binary[
            max(0, y1 - y_pad) : min(binary.shape[0], y2 + y_pad),
            max(0, x1 - x_pad) : min(binary.shape[1], x2 + x_pad),
        ]
        slot = cv2.resize(slot, (50, 80), interpolation=cv2.INTER_NEAREST)
        regions = [
            (14, 0, 36, 12),
            (0, 10, 14, 36),
            (36, 10, 50, 36),
            (14, 34, 36, 46),
            (0, 44, 14, 70),
            (36, 44, 50, 70),
            (14, 68, 36, 80),
        ]
        values = []
        for x1, y1, x2, y2 in regions:
            values.append(float(np.mean(slot[y1:y2, x1:x2] > 0)))
        on = tuple(1 if value > 0.16 else 0 for value in values)
        best = min(digit_patterns, key=lambda pattern: sum(a != b for a, b in zip(pattern, on)))
        distance = sum(a != b for a, b in zip(best, on))
        if distance > 2:
            return None
        return digit_patterns[best]

    chars = []
    for box in digit_boxes:
        digit = classify(box)
        if digit is None:
            return None
        chars.append(digit)
    return f"{chars[0]}.{''.join(chars[1:])}"


def ocr_timer(image, roi, config):
    if roi is None:
        roi = auto_timer_roi(image)
    if roi is None:
        return None, None, None, "roi-not-found"
    x, y, w, h = roi
    crop = image[y : y + h, x : x + w]
    binary = preprocess_timer_crop(crop)
    value_text, raw = try_tesseract(binary, config)
    method = "tesseract"
    if value_text is None:
        value_text = infer_seven_segment(binary)
        method = "seven-segment"
        raw = raw or "unreadable"
    if value_text is None:
        return None, roi, binary, raw or "unreadable"
    try:
        return float(value_text), roi, binary, f"{method}:{value_text}"
    except ValueError:
        return None, roi, binary, f"{method}:{value_text}"


def make_contact_sheet(records, output_path, display_height):
    tiles = []
    for record in records:
        frame = record["image"].copy()
        label = f"cam {record['camera_id']} host={record['elapsed_s']:.6f}s"
        if record["timer_s"] is None:
            label += " timer=?"
        else:
            label += f" timer={record['timer_s']:.6f}s"
        cv2.rectangle(frame, (6, 6), (min(frame.shape[1] - 6, 520), 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if record["roi"] is not None:
            x, y, w, h = record["roi"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        tiles.append(resize_to_height(frame, display_height))

    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    rows = []
    for start in range(0, len(tiles), 3):
        row = [fit_to_tile(tile, tile_width, tile_height) for tile in tiles[start : start + 3]]
        while len(row) < 3:
            row.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_path), np.vstack(rows))


def combine_scatter_images(summary_paths, output_path):
    images = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scatter_path = summary.get("scatter")
        if not scatter_path:
            continue
        image = cv2.imread(scatter_path)
        if image is not None:
            images.append(image)
    if not images:
        return False

    tile_height = max(image.shape[0] for image in images)
    tile_width = max(image.shape[1] for image in images)
    tiles = [fit_to_tile(image, tile_width, tile_height) for image in images]
    while len(tiles) < 4:
        tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:4])
    cv2.imwrite(str(output_path), np.vstack([top, bottom]))
    return True


def plot_records(records, output_path, grid_ms, x_range_ms, y_range_ms, target_time):
    valid = [record for record in records if record["timer_s"] is not None]
    if not valid:
        print("No valid timer OCR readings; scatter plot not written.", flush=True)
        return None
    if len(valid) < 2:
        print("Only one valid timer OCR reading; scatter plot not written.", flush=True)
        if output_path.exists():
            output_path.unlink()
        return {
            "valid_readings": len(valid),
            "visible_y_readings": len(valid),
            "hidden_by_y_axis": [],
            "unreadable_cameras": [record["camera_id"] for record in records if record["timer_s"] is None],
            "scatter_written": False,
            "note": "Need at least two valid timer OCR readings for timing statistics.",
        }
    timer_values = [record["timer_s"] for record in valid]
    timer_mean = float(np.mean(timer_values))

    y_center = timer_mean
    visible_y_records = valid
    if y_range_ms > 0:
        y_range_s = y_range_ms / 1000.0
        sorted_values = sorted(timer_values)
        best_values = []
        for start in sorted_values:
            values = [value for value in sorted_values if start <= value <= start + y_range_s]
            if len(values) > len(best_values):
                best_values = values
        if best_values:
            y_center = float(np.mean(best_values))
            y_min = y_center - y_range_s / 2.0
            y_max = y_center + y_range_s / 2.0
            visible_y_records = [record for record in valid if y_min <= record["timer_s"] <= y_max]

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 5))
    for index, record in enumerate(valid):
        color = cmap(index % 10)
        plt.scatter(record["elapsed_s"], record["timer_s"], color=color, label=f"cam {record['camera_id']}", s=56)
        plt.annotate(f"{(record['timer_s'] - y_center) * 1000:+.1f} ms", (record["elapsed_s"], record["timer_s"]), fontsize=8)
    plt.xlabel("Host capture timestamp, elapsed seconds")
    plt.ylabel("Timer OCR readout, raw parsed value")
    plt.title("One-frame timer sync comparison")
    grid_s = max(grid_ms / 1000.0, 1e-6)
    axis = plt.gca()
    axis.xaxis.set_major_locator(MultipleLocator(grid_s))
    axis.yaxis.set_major_locator(MultipleLocator(grid_s))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    if x_range_ms > 0:
        half_range_s = x_range_ms / 2000.0
        axis.set_xlim(target_time - half_range_s, target_time + half_range_s)
    if y_range_ms > 0:
        half_range_s = y_range_ms / 2000.0
        axis.set_ylim(y_center - half_range_s, y_center + half_range_s)
    plt.grid(True, which="major", alpha=0.35)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return {
        "mean_timer_s": timer_mean,
        "mean_abs_error_ms": float(np.mean([abs(record["timer_s"] - timer_mean) * 1000 for record in valid])),
        "max_abs_error_ms": float(np.max([abs(record["timer_s"] - timer_mean) * 1000 for record in valid])),
        "cluster_center_timer_value": y_center,
        "cluster_mean_abs_error_ms": float(
            np.mean([abs(record["timer_s"] - y_center) * 1000 for record in visible_y_records])
        )
        if visible_y_records
        else None,
        "cluster_max_abs_error_ms": float(
            np.max([abs(record["timer_s"] - y_center) * 1000 for record in visible_y_records])
        )
        if visible_y_records
        else None,
        "valid_readings": len(valid),
        "visible_y_readings": len(visible_y_records),
        "hidden_by_y_axis": [record["camera_id"] for record in valid if record not in visible_y_records],
        "unreadable_cameras": [record["camera_id"] for record in records if record["timer_s"] is None],
        "y_axis_center_s": y_center,
        "x_axis_center_s": target_time,
        "scatter_written": True,
    }


def analyze_target(session_dir, cameras, target_time, output_prefix, args):
    debug_dir = session_dir / f"{args.output_prefix}_ocr_debug"
    if output_prefix.name != args.output_prefix:
        debug_dir = output_prefix.with_name(output_prefix.name + "_ocr_debug")
    debug_dir.mkdir(exist_ok=True)

    records = []
    missing_frame_cameras = []
    for camera_id, rows in sorted(cameras.items()):
        if not rows:
            missing_frame_cameras.append(camera_id)
            continue
        row = nearest_row(rows, target_time)
        image_path = session_dir / row["filename"]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"cam {camera_id}: could not read {image_path}", flush=True)
            continue
        roi = parse_roi(args.roi, image.shape)
        timer_s, used_roi, binary, ocr_raw = ocr_timer(image, roi, args.tesseract_config)
        if binary is not None:
            cv2.imwrite(str(debug_dir / f"cam_{camera_id}_ocr.png"), binary)
        records.append(
            {
                "camera_id": camera_id,
                "elapsed_s": row["elapsed_s"],
                "elapsed_ns": row["elapsed_ns"],
                "source_frame_index": row["source_frame_index"],
                "filename": row["filename"],
                "image": image,
                "roi": used_roi,
                "timer_s": timer_s,
                "ocr_raw": ocr_raw,
            }
        )
        print(f"cam {camera_id}: host {row['elapsed_s']:.6f}s, timer {timer_s}, OCR {ocr_raw}", flush=True)

    csv_path = output_prefix.with_suffix(".csv")
    valid_timer_values = [record["timer_s"] for record in records if record["timer_s"] is not None]
    mean_timer = float(np.mean(valid_timer_values)) if valid_timer_values else None
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "camera_id",
            "host_elapsed_s",
            "host_delta_from_target_ms",
            "source_frame_index",
            "timer_s",
            "timer_error_from_mean_ms",
            "ocr_raw",
            "roi",
            "filename",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            error_ms = "" if record["timer_s"] is None or mean_timer is None else (record["timer_s"] - mean_timer) * 1000
            writer.writerow(
                {
                    "camera_id": record["camera_id"],
                    "host_elapsed_s": f"{record['elapsed_s']:.9f}",
                    "host_delta_from_target_ms": f"{(record['elapsed_s'] - target_time) * 1000:.3f}",
                    "source_frame_index": record["source_frame_index"],
                    "timer_s": "" if record["timer_s"] is None else f"{record['timer_s']:.9f}",
                    "timer_error_from_mean_ms": "" if error_ms == "" else f"{error_ms:.3f}",
                    "ocr_raw": record["ocr_raw"],
                    "roi": "" if record["roi"] is None else ",".join(str(value) for value in record["roi"]),
                    "filename": record["filename"],
                }
            )

    contact_path = output_prefix.with_name(output_prefix.name + "_frames.jpg")
    make_contact_sheet(records, contact_path, args.display_height)
    plot_path = output_prefix.with_name(output_prefix.name + "_scatter.png")
    stats = plot_records(records, plot_path, args.grid_ms, args.x_range_ms, args.y_range_ms, target_time)
    summary = {
        "session": str(session_dir),
        "target_host_elapsed_s": target_time,
        "csv": str(csv_path),
        "contact_sheet": str(contact_path),
        "scatter": str(plot_path) if stats is not None and stats.get("scatter_written") else None,
        "stats": stats,
        "missing_frame_cameras": missing_frame_cameras,
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {contact_path}", flush=True)
    if stats is not None and stats.get("scatter_written"):
        print(f"Wrote {plot_path}", flush=True)
        print(f"Mean abs timer error: {stats['mean_abs_error_ms']:.3f} ms", flush=True)
        print(f"Max abs timer error: {stats['max_abs_error_ms']:.3f} ms", flush=True)
    elif stats is not None:
        print(stats.get("note", "Scatter plot not written."), flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return summary


def main():
    args = parse_args()
    session_dir = args.session or newest_session(args.captures_dir)
    cameras = load_session_rows(session_dir)

    if args.four_500ms:
        summaries = []
        summary_paths = []
        for target_time in (0.25, 0.75, 1.25, 1.75):
            output_prefix = session_dir / f"{args.output_prefix}_{int(target_time * 1000):04d}ms"
            print(f"\nAnalyzing target host time {target_time:.3f}s", flush=True)
            summary = analyze_target(session_dir, cameras, target_time, output_prefix, args)
            summaries.append(summary)
            summary_paths.append(output_prefix.with_name(output_prefix.name + "_summary.json"))
        combined_path = session_dir / f"{args.output_prefix}_four_500ms_summary.json"
        combined_path.write_text(json.dumps({"session": str(session_dir), "analyses": summaries}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {combined_path}", flush=True)
        combined_scatter_path = session_dir / f"{args.output_prefix}_four_500ms_scatter_grid.png"
        if combine_scatter_images(summary_paths, combined_scatter_path):
            print(f"Wrote {combined_scatter_path}", flush=True)
        return

    target_time = choose_target_time(cameras, args.target_time)
    output_prefix = session_dir / args.output_prefix
    analyze_target(session_dir, cameras, target_time, output_prefix, args)


if __name__ == "__main__":
    main()
