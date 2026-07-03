import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an explicit ArUco board config from a photographed/annotated tag-id CSV."
    )
    parser.add_argument("csv_path", type=Path, help="CSV with one row label column followed by marker ID columns.")
    parser.add_argument("--output", type=Path, default=Path("composite_aruco_board_config.json"))
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--marker-mm", type=float, default=27.0)
    parser.add_argument("--separation-mm", type=float, default=7.0)
    parser.add_argument("--corner-shift", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument(
        "--origin",
        choices=("center", "corner"),
        default="center",
        help="center puts the full printed grid center at world (0,0,0); corner puts row 1/col 1 at (0,0,0).",
    )
    parser.add_argument("--board-name", default="tag_grid")
    return parser.parse_args()


def parse_marker_id(value):
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def read_id_grid(csv_path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            marker_ids = []
            for key, value in row.items():
                if key and key.lower() == "row":
                    continue
                marker_ids.append(parse_marker_id(value))
            rows.append(marker_ids)
    if not rows:
        raise ValueError(f"No marker rows found in {csv_path}")
    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([None] * (width - len(row)))
    return rows


def main():
    args = parse_args()
    id_grid = read_id_grid(args.csv_path)
    rows = len(id_grid)
    cols = max(len(row) for row in id_grid)
    marker_m = args.marker_mm / 1000.0
    separation_m = args.separation_mm / 1000.0
    width_m = cols * marker_m + (cols - 1) * separation_m
    height_m = rows * marker_m + (rows - 1) * separation_m
    origin_m = [-width_m / 2.0, -height_m / 2.0, 0.0] if args.origin == "center" else [0.0, 0.0, 0.0]

    config = {
        "units": "meters",
        "dictionary": args.dictionary,
        "note": (
            "Explicit 14x10 ArUco 6x6 board layout from tag_id_grid.csv. "
            "IDs marked with '?' in the CSV were inferred from neighboring decoded IDs."
        ),
        "camera_boards": {
            "front": [args.board_name],
            "back": [args.board_name],
            "side": [args.board_name],
            "top": [args.board_name],
        },
        "boards": {
            args.board_name: {
                "id_grid": id_grid,
                "marker_m": marker_m,
                "separation_m": separation_m,
                "corner_shift": args.corner_shift,
                "origin_m": origin_m,
                "x_axis_m": [1.0, 0.0, 0.0],
                "y_axis_m": [0.0, 1.0, 0.0],
                "metadata": {
                    "source_csv": str(args.csv_path),
                    "rows": rows,
                    "cols": cols,
                    "width_m": width_m,
                    "height_m": height_m,
                    "origin_mode": args.origin,
                    "corner_shift": args.corner_shift,
                },
            }
        },
    }

    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"{rows} rows x {cols} cols, marker {marker_m:.3f} m, separation {separation_m:.3f} m")
    print(f"Board span: {width_m*1000:.1f} mm x {height_m*1000:.1f} mm")


if __name__ == "__main__":
    main()
