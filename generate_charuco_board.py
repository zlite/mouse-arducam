import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a printable ChArUco calibration board.")
    parser.add_argument("--output", type=Path, default=Path("charuco_board.png"))
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--pixels-per-mm", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm / 1000.0,
        args.marker_mm / 1000.0,
        dictionary,
    )
    size = (
        int(args.squares_x * args.square_mm * args.pixels_per_mm),
        int(args.squares_y * args.square_mm * args.pixels_per_mm),
    )
    image = board.generateImage(size, marginSize=int(8 * args.pixels_per_mm))
    cv2.imwrite(str(args.output), image)
    print(f"Wrote {args.output}")
    print(f"Board: {args.squares_x}x{args.squares_y}, square={args.square_mm} mm, marker={args.marker_mm} mm")


if __name__ == "__main__":
    main()
