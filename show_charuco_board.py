import argparse

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Show a ChArUco board fullscreen for screen-based calibration.")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--window", default="ChArUco Calibration Board")
    parser.add_argument("--width", type=int, default=1600, help="Generated board image width in pixels.")
    parser.add_argument("--height", type=int, default=1100, help="Generated board image height in pixels.")
    return parser.parse_args()


def main():
    args = parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    # Physical square size is supplied later to the calibration scripts.
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), 1.0, 0.73, dictionary)
    image = board.generateImage((args.width, args.height), marginSize=40)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(args.window, image)
    print("Measure one chessboard square on the screen in millimeters.")
    print("Use that value as --square-mm when running the calibration scripts.")
    print("Press q or Esc to close.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
