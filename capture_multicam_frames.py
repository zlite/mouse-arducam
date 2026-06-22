import argparse
import time
from pathlib import Path

import cv2

from dshow_arducam_viewer import DShowCamera, find_format_index, list_devices


def parse_args():
    parser = argparse.ArgumentParser(description="Capture synchronized-ish frames from DirectShow cameras.")
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="DirectShow camera indices. Defaults to devices whose name contains --device-name.",
    )
    parser.add_argument(
        "--device-name",
        default="Arducam",
        help="Device name substring used when --cameras is omitted.",
    )
    parser.add_argument("--scan", action="store_true", help="List DirectShow devices and exit.")
    parser.add_argument("--format", default="MJPG")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--out-dir", type=Path, default=Path("calibration_frames"))
    parser.add_argument("--count", type=int, default=20, help="Number of frame sets to save.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frame sets.")
    parser.add_argument("--warmup", type=float, default=1.0)
    return parser.parse_args()


def default_camera_indices(device_name):
    devices = list_devices()
    matches = [index for index, name in enumerate(devices) if device_name.lower() in name.lower()]
    if not matches:
        raise RuntimeError(f"No DirectShow devices matched --device-name {device_name!r}.")
    return matches


def main():
    args = parse_args()
    if args.scan:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return
    if args.cameras is None:
        args.cameras = default_camera_indices(args.device_name)
        print(f"Using {args.device_name!r} devices: {' '.join(str(index) for index in args.cameras)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    try:
        for device_index in args.cameras:
            format_index = find_format_index(device_index, args.format, args.width, args.height)
            camera = DShowCamera(device_index, format_index)
            camera.start()
            cameras.append(camera)
            print(f"Started camera {device_index}: {args.format.upper()} {args.width}x{args.height}")

        end_warmup = time.perf_counter() + args.warmup
        while time.perf_counter() < end_warmup:
            time.sleep(0.01)

        for frame_set in range(args.count):
            deadline = time.perf_counter() + args.interval
            while time.perf_counter() < deadline:
                time.sleep(0.01)

            stamp = f"{frame_set:04d}"
            for camera in cameras:
                if camera.latest_frame is None:
                    print(f"Camera {camera.device_index}: no frame for set {stamp}")
                    continue
                path = args.out_dir / f"set_{stamp}_cam_{camera.device_index}.png"
                cv2.imwrite(str(path), camera.latest_frame)
            print(f"Saved frame set {stamp}")
    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
