import argparse
import sys
from pathlib import Path


def default_config_path():
    return (
        Path(__file__).parent
        / "external"
        / "ArduCAM_USB_Camera_Shield"
        / "Config"
        / "USB2.0_UC-391_Rev.E"
        / "MIPI"
        / "OV9281"
        / "OV9281_MIPI_2Lane_RAW8_640x400_30fps.cfg"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Probe ArducamSDK access to OV9281 cameras.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    demo_dir = Path(__file__).parent / "external" / "ArduCAM_USB_Camera_Shield_Python_Demo"
    sys.path.insert(0, str(demo_dir.resolve()))

    import ArducamSDK
    from utils import camera_initFromFile

    if not args.config.exists():
        raise FileNotFoundError(args.config)

    print(f"Using config: {args.config}")
    for index in range(args.count):
        print(f"SDK index {index}: ", end="", flush=True)
        ok, handle, camera_cfg, color_mode = camera_initFromFile(str(args.config), index)
        if ok:
            print(f"opened {camera_cfg['u32Width']}x{camera_cfg['u32Height']} color_mode={color_mode}")
            ArducamSDK.Py_ArduCam_close(handle)
        else:
            print("not opened")


if __name__ == "__main__":
    main()
