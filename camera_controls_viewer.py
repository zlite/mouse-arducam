import argparse
import json
import threading
import time
from ctypes import POINTER, c_long
from pathlib import Path

import cv2
import numpy as np
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown, COMError
from pygrabber.dshow_graph import FilterGraph

from dshow_arducam_viewer import find_format_index


WINDOW = "ArduCam Camera Controls"
CONFIG_PATH = Path("camera_controls_config.json")


CONTROL_SPECS = {
    "brightness": ("procamp", 0),
    "contrast": ("procamp", 1),
    "saturation": ("procamp", 3),
    "sharpness": ("procamp", 4),
    "wb_temp": ("procamp", 7),
    "gain": ("procamp", 9),
}

CAMERA_CONTROL_EXPOSURE = 4
CONTROL_AUTO = 0x0001
CONTROL_MANUAL = 0x0002


class IAMVideoProcAmp(IUnknown):
    _iid_ = GUID("{C6E13360-30AC-11d0-A18C-00A0C9118956}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetRange", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "pMin"), (["out"], POINTER(c_long), "pMax"), (["out"], POINTER(c_long), "pSteppingDelta"), (["out"], POINTER(c_long), "pDefault"), (["out"], POINTER(c_long), "pCapsFlags")),
        COMMETHOD([], HRESULT, "Set", (["in"], c_long, "Property"), (["in"], c_long, "lValue"), (["in"], c_long, "Flags")),
        COMMETHOD([], HRESULT, "Get", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "lValue"), (["out"], POINTER(c_long), "Flags")),
    ]


class IAMCameraControl(IUnknown):
    _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetRange", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "pMin"), (["out"], POINTER(c_long), "pMax"), (["out"], POINTER(c_long), "pSteppingDelta"), (["out"], POINTER(c_long), "pDefault"), (["out"], POINTER(c_long), "pCapsFlags")),
        COMMETHOD([], HRESULT, "Set", (["in"], c_long, "Property"), (["in"], c_long, "lValue"), (["in"], c_long, "Flags")),
        COMMETHOD([], HRESULT, "Get", (["in"], c_long, "Property"), (["out"], POINTER(c_long), "lValue"), (["out"], POINTER(c_long), "Flags")),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Live ArduCam viewer with manual camera-control sliders.")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3], help="DirectShow camera indices.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG", help="DirectShow subtype, usually MJPG for these ArduCams.")
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--scan", action="store_true", help="List available camera indices and exit.")
    return parser.parse_args()


def scan_cameras(max_index=10):
    del max_index
    for camera_id, name in enumerate(FilterGraph().get_input_devices()):
        print(f"{camera_id}: {name}")


def load_config(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(path, cameras, values):
    payload = {
        str(camera_id): {
            name: value
            for name, value in values[camera_id].items()
            if name != "selected_camera"
        }
        for camera_id in cameras
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class DShowControlledCamera:
    def __init__(self, camera_id, args):
        self.camera_id = camera_id
        self.args = args
        self.format_index = find_format_index(camera_id, args.fourcc, args.width, args.height)
        self.graph = FilterGraph()
        self.latest_frame = None
        self.frame_count = 0
        self.started_at = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.camera_control = None
        self.procamp = None

    def start(self):
        self.graph.add_video_input_device(self.camera_id)
        source = self.graph.get_input_device()
        source.set_format(self.format_index)
        self.camera_control = self._query_interface(source.instance, IAMCameraControl)
        self.procamp = self._query_interface(source.instance, IAMVideoProcAmp)
        self.graph.add_sample_grabber(self._on_frame)
        self.graph.add_null_render()
        self.graph.prepare_preview_graph()
        self.graph.run()
        self.started_at = time.perf_counter()
        self.running = True
        self.thread = threading.Thread(target=self._request_loop, name=f"camera-controls-{self.camera_id}", daemon=True)
        self.thread.start()

    def _query_interface(self, instance, interface):
        try:
            return instance.QueryInterface(interface)
        except COMError:
            return None

    def _on_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            self.frame_count += 1

    def _request_loop(self):
        while self.running:
            self.graph.grab_frame()
            time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def average_fps(self):
        if self.started_at is None:
            return 0.0
        return self.frame_count / max(time.perf_counter() - self.started_at, 1e-6)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.graph.stop()
        finally:
            self.graph.remove_filters()

    def _get_iface_and_prop(self, name):
        if name == "exposure":
            return self.camera_control, CAMERA_CONTROL_EXPOSURE
        iface_name, prop_id = CONTROL_SPECS[name]
        if iface_name == "procamp":
            return self.procamp, prop_id
        return None, None

    def get_range(self, name):
        iface, prop_id = self._get_iface_and_prop(name)
        if iface is None:
            if name == "exposure":
                return {"min": -13, "max": 0, "step": 1, "default": -6, "caps": CONTROL_MANUAL}
            if name == "wb_temp":
                return {"min": 2000, "max": 8000, "step": 10, "default": 4000, "caps": CONTROL_MANUAL}
            return {"min": 0, "max": 255, "step": 1, "default": 0, "caps": CONTROL_MANUAL}
        try:
            minimum, maximum, step, default, caps = iface.GetRange(prop_id)
            step = max(1, int(step))
            return {"min": int(minimum), "max": int(maximum), "step": step, "default": int(default), "caps": int(caps)}
        except COMError:
            return {"min": 0, "max": 255, "step": 1, "default": 0, "caps": CONTROL_MANUAL}

    def get_value_and_flags(self, name):
        iface, prop_id = self._get_iface_and_prop(name)
        if iface is None:
            return 0, CONTROL_MANUAL
        try:
            value, flags = iface.Get(prop_id)
            return int(value), int(flags)
        except COMError:
            return self.get_range(name)["default"], CONTROL_MANUAL

    def set_value(self, name, value, auto=False):
        iface, prop_id = self._get_iface_and_prop(name)
        if iface is None:
            return
        flags = CONTROL_AUTO if auto else CONTROL_MANUAL
        try:
            iface.Set(prop_id, int(value), flags)
        except COMError as exc:
            print(f"camera {self.camera_id}: could not set {name}={value}: {exc}", flush=True)

    def read_values(self):
        exposure_value, exposure_flags = self.get_value_and_flags("exposure")
        wb_value, wb_flags = self.get_value_and_flags("wb_temp")
        values = {
            "auto_exposure": 1 if exposure_flags & CONTROL_AUTO else 0,
            "exposure": exposure_value,
            "auto_wb": 1 if wb_flags & CONTROL_AUTO else 0,
            "wb_temp": wb_value,
        }
        for name in CONTROL_SPECS:
            if name == "wb_temp":
                continue
            value, _flags = self.get_value_and_flags(name)
            values[name] = value
        return values

    def apply_control(self, name, value, values):
        if name == "auto_exposure":
            self.set_value("exposure", values.get("exposure", self.get_range("exposure")["default"]), auto=bool(value))
        elif name == "exposure":
            self.set_value("exposure", value, auto=bool(values.get("auto_exposure", 0)))
        elif name == "auto_wb":
            self.set_value("wb_temp", values.get("wb_temp", self.get_range("wb_temp")["default"]), auto=bool(value))
        elif name == "wb_temp":
            self.set_value("wb_temp", value, auto=bool(values.get("auto_wb", 0)))
        else:
            self.set_value(name, value, auto=False)


def build_ranges(camera):
    ranges = {"exposure": camera.get_range("exposure")}
    for name in CONTROL_SPECS:
        ranges[name] = camera.get_range(name)
    return ranges


def value_to_slider(name, value, ranges):
    control_range = ranges[name]
    step = max(1, control_range["step"])
    return int(round((int(value) - control_range["min"]) / step))


def slider_to_value(name, position, ranges):
    control_range = ranges[name]
    value = control_range["min"] + int(position) * max(1, control_range["step"])
    return int(clamp(value, control_range["min"], control_range["max"]))


def create_trackbars(camera_count, ranges):
    cv2.createTrackbar("camera", WINDOW, 0, max(0, camera_count - 1), lambda _value: None)
    cv2.createTrackbar("apply_all", WINDOW, 0, 1, lambda _value: None)
    cv2.createTrackbar("auto_exposure", WINDOW, 0, 1, lambda _value: None)
    exposure_range = ranges["exposure"]
    cv2.createTrackbar("exposure", WINDOW, 0, max(1, (exposure_range["max"] - exposure_range["min"]) // max(1, exposure_range["step"])), lambda _value: None)
    cv2.createTrackbar("auto_wb", WINDOW, 0, 1, lambda _value: None)
    for name in CONTROL_SPECS:
        control_range = ranges[name]
        max_position = max(1, (control_range["max"] - control_range["min"]) // max(1, control_range["step"]))
        cv2.createTrackbar(name, WINDOW, 0, max_position, lambda _value: None)


def set_trackbars_from_values(values, ranges):
    cv2.setTrackbarPos("auto_exposure", WINDOW, int(values.get("auto_exposure", 0)))
    cv2.setTrackbarPos("exposure", WINDOW, value_to_slider("exposure", values.get("exposure", ranges["exposure"]["default"]), ranges))
    cv2.setTrackbarPos("auto_wb", WINDOW, int(values.get("auto_wb", 0)))
    for name in CONTROL_SPECS:
        cv2.setTrackbarPos(name, WINDOW, value_to_slider(name, values.get(name, ranges[name]["default"]), ranges))


def read_trackbars(ranges):
    values = {
        "auto_exposure": cv2.getTrackbarPos("auto_exposure", WINDOW),
        "exposure": slider_to_value("exposure", cv2.getTrackbarPos("exposure", WINDOW), ranges),
        "auto_wb": cv2.getTrackbarPos("auto_wb", WINDOW),
    }
    for name in CONTROL_SPECS:
        values[name] = slider_to_value(name, cv2.getTrackbarPos(name, WINDOW), ranges)
    return values


def resize_to_height(frame, target_height):
    height, width = frame.shape[:2]
    if height <= target_height:
        return frame
    scale = target_height / height
    return cv2.resize(frame, (max(1, int(width * scale)), target_height), interpolation=cv2.INTER_AREA)


def rotate_frame(frame, degrees):
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_overlay(frame, camera_id, values, fps):
    frame = frame.copy()
    height, width = frame.shape[:2]
    lines = [
        f"cam {camera_id} | {width}x{height} | {fps:4.1f} FPS",
        "q/Esc quit | s save | r read camera",
        f"AE {values['auto_exposure']} EXP {values['exposure']} GAIN {values['gain']} CONTRAST {values['contrast']} WB {values['auto_wb']}/{values['wb_temp']}",
    ]
    box_height = 30 + 27 * len(lines)
    cv2.rectangle(frame, (8, 8), (min(width - 8, 780), box_height), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 38 + 27 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def main():
    args = parse_args()
    if args.scan:
        scan_cameras()
        return

    saved = load_config(args.config)
    cameras = {camera_id: DShowControlledCamera(camera_id, args) for camera_id in args.cameras}
    for camera in cameras.values():
        camera.start()
    ranges = build_ranges(cameras[args.cameras[0]])
    values_by_camera = {camera_id: camera.read_values() for camera_id, camera in cameras.items()}

    for camera_id, saved_values in saved.items():
        camera_id = int(camera_id)
        if camera_id in values_by_camera:
            values_by_camera[camera_id].update(saved_values)
            for name, value in values_by_camera[camera_id].items():
                if name in CONTROL_SPECS or name in ("auto_exposure", "exposure", "auto_wb"):
                    cameras[camera_id].apply_control(name, value, values_by_camera[camera_id])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    create_trackbars(len(args.cameras), ranges)
    selected_index = 0
    set_trackbars_from_values(values_by_camera[args.cameras[selected_index]], ranges)
    last_values = read_trackbars(ranges)

    try:
        while True:
            new_index = cv2.getTrackbarPos("camera", WINDOW)
            new_index = int(clamp(new_index, 0, len(args.cameras) - 1))
            if new_index != selected_index:
                selected_index = new_index
                set_trackbars_from_values(values_by_camera[args.cameras[selected_index]], ranges)
                last_values = read_trackbars(ranges)

            selected_camera = args.cameras[selected_index]
            current_values = read_trackbars(ranges)
            if current_values != last_values:
                target_cameras = args.cameras if cv2.getTrackbarPos("apply_all", WINDOW) else [selected_camera]
                for camera_id in target_cameras:
                    for name, value in current_values.items():
                        if values_by_camera[camera_id].get(name) != value:
                            values_by_camera[camera_id][name] = value
                            cameras[camera_id].apply_control(name, value, values_by_camera[camera_id])
                last_values = current_values.copy()

            frame = cameras[selected_camera].get_frame()
            if frame is None:
                frame = np.zeros((args.display_height, int(args.display_height * 16 / 9), 3), dtype=np.uint8)
                cv2.putText(frame, f"Camera {selected_camera} frame read failed", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            frame = rotate_frame(frame, args.rotation)
            frame = resize_to_height(frame, args.display_height)
            frame = draw_overlay(frame, selected_camera, values_by_camera[selected_camera], cameras[selected_camera].average_fps())
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                save_config(args.config, args.cameras, values_by_camera)
            if key == ord("r"):
                values_by_camera[selected_camera] = cameras[selected_camera].read_values()
                set_trackbars_from_values(values_by_camera[selected_camera], ranges)
                last_values = read_trackbars(ranges)
    finally:
        for camera in cameras.values():
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
