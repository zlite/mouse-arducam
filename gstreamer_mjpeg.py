import os
import sys
import threading
import time
from collections import deque

import ten_v4l2_camera_grid as grid


Gst = None


def require_gstreamer():
    global Gst
    if Gst is not None:
        return Gst
    # Ubuntu's legacy VA-API plugin otherwise hides the AMD encoder elements.
    os.environ.setdefault("GST_VAAPI_ALL_DRIVERS", "1")
    if "/usr/lib/python3/dist-packages" not in sys.path:
        sys.path.append("/usr/lib/python3/dist-packages")
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst as gst_module
    except (ImportError, ValueError) as exc:
        raise RuntimeError("GStreamer Python bindings are required for 30 FPS MJPEG capture") from exc
    gst_module.init(None)
    Gst = gst_module
    return Gst


class GStreamerMJPEGCamera:
    def __init__(self, device, width, height, fps, fourcc="MJPG", buffer_frames=600):
        if fourcc.upper() not in ("MJPG", "MJPEG"):
            raise ValueError("The GStreamer pass-through backend requires MJPG")
        self.device = device
        self.stable_id = grid.stable_device_id(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.buffer_frames = max(30, int(buffer_frames))
        self.pipeline = None
        self.appsink = None
        self.lock = threading.Lock()
        self.packets = deque(maxlen=self.buffer_frames)
        self.frame_times = deque(maxlen=180)
        self.frame_count = 0
        self.latest_packet = None
        self.latest_shape = (self.height, self.width, 3)
        self.decoded_index = None
        self.decoded_frame = None
        self.running = False
        self.error = None
        self.started_at = None

    def start(self):
        gst = require_gstreamer()
        fps_num = int(round(self.fps))
        description = (
            f'v4l2src device="{self.device}" do-timestamp=true '
            f'! image/jpeg,width={self.width},height={self.height},framerate={fps_num}/1 '
            f'! appsink name=sink emit-signals=true sync=false max-buffers=8 drop=true'
        )
        self.pipeline = gst.parse_launch(description)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self._on_sample)
        self.running = True
        self.started_at = time.perf_counter()
        result = self.pipeline.set_state(gst.State.PLAYING)
        if result == gst.StateChangeReturn.FAILURE:
            self.running = False
            self.pipeline.set_state(gst.State.NULL)
            raise RuntimeError(f"Could not open {self.device} with GStreamer")
        self.pipeline.get_state(5 * gst.SECOND)
        self._poll_bus()
        if self.error is not None:
            raise RuntimeError(self.error)
        print(
            f"{self.device}: GStreamer MJPEG pass-through {self.width}x{self.height}@{fps_num}",
            flush=True,
        )

    def _on_sample(self, sink):
        gst = require_gstreamer()
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        now_perf = time.perf_counter()
        now_unix = time.time()
        jpeg = buffer.extract_dup(0, buffer.get_size())
        with self.lock:
            packet = {
                "jpeg": jpeg,
                "source_frame_index": self.frame_count,
                "capture_perf_time": now_perf,
                "capture_unix_time": now_unix,
                "source_pts_ns": int(buffer.pts) if buffer.pts != gst.CLOCK_TIME_NONE else None,
            }
            self.packets.append(packet)
            self.latest_packet = packet
            self.frame_count += 1
            self.frame_times.append(now_perf)
        return gst.FlowReturn.OK

    def _poll_bus(self):
        if self.pipeline is None:
            return
        gst = require_gstreamer()
        bus = self.pipeline.get_bus()
        while True:
            message = bus.pop_filtered(gst.MessageType.ERROR | gst.MessageType.EOS)
            if message is None:
                return
            if message.type == gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self.error = f"{self.device}: GStreamer error: {error}; {debug or ''}".strip()
                return
            if message.type == gst.MessageType.EOS:
                return

    def _current_fps_locked(self):
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / max(elapsed, 1e-9)

    def current_fps(self):
        with self.lock:
            return self._current_fps_locked()

    def read_latest_packet(self, copy_frame=True):
        self._poll_bus()
        if self.error is not None:
            raise RuntimeError(self.error)
        with self.lock:
            packet = self.latest_packet
            fps = self._current_fps_locked()
            if packet is None:
                return False, None, self.latest_shape, fps, None, None, None
            source_index = packet["source_frame_index"]
            if self.decoded_index == source_index and self.decoded_frame is not None:
                frame = self.decoded_frame.copy() if copy_frame else self.decoded_frame
                return (
                    True,
                    frame,
                    self.latest_shape,
                    fps,
                    source_index,
                    packet["capture_perf_time"],
                    packet["capture_unix_time"],
                )
            jpeg = packet["jpeg"]

        encoded = grid.np.frombuffer(jpeg, dtype=grid.np.uint8)
        decoded = grid.cv2.imdecode(encoded, grid.cv2.IMREAD_COLOR)
        if decoded is None:
            return False, None, self.latest_shape, fps, source_index, None, None
        with self.lock:
            self.decoded_index = source_index
            self.decoded_frame = decoded
            self.latest_shape = decoded.shape
        frame = decoded.copy() if copy_frame else decoded
        return (
            True,
            frame,
            decoded.shape,
            fps,
            source_index,
            packet["capture_perf_time"],
            packet["capture_unix_time"],
        )

    def encoded_packets_after(self, source_frame_index):
        with self.lock:
            if not self.packets:
                return [], 0
            first_index = self.packets[0]["source_frame_index"]
            expected = first_index if source_frame_index is None else source_frame_index + 1
            missed = max(0, first_index - expected)
            packets = [
                packet
                for packet in self.packets
                if source_frame_index is None or packet["source_frame_index"] > source_frame_index
            ]
            return packets, missed

    def has_frame(self):
        with self.lock:
            return self.latest_packet is not None

    def stop(self):
        self.running = False
        elapsed = time.perf_counter() - self.started_at if self.started_at is not None else 0.0
        recent_fps = self.current_fps()
        if self.pipeline is not None:
            self.pipeline.set_state(require_gstreamer().State.NULL)
        self.pipeline = None
        self.appsink = None
        if elapsed > 0:
            print(
                f"{self.device}: captured {self.frame_count} compressed frames "
                f"({self.frame_count / elapsed:.1f} FPS average, {recent_fps:.1f} FPS recent)",
                flush=True,
            )


class MJPEGQuickTimeWriter:
    def __init__(self, path, width, height, fps):
        gst = require_gstreamer()
        self.path = path
        self.fps = int(round(fps))
        description = (
            f'appsrc name=src is-live=false format=time '
            f'caps="image/jpeg,width={int(width)},height={int(height)},framerate={self.fps}/1" '
            f'! jpegparse ! qtmux ! filesink location="{path}"'
        )
        self.pipeline = gst.parse_launch(description)
        self.appsrc = self.pipeline.get_by_name("src")
        if self.pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(gst.State.NULL)
            raise RuntimeError(f"Could not create MJPEG video {path}")
        self.written = 0

    def write(self, packet):
        gst = require_gstreamer()
        jpeg = packet["jpeg"]
        buffer = gst.Buffer.new_allocate(None, len(jpeg), None)
        buffer.fill(0, jpeg)
        buffer.pts = self.written * gst.SECOND // self.fps
        buffer.dts = buffer.pts
        buffer.duration = gst.SECOND // self.fps
        result = self.appsrc.emit("push-buffer", buffer)
        if result != gst.FlowReturn.OK:
            raise RuntimeError(f"Could not write MJPEG frame to {self.path}: {result}")
        self.written += 1

    def close(self):
        gst = require_gstreamer()
        self.appsrc.emit("end-of-stream")
        message = self.pipeline.get_bus().timed_pop_filtered(
            10 * gst.SECOND,
            gst.MessageType.EOS | gst.MessageType.ERROR,
        )
        try:
            if message is None:
                raise RuntimeError(f"Timed out finalizing {self.path}")
            if message.type == gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"Could not finalize {self.path}: {error}; {debug or ''}")
        finally:
            self.pipeline.set_state(gst.State.NULL)


class VAAPIH264MP4Writer:
    def __init__(self, path, width, height, fps):
        gst = require_gstreamer()
        self.path = path
        self.fps = int(round(fps))
        description = (
            f'appsrc name=src is-live=false block=true format=time '
            f'caps="image/jpeg,width={int(width)},height={int(height)},framerate={self.fps}/1" '
            f'! jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=NV12 '
            f'! vaapih264enc '
            f'rate-control=cqp init-qp=18 quality-level=7 keyframe-period={self.fps} '
            f'! h264parse config-interval=-1 '
            f'! video/x-h264,stream-format=avc,alignment=au '
            f'! qtmux faststart=true ! filesink location="{path}"'
        )
        self.pipeline = gst.parse_launch(description)
        self.appsrc = self.pipeline.get_by_name("src")
        if self.pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(gst.State.NULL)
            raise RuntimeError(f"Could not create hardware H.264 video {path}")
        state_result, _state, _pending = self.pipeline.get_state(5 * gst.SECOND)
        if state_result == gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(gst.State.NULL)
            raise RuntimeError(f"Could not initialize hardware H.264 video {path}")
        self.written = 0

    def write(self, packet):
        gst = require_gstreamer()
        jpeg = packet["jpeg"]
        buffer = gst.Buffer.new_allocate(None, len(jpeg), None)
        buffer.fill(0, jpeg)
        buffer.pts = self.written * gst.SECOND // self.fps
        buffer.dts = buffer.pts
        buffer.duration = gst.SECOND // self.fps
        result = self.appsrc.emit("push-buffer", buffer)
        if result != gst.FlowReturn.OK:
            raise RuntimeError(f"Could not write H.264 frame to {self.path}: {result}")
        self.written += 1

    def close(self):
        gst = require_gstreamer()
        self.appsrc.emit("end-of-stream")
        message = self.pipeline.get_bus().timed_pop_filtered(
            30 * gst.SECOND,
            gst.MessageType.EOS | gst.MessageType.ERROR,
        )
        try:
            if message is None:
                raise RuntimeError(f"Timed out finalizing {self.path}")
            if message.type == gst.MessageType.ERROR:
                error, debug = message.parse_error()
                raise RuntimeError(f"Could not finalize {self.path}: {error}; {debug or ''}")
        finally:
            self.pipeline.set_state(gst.State.NULL)


def create_aligned_video_writer(path_stem, width, height, fps):
    gst = require_gstreamer()
    required_elements = ("jpegdec", "videoconvert", "vaapih264enc", "h264parse")
    if all(gst.ElementFactory.find(name) for name in required_elements):
        mp4_path = path_stem.with_suffix(".mp4")
        try:
            return VAAPIH264MP4Writer(mp4_path, width, height, fps)
        except Exception as exc:
            print(f"hardware MP4 unavailable ({exc}); using MJPEG MOV", flush=True)
    mov_path = path_stem.with_suffix(".mov")
    return MJPEGQuickTimeWriter(mov_path, width, height, fps)
