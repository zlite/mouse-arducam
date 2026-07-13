"""Registry of runnable project scripts.

Only scripts listed here can be launched from the web UI (whitelist). Each entry
defines a base command (a list of tokens) that is run with REPO_ROOT as the
working directory. Users may append extra CLI args from the UI, but they can
never change which executable runs.

Paths are resolved relative to the repository root (the parent of pm/).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"

# Prefer the project virtualenv interpreter; fall back to "python3" if absent.
PY = str(VENV_PY) if VENV_PY.exists() else "python3"

WORKSPACE = "/home/cat/calibration"

SCRIPTS = [
    {
        "id": "scan_devices",
        "name": "Scan camera devices",
        "category": "Diagnostics",
        "description": "List all connected V4L2 video devices and the modes they support. "
        "Run this first to confirm all 10 cameras are detected.",
        "command": [PY, "ten_v4l2_camera_grid.py", "--scan"],
        "gui": False,
        "long_running": False,
    },
    {
        "id": "camera_grid",
        "name": "Live camera grid preview",
        "category": "Diagnostics",
        "description": "Open a live tiled preview of all cameras to check framing, focus, "
        "and that every camera is streaming. Press q to quit. (Opens a window.)",
        "command": [PY, "ten_v4l2_camera_grid.py", "--width", "1280", "--height", "800", "--fps", "30"],
        "gui": True,
        "long_running": True,
    },
    {
        "id": "motion_detector",
        "name": "Multi-camera motion detector",
        "category": "Diagnostics",
        "description": "Low-compute motion detector across all cameras. Useful to sanity-check "
        "the rig sees the cage and to log synced motion events. Press q to quit.",
        "command": [PY, "ten_v4l2_motion_detector.py"],
        "gui": True,
        "long_running": True,
    },
    {
        "id": "record_intrinsic",
        "name": "Record intrinsic calibration",
        "category": "Calibration",
        "description": "Record a 90s intrinsic calibration video for all 10 cameras into the "
        "Caliscope workspace. Move the ChArUco board across each camera's whole field of view.",
        "command": [
            PY, "record_caliscope_intrinsics_v4l2.py",
            "--mode", "intrinsic", "--workspace", WORKSPACE,
            "--width", "1280", "--height", "800", "--fps", "30",
            "--duration", "90", "--cols", "4", "--display-height", "160", "--overwrite",
        ],
        "gui": True,
        "long_running": True,
    },
    {
        "id": "record_extrinsic",
        "name": "Record extrinsic calibration",
        "category": "Calibration",
        "description": "Record a 90s extrinsic calibration video for all 10 cameras. Move the "
        "board slowly and pause in each overlap region so pairs of cameras share poses. "
        "Focus on the weak links listed in the calibration notes.",
        "command": [
            PY, "record_caliscope_intrinsics_v4l2.py",
            "--mode", "extrinsic", "--workspace", WORKSPACE,
            "--width", "1280", "--height", "800", "--fps", "30",
            "--duration", "90", "--cols", "4", "--display-height", "160", "--overwrite",
        ],
        "gui": True,
        "long_running": True,
    },
    {
        "id": "solve_extrinsic",
        "name": "Solve extrinsic calibration (headless)",
        "category": "Calibration",
        "description": "Run the headless bundle-adjustment solve on the latest extrinsic "
        "recording and write the capture volume + camera_array.toml. Reports final "
        "reprojection RMSE and volumetric scale RMSE — log these as a Calibration run.",
        "command": [
            PY, "solve_caliscope_extrinsics.py",
            "--workspace", WORKSPACE,
            "--frame-step", "10", "--initial-nfev", "120", "--final-nfev", "80",
        ],
        "gui": False,
        "long_running": True,
    },
    {
        "id": "launch_caliscope",
        "name": "Launch Caliscope GUI",
        "category": "Calibration",
        "description": "Open the full Caliscope desktop app on the calibration workspace to "
        "inspect intrinsics, the 3D capture volume, and run reconstruction.",
        "command": ["bash", "run_caliscope.sh"],
        "gui": True,
        "long_running": True,
    },
]

SCRIPTS_BY_ID = {s["id"]: s for s in SCRIPTS}


def command_string(script: dict) -> str:
    return " ".join(script["command"])
