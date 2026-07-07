#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CALISCOPE_WORKSPACE="/home/cat/calibration"

export XDG_DATA_HOME="$SCRIPT_DIR/.local/share"
export XDG_CONFIG_HOME="$SCRIPT_DIR/.local/config"
export XDG_CACHE_HOME="$SCRIPT_DIR/.local/cache"
export QT_PLUGIN_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH/platforms"

exec "$SCRIPT_DIR/.venv/bin/python" -c '
import os
import sys
from pathlib import Path

workspace = Path(os.environ["CALISCOPE_WORKSPACE"]).resolve()

if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_API", "pyside6")

import PySide6
from PySide6.QtCore import __version__ as qt_version

PySide6.__version__ = qt_version

from caliscope import APP_SETTINGS_PATH, MODELS_DIR
from caliscope.__main__ import _seed_default_model_cards
from caliscope.logger import setup_logging
from caliscope.startup import initialize_app
from caliscope.trackers import tracker_registry

setup_logging()
settings = initialize_app()
settings["last_project_parent"] = str(workspace.parent)
recent_projects = [str(path) for path in settings.get("recent_projects", []) if str(path) != str(workspace)]
settings["recent_projects"] = recent_projects + [str(workspace)]

import rtoml

with open(APP_SETTINGS_PATH, "w") as settings_file:
    rtoml.dump(settings, settings_file)

_seed_default_model_cards(MODELS_DIR)
tracker_registry.scan_onnx_models(MODELS_DIR)

from PySide6.QtWidgets import QApplication
from caliscope.gui.gc_confinement import disable, enable
from caliscope.gui.main_widget import MainWindow

app = QApplication(sys.argv)
gc_timer = enable()
window = MainWindow()
window.launch_workspace(str(workspace))
window.show()
app.exec()
disable(gc_timer)
' "$@"
