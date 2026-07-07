#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export XDG_DATA_HOME="$SCRIPT_DIR/.local/share"
export XDG_CONFIG_HOME="$SCRIPT_DIR/.local/config"
export XDG_CACHE_HOME="$SCRIPT_DIR/.local/cache"
export QT_PLUGIN_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH/platforms"

exec "$SCRIPT_DIR/.venv/bin/caliscope" "$@"
