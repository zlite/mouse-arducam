#!/usr/bin/env bash
# Launch the Mouse-Arducam project-management web app.
# Creates an isolated venv (separate from the vision project's .venv) on first run.
set -euo pipefail

cd "$(dirname "$0")"

HOST="${PM_HOST:-0.0.0.0}"
PORT="${PM_PORT:-8000}"

# --- Create/refresh the isolated venv ---
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv >/dev/null 2>&1 || true
  uv pip install --python .venv/bin/python -r requirements.txt
else
  echo "[pm] uv not found; falling back to python -m venv + pip"
  [ -d .venv ] || python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip >/dev/null
  ./.venv/bin/pip install -r requirements.txt
fi

echo ""
echo "[pm] Starting on http://${HOST}:${PORT}"
if [ -z "${PM_PASSWORD:-}" ]; then
  echo "[pm] NOTE: PM_PASSWORD is not set — the app is unauthenticated."
  echo "[pm] To require a shared password:  PM_PASSWORD='yourpass' ./run.sh"
fi
echo ""

exec ./.venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
