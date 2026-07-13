"""Serve project documentation as raw markdown for in-app rendering.

Exposes a curated set of markdown files: the app's own onboarding guide plus the
existing project notes at the repo root. Paths are whitelisted (no arbitrary reads).
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/docs", tags=["docs"])

PM_DIR = Path(__file__).resolve().parent.parent.parent  # .../pm
REPO_ROOT = PM_DIR.parent  # .../mouse-arducam

# key -> (title, absolute path)
DOC_FILES = {
    "onboarding": ("Team Onboarding & Handoff", PM_DIR / "docs" / "onboarding.md"),
    "system": ("System Documentation", REPO_ROOT / "documentation.md"),
    "calib_log_9jul": ("Calibration Log — 9 Jul", REPO_ROOT / "9Jul.md"),
}


@router.get("")
def list_docs():
    out = []
    for key, (title, path) in DOC_FILES.items():
        out.append({"key": key, "title": title, "available": path.exists()})
    return out


@router.get("/{key}")
def get_doc(key: str):
    if key not in DOC_FILES:
        raise HTTPException(404, "Unknown document")
    title, path = DOC_FILES[key]
    if not path.exists():
        raise HTTPException(404, f"Document file missing: {path.name}")
    return {"key": key, "title": title, "markdown": path.read_text(encoding="utf-8")}
