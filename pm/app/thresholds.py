"""Compute pass/warn/fail status for a calibration run against editable thresholds.

Thresholds live in the Setting table so they can be tuned in-app. Defaults chosen
so the currently-achieved calibration (~32px / ~68mm) reads as 'fail' — i.e. the
goal is to drive both metrics well below these lines.
"""
from sqlalchemy.orm import Session

from . import models

DEFAULTS = {
    "reproj_pass_px": 5.0,
    "reproj_warn_px": 15.0,
    "scale_pass_mm": 20.0,
    "scale_warn_mm": 100.0,
    "project_title": "Mouse-Arducam 3D Tracking Rig",
}


def get_thresholds(db: Session) -> dict:
    rows = {s.key: s.value for s in db.query(models.Setting).all()}
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in rows and rows[key] != "":
            if key == "project_title":
                out[key] = rows[key]
            else:
                try:
                    out[key] = float(rows[key])
                except ValueError:
                    pass
    return out


def compute_status(run: "models.CalibrationRun", th: dict) -> str:
    """Worst of the two metrics wins. Intrinsic runs only use reprojection RMSE."""
    reproj = run.reprojection_rmse_px
    scale = run.volumetric_scale_rmse_mm

    levels = []  # 0=pass 1=warn 2=fail
    if reproj is not None:
        if reproj <= th["reproj_pass_px"]:
            levels.append(0)
        elif reproj <= th["reproj_warn_px"]:
            levels.append(1)
        else:
            levels.append(2)
    if scale is not None:
        if scale <= th["scale_pass_mm"]:
            levels.append(0)
        elif scale <= th["scale_warn_mm"]:
            levels.append(1)
        else:
            levels.append(2)

    if not levels:
        return "unknown"
    worst = max(levels)
    return {0: "pass", 1: "warn", 2: "fail"}[worst]
