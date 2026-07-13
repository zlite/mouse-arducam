"""Idempotent seed of initial data drawn from documentation.md and 9Jul.md.

Runs once on first startup (only when the relevant table is empty). Costs are left
as None (TBD) for the user to fill in-app. Re-running never duplicates rows.
"""
from sqlalchemy.orm import Session

from . import models


def seed_all(db: Session) -> None:
    _seed_team(db)
    _seed_calibration(db)
    _seed_equipment(db)
    _seed_models(db)
    _seed_tasks(db)
    db.commit()


def _seed_team(db: Session) -> None:
    if db.query(models.TeamMember).count():
        return
    db.add_all([
        models.TeamMember(name="You (Project Lead)", role="Project lead / rig owner", email=""),
        models.TeamMember(name="Unassigned", role="Placeholder for new team members", email=""),
    ])


def _seed_calibration(db: Session) -> None:
    if db.query(models.CalibrationRun).count():
        return
    db.add_all([
        # Extrinsic solve #1 (documentation.md / 9Jul.md "First Extrinsic Attempt")
        models.CalibrationRun(
            run_date="2026-07-09",
            type="extrinsic",
            label="Extrinsic solve #1 (60s recording, subset)",
            reprojection_rmse_px=43.25,
            volumetric_scale_rmse_mm=22594.0,
            matched_observations=3969,
            num_cameras=10,
            per_camera_rmse=None,
            notes="First attempt. GUI stuck ~50%; ran headless lighter solve on a subset "
                  "(4,776 obs / 48 sync indices). Poor quality, not trustworthy.",
        ),
        # Extrinsic solve #2
        models.CalibrationRun(
            run_date="2026-07-09",
            type="extrinsic",
            label="Extrinsic solve #2 (90s recording)",
            reprojection_rmse_px=27.216,
            volumetric_scale_rmse_mm=997.48,
            matched_observations=9199,
            num_cameras=10,
            per_camera_rmse={
                "cam_0": 19.472, "cam_1": 36.694, "cam_2": 32.408, "cam_3": 64.605,
                "cam_4": 20.436, "cam_5": 22.772, "cam_8": 20.128, "cam_9": 24.176,
                "cam_10": 31.284, "cam_11": 30.912,
            },
            notes="Better pixel RMSE but poor physical scale (997mm). Weak overlap graph. "
                  "frame-step 10, initial-nfev 120, final-nfev 80.",
        ),
        # Extrinsic solve #3 (current)
        models.CalibrationRun(
            run_date="2026-07-09",
            type="extrinsic",
            label="Extrinsic solve #3 (90s, focus on weak overlaps) — CURRENT",
            reprojection_rmse_px=32.488,
            volumetric_scale_rmse_mm=68.53,
            matched_observations=12697,
            num_cameras=10,
            per_camera_rmse={
                "cam_0": 33.204, "cam_1": 38.779, "cam_2": 42.017, "cam_3": 16.778,
                "cam_4": 33.743, "cam_5": 24.920, "cam_8": 34.223, "cam_9": 33.771,
                "cam_10": 30.471, "cam_11": 22.457,
            },
            notes="Higher pixel RMSE than #2 but much better physical scale (68.53mm over "
                  "129 frames). Current loaded capture volume. Still too high for reliable 3D.",
        ),
        # Intrinsic summary (the two cameras that needed manual solves)
        models.CalibrationRun(
            run_date="2026-07-09",
            type="intrinsic",
            label="Intrinsics — all 10 cameras (cam_8/cam_9 manual)",
            reprojection_rmse_px=1.04,
            volumetric_scale_rmse_mm=None,
            matched_observations=None,
            num_cameras=10,
            per_camera_rmse={"cam_8": 1.04, "cam_9": 0.916},
            notes="1280x800, 0 dropped frames, 0 decode errors. Most cameras solved directly "
                  "in Caliscope; cam_8 (1.04px, 21 frames) and cam_9 (0.916px, 15 frames) "
                  "needed manual filtered solves. Worst-camera RMSE shown.",
        ),
    ])


def _seed_equipment(db: Session) -> None:
    if db.query(models.Equipment).count():
        return
    db.add_all([
        models.Equipment(name="Arducam USB camera (active in rig)", category="Cameras",
                         quantity=10, unit_cost=None, supplier="Arducam", url="",
                         status="owned",
                         notes="10 active cameras: cam_0,1 (front), cam_2,3 (back), "
                               "cam_4,5 (right), cam_8,9,10,11 (top). 1280x800 @ 30fps."),
        models.Equipment(name="Arducam USB camera (spare / left side)", category="Cameras",
                         quantity=2, unit_cost=None, supplier="Arducam", url="",
                         status="owned",
                         notes="cam_6/cam_7 left-side positions — currently not plugged in / "
                               "inactive in Caliscope array."),
        models.Equipment(name="Mini PC (recording host)", category="Compute",
                         quantity=1, unit_cost=None, supplier="", url="", status="owned",
                         notes="Runs recording scripts + Caliscope. Ubuntu, Python 3.12, "
                               "10x USB cameras attached."),
        models.Equipment(name="USB cabling / powered USB hubs", category="Cabling",
                         quantity=1, unit_cost=None, supplier="", url="", status="owned",
                         notes="Connects 10 cameras to the mini PC. See v4l2_camera_positions.json "
                               "for the by-path USB topology."),
        models.Equipment(name="Mouse cage / arena", category="Enclosure",
                         quantity=1, unit_cost=None, supplier="", url="", status="owned",
                         notes="Arena ~0.29m wide x 0.38m long (manual_rig_geometry.json). "
                               "Cameras mounted around it."),
        models.Equipment(name="ChArUco calibration board (printed)", category="Calibration",
                         quantity=1, unit_cost=None, supplier="", url="", status="owned",
                         notes="12 cols x 6 rows, 5.4cm squares, DICT_4X4_50, inverted. "
                               "Mounted on a 3D-printed holder."),
        models.Equipment(name="Camera mounts (3D printed)", category="Mounts",
                         quantity=10, unit_cost=None, supplier="Self-printed", url="",
                         status="owned",
                         notes="Custom mounts fixing cameras to the cage. See 3D Models tab."),
    ])


def _seed_models(db: Session) -> None:
    if db.query(models.Model3D).count():
        return
    db.add_all([
        models.Model3D(name="ChArUco board holder", purpose="Rigid holder so the printed "
                       "ChArUco board can be moved steadily through the camera volume during "
                       "extrinsic calibration.", quantity=1, material="PLA", file_link="",
                       print_status="printed", printed_by="",
                       notes="Referenced in documentation.md hardware setup."),
        models.Model3D(name="Camera mount (cage-attached)", purpose="Fixes an Arducam camera "
                       "to the cage at a defined position/angle.", quantity=10, material="PLA",
                       file_link="", print_status="printed", printed_by="",
                       notes="10 mounts for the 10 active cameras. Add STL/link when available."),
        models.Model3D(name="Left-side camera mounts (planned)", purpose="Mounts to bring the "
                       "two inactive left-side cameras (cam_6/cam_7) online to close overlap gaps.",
                       quantity=2, material="PLA", file_link="", print_status="not_started",
                       printed_by="", notes="Left side currently unplugged; adding these could "
                       "improve the weak overlap graph."),
    ])


def _seed_tasks(db: Session) -> None:
    if db.query(models.Task).count():
        return
    db.add_all([
        models.Task(
            title="Re-record extrinsics focusing on weak overlap pairs",
            description="Move board slowly, pause in each overlap region with the board visible "
                        "in >=2 cameras. Prioritize weak links: cam_3-cam_5, cam_4-cam_10, "
                        "cam_8-cam_10, cam_9-cam_10, cam_9-cam_11, cam_4-cam_11, cam_5-cam_11. "
                        "Include back-to-top, right-to-top, side-to-top transitions.",
            category="Calibration", status="todo", priority="critical", order=0),
        models.Task(
            title="Re-run headless extrinsic solve & compare RMSE/scale",
            description="Run solve_caliscope_extrinsics.py (--frame-step 10 --initial-nfev 120 "
                        "--final-nfev 80). Log the result in the Calibration tab. Target: "
                        "reprojection RMSE well below 32px and scale RMSE at/under 68mm.",
            category="Calibration", status="todo", priority="high", order=1),
        models.Task(
            title="Inspect 3D view / reconstruction with solve #3",
            description="Open Caliscope 3D view on the current (third) capture volume and check "
                        "geometric plausibility before more trial recordings.",
            category="Calibration", status="todo", priority="high", order=2),
        models.Task(
            title="Bring left-side cameras (cam_6/cam_7) online",
            description="Left camera path is not plugged in/visible. Reconnect and add back to "
                        "the Caliscope array to improve overlap coverage.",
            category="Hardware", status="todo", priority="medium", order=3),
        models.Task(
            title="Fill in real equipment costs",
            description="Enter actual unit costs/suppliers/links in the Equipment tab so the "
                        "budget rollup is accurate.",
            category="Procurement", status="todo", priority="low", order=4),
        models.Task(
            title="Attach STL/source files for 3D-printed parts",
            description="Add file links for the ChArUco holder and camera mounts in the 3D Models "
                        "tab so parts can be reprinted by anyone.",
            category="3D-Printing", status="todo", priority="low", order=5),
    ])
