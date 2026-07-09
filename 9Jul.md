# 9 July Caliscope Calibration Notes

## Goal

Set up the 10-camera Arducam/V4L2 rig in Caliscope, record intrinsic and
extrinsic calibration videos, solve calibration, and confirm that reconstruction
can run.

## Workspace

- Repo: `/home/cat/mouse-arducam`
- Caliscope workspace: `/home/cat/calibration`
- Launcher: `run_caliscope.sh`
- Active camera count: 10
- Active camera ids: `cam_0`, `cam_1`, `cam_2`, `cam_3`, `cam_4`, `cam_5`, `cam_8`, `cam_9`, `cam_10`, `cam_11`
- Missing/inactive ids: `cam_6`, `cam_7`

`cam_6` and `cam_7` were removed from the active Caliscope camera array because
the left-side camera assignment was not plugged in/visible. The active mapping
comes from `v4l2_camera_positions.json`.

## Code Changes Made

### `record_caliscope_intrinsics_v4l2.py`

- Added an `--output-dir` option.
- This allows recording directly into a reconstruction directory such as:
  `/home/cat/calibration/recordings/test_reconstruction_20260709_01`
- Kept the same recorder usable for intrinsic, extrinsic, and reconstruction-style
  multi-camera recordings.

### `ten_v4l2_camera_grid.py`

- Patched camera opening so `/dev/videoN` and resolved symlink device paths are
  converted to numeric V4L2 indices before calling OpenCV.
- This fixed OpenCV 5 device-opening issues.

### Caliscope package patch: `charuco_tracker.py`

Path:

`/home/cat/mouse-arducam/.venv/lib/python3.12/site-packages/caliscope/trackers/charuco_tracker.py`

Patched ChArUco point output shape handling:

- `ids` normalized to a flat `int32` array.
- `img_loc` normalized to an `N x 2` `float64` array.

This fixed a crash caused by OpenCV 5 returning different ChArUco array shapes.

### Caliscope package patch: `extrinsic_calibration_presenter.py`

Path:

`/home/cat/mouse-arducam/.venv/lib/python3.12/site-packages/caliscope/gui/presenters/extrinsic_calibration_presenter.py`

Reduced/capped the bundle adjustment optimization work so Caliscope does not sit
indefinitely at 50%:

- Initial optimize: `max_nfev=120`, `ftol=1e-6`, `strict=False`
- Final optimize: `max_nfev=80`, `ftol=1e-6`, `strict=False`

### `solve_caliscope_extrinsics.py`

Added a local helper script to solve extrinsics headlessly:

- Loads the current Caliscope workspace.
- Extracts ChArUco image points from the extrinsic videos.
- Bootstraps camera poses.
- Runs capped optimization.
- Filters the worst 2.5% observations per camera.
- Runs final capped optimization.
- Saves results back to Caliscope:
  `/home/cat/calibration/calibration/extrinsic/capture_volume`
- Updates:
  `/home/cat/calibration/camera_array.toml`
  `/home/cat/calibration/camera_array_aniposelib.toml`

## Camera/Recording Setup

The active 10-camera map during recording was:

- `cam_0`: front 1, `/dev/video0`
- `cam_1`: front 2, `/dev/video8`
- `cam_2`: back 1, `/dev/video12`
- `cam_3`: back 2, `/dev/video6`
- `cam_4`: right 1, `/dev/video2`
- `cam_5`: right 2, `/dev/video4`
- `cam_8`: top 1, `/dev/video16`
- `cam_9`: top 2, `/dev/video10`
- `cam_10`: top 3, `/dev/video14`
- `cam_11`: top 4, `/dev/video18`

One assigned left camera path was not plugged in:

`/dev/v4l/by-path/pci-0000:04:00.3-usb-0:1.1.1:1.0-video-index0`

## Intrinsic Calibration

Recorded fresh intrinsic videos with:

```bash
.venv/bin/python record_caliscope_intrinsics_v4l2.py \
  --mode intrinsic \
  --workspace /home/cat/calibration \
  --width 1280 \
  --height 800 \
  --fps 30 \
  --duration 90 \
  --cols 4 \
  --display-height 160 \
  --overwrite
```

Output:

`/home/cat/calibration/calibration/intrinsic`

All 10 active cameras recorded successfully with:

- 0 dropped frames
- 0 decode errors

Caliscope solved most intrinsics directly. Two cameras needed manual filtered
solves:

- `cam_8`: manually solved, RMSE about `1.04 px`, 21 frames used.
- `cam_9`: manually solved, RMSE about `0.916 px`, 15 frames used.

The solved intrinsics were written into:

- `/home/cat/calibration/camera_array.toml`
- `/home/cat/calibration/calibration/intrinsic/reports/cam_8.toml`
- `/home/cat/calibration/calibration/intrinsic/reports/cam_9.toml`

## First Extrinsic Calibration Attempt

Recorded extrinsic videos with:

```bash
.venv/bin/python record_caliscope_intrinsics_v4l2.py \
  --mode extrinsic \
  --workspace /home/cat/calibration \
  --width 1280 \
  --height 800 \
  --fps 30 \
  --duration 60 \
  --cols 4 \
  --display-height 160 \
  --overwrite
```

Output:

`/home/cat/calibration/calibration/extrinsic`

All 10 active cameras recorded with:

- 0 dropped frames
- 0 decode errors

Caliscope GUI got stuck around 50% during 3D/extrinsic optimization. A headless
lighter solve was run on a subset of sync indices:

- Full detections: 18,835 observations across 189 sync indices.
- Subset used: 4,776 observations across 48 sync indices.
- Matched observations: 3,969.
- Final RMSE: about `43.25 px`.
- Volumetric scale RMSE reported by Caliscope: about `22,594 mm`.

This was enough to make Caliscope load a capture volume and show a 3D scene, but
the calibration quality was poor and not trustworthy.

## Reconstruction Test Recording

Recorded a short test reconstruction clip:

```bash
.venv/bin/python record_caliscope_intrinsics_v4l2.py \
  --mode extrinsic \
  --workspace /home/cat/calibration \
  --output-dir /home/cat/calibration/recordings/test_reconstruction_20260709_01 \
  --width 1280 \
  --height 800 \
  --fps 30 \
  --duration 15 \
  --cols 4 \
  --display-height 160 \
  --overwrite
```

Output:

`/home/cat/calibration/recordings/test_reconstruction_20260709_01`

Files recorded:

- `cam_0.mp4`
- `cam_1.mp4`
- `cam_2.mp4`
- `cam_3.mp4`
- `cam_4.mp4`
- `cam_5.mp4`
- `cam_8.mp4`
- `cam_9.mp4`
- `cam_10.mp4`
- `cam_11.mp4`
- `extrinsic_recording_manifest.json`

Caliscope recognized the recording after restart.

## Reconstruction Test

Caliscope downloaded the RTMPose model:

`RTMPose-l Halpe26`

Model file:

`/home/cat/mouse-arducam/.local/share/caliscope/models/rtmpose_l_halpe26.onnx`

Reconstruction was run on:

`test_reconstruction_20260709_01`

Output directory:

`/home/cat/calibration/recordings/test_reconstruction_20260709_01/ONNX_rtmpose_l_halpe26`

Important output files:

- `xy_ONNX_rtmpose_l_halpe26.csv`
- `xyz_ONNX_rtmpose_l_halpe26.csv`
- `xyz_ONNX_rtmpose_l_halpe26_labelled.csv`
- `xyz_ONNX_rtmpose_l_halpe26.trc`
- `timestamps.csv`
- labelled per-camera preview videos

Result:

- Triangulated 3,492 3D points.
- Used 216 synchronized frames.
- Pipeline works, but output quality depends on the poor extrinsic calibration.

## Second Extrinsic Recording

Because the first extrinsic calibration was poor, a new extrinsic recording was
made:

```bash
.venv/bin/python record_caliscope_intrinsics_v4l2.py \
  --mode extrinsic \
  --workspace /home/cat/calibration \
  --width 1280 \
  --height 800 \
  --fps 30 \
  --duration 90 \
  --cols 4 \
  --display-height 160 \
  --overwrite
```

Output:

`/home/cat/calibration/calibration/extrinsic`

Recording result:

- All 10 cameras opened at `1280x800`.
- 0 dropped frames.
- 0 decode errors.

Frames written:

- `cam_0`: 1,268 frames
- `cam_1`: 1,324 frames
- `cam_2`: 1,413 frames
- `cam_3`: 1,400 frames
- `cam_4`: 1,257 frames
- `cam_5`: 1,323 frames
- `cam_8`: 1,301 frames
- `cam_9`: 1,368 frames
- `cam_10`: 1,377 frames
- `cam_11`: 1,374 frames

## Second Extrinsic Solve

Ran:

```bash
.venv/bin/python solve_caliscope_extrinsics.py \
  --workspace /home/cat/calibration \
  --frame-step 10 \
  --initial-nfev 120 \
  --final-nfev 80
```

Detection summary:

- 12,142 ChArUco observations.
- 139 sync indices.

Per-camera observations:

- `cam_0`: 2,320
- `cam_1`: 1,537
- `cam_2`: 345
- `cam_3`: 468
- `cam_4`: 1,387
- `cam_5`: 653
- `cam_8`: 2,919
- `cam_9`: 1,335
- `cam_10`: 527
- `cam_11`: 651

Optimization result:

- Final RMSE: `27.216 px`
- Matched observations: `9,199 / 9,199`
- Final optimizer: converged by `ftol`
- Volumetric scale RMSE: `997.48 mm` over 114 frames

Per-camera reprojection RMSE:

- `cam_0`: `19.472 px`
- `cam_1`: `36.694 px`
- `cam_2`: `32.408 px`
- `cam_3`: `64.605 px`
- `cam_4`: `20.436 px`
- `cam_5`: `22.772 px`
- `cam_8`: `20.128 px`
- `cam_9`: `24.176 px`
- `cam_10`: `31.284 px`
- `cam_11`: `30.912 px`

Saved to:

- `/home/cat/calibration/calibration/extrinsic/capture_volume`
- `/home/cat/calibration/calibration/extrinsic/CHARUCO`
- `/home/cat/calibration/camera_array.toml`
- `/home/cat/calibration/camera_array_aniposelib.toml`

Caliscope was restarted and loaded the new capture volume successfully.

## Weak Overlap Pairs

The main remaining issue is not enough simultaneous board visibility between
some camera pairs. The solve improved, but the overlap graph is still weak.

Low-overlap pairs found from `CHARUCO/image_points.csv`:

- `cam_1-cam_5`: 1 usable shared pose
- `cam_2-cam_11`: 1 usable shared pose
- `cam_3-cam_5`: 1 usable shared pose
- `cam_3-cam_10`: 2 usable shared poses
- `cam_4-cam_10`: 2 usable shared poses
- `cam_5-cam_9`: 2 usable shared poses
- `cam_8-cam_10`: 2 usable shared poses
- `cam_4-cam_9`: 3 usable shared poses
- `cam_1-cam_4`: 4 usable shared poses
- `cam_0-cam_5`: 5 usable shared poses
- `cam_2-cam_5`: 5 usable shared poses
- `cam_5-cam_10`: 5 usable shared poses
- `cam_10-cam_11`: 5 usable shared poses

Strongest links were mostly between front/top/right cameras, for example:

- `cam_0-cam_8`: 50 usable shared poses
- `cam_1-cam_9`: 44 usable shared poses
- `cam_0-cam_1`: 42 usable shared poses
- `cam_1-cam_8`: 35 usable shared poses
- `cam_8-cam_9`: 32 usable shared poses

## Third Extrinsic Recording

Another 90-second extrinsic recording was made after focusing more deliberately
on the weak back/right/top overlap regions.

Command used:

```bash
.venv/bin/python record_caliscope_intrinsics_v4l2.py \
  --mode extrinsic \
  --workspace /home/cat/calibration \
  --width 1280 \
  --height 800 \
  --fps 30 \
  --duration 90 \
  --cols 4 \
  --display-height 160 \
  --overwrite
```

Recording result:

- All 10 cameras opened at `1280x800`.
- 0 dropped frames.
- 0 decode errors.

Frames written:

- `cam_0`: 1,291 frames
- `cam_1`: 1,349 frames
- `cam_2`: 1,359 frames
- `cam_3`: 1,365 frames
- `cam_4`: 1,229 frames
- `cam_5`: 1,272 frames
- `cam_8`: 1,351 frames
- `cam_9`: 1,383 frames
- `cam_10`: 1,304 frames
- `cam_11`: 1,365 frames

## Third Extrinsic Solve

Ran:

```bash
.venv/bin/python solve_caliscope_extrinsics.py \
  --workspace /home/cat/calibration \
  --frame-step 10 \
  --initial-nfev 120 \
  --final-nfev 80
```

Detection summary:

- 14,968 ChArUco observations.
- 138 sync indices.

Per-camera observations:

- `cam_0`: 2,012
- `cam_1`: 1,060
- `cam_2`: 954
- `cam_3`: 684
- `cam_4`: 2,029
- `cam_5`: 1,794
- `cam_8`: 2,822
- `cam_9`: 1,304
- `cam_10`: 1,408
- `cam_11`: 901

Optimization result:

- Final RMSE: `32.488 px`
- Matched observations: `12,697 / 12,697`
- Final optimizer: converged by `ftol`
- Volumetric scale RMSE: `68.53 mm` over 129 frames

Per-camera reprojection RMSE:

- `cam_0`: `33.204 px`
- `cam_1`: `38.779 px`
- `cam_2`: `42.017 px`
- `cam_3`: `16.778 px`
- `cam_4`: `33.743 px`
- `cam_5`: `24.920 px`
- `cam_8`: `34.223 px`
- `cam_9`: `33.771 px`
- `cam_10`: `30.471 px`
- `cam_11`: `22.457 px`

This solve had a higher pixel RMSE than the second solve, but much better
physical scale consistency:

- Second solve: `27.216 px`, `997.48 mm` scale RMSE
- Third solve: `32.488 px`, `68.53 mm` scale RMSE

The third solve was saved to:

- `/home/cat/calibration/calibration/extrinsic/capture_volume`
- `/home/cat/calibration/calibration/extrinsic/CHARUCO`
- `/home/cat/calibration/camera_array.toml`
- `/home/cat/calibration/camera_array_aniposelib.toml`

Caliscope was restarted and loaded this new capture volume successfully. The
GUI reported:

- `All extrinsics calculated: True`
- `Point estimates available: True`
- `Volumetric scale accuracy: pooled RMSE=68.53mm, 129 frames sampled`

## Third Recording Overlap Results

The third recording improved several of the previously weak links:

- `cam_3-cam_10`: improved from 2 to 11 usable shared poses
- `cam_5-cam_10`: improved from 5 to 27 usable shared poses
- `cam_10-cam_11`: improved from 5 to 18 usable shared poses
- `cam_3-cam_11`: 21 usable shared poses
- `cam_2-cam_10`: 26 usable shared poses
- `cam_4-cam_8`: 44 usable shared poses
- `cam_0-cam_8`: 54 usable shared poses

Remaining weak links after the third recording:

- `cam_1-cam_5`: 1 usable shared pose
- `cam_2-cam_4`: 1 usable shared pose
- `cam_3-cam_5`: 1 usable shared pose
- `cam_4-cam_11`: 1 usable shared pose
- `cam_8-cam_11`: 1 usable shared pose
- `cam_9-cam_10`: 1 usable shared pose
- `cam_9-cam_11`: 1 usable shared pose
- `cam_8-cam_10`: 2 usable shared poses
- `cam_4-cam_9`: 3 usable shared poses
- `cam_4-cam_10`: 3 usable shared poses
- `cam_5-cam_11`: 3 usable shared poses

## Current Status

Caliscope is open and loads the third extrinsic calibration.

The pipeline is functional:

- Cameras record.
- Intrinsics exist for all 10 active cameras.
- Extrinsics can be solved and loaded.
- Reconstruction can run and produce `xyz`, labelled CSV, and TRC output.

The current extrinsic calibration is the third solve. It has better physical
scale consistency than the second solve, but the reprojection RMSE is still high:

- Current pixel RMSE: `32.488 px`
- Current volumetric scale RMSE: `68.53 mm`

This may behave better for reconstruction than the second solve because the
scale is much more plausible, but the 3D output still needs visual inspection.

## Recommended Next Step

Inspect the 3D view and reconstruction output using the third solve. If it still
looks geometrically unstable, redo extrinsic recording again with even more focus
on the remaining weak links:

- Move slowly.
- Pause with the board visible in two or more cameras at once.
- Prioritize weak links:
  `cam_3-cam_5`, `cam_4-cam_10`, `cam_8-cam_10`, `cam_9-cam_10`,
  `cam_9-cam_11`, `cam_4-cam_11`, and `cam_5-cam_11`.
- Make sure the board is not only visible in one camera at a time.
- Use short pauses in each overlap region so the solver gets multiple stable
  shared poses.

Once a better extrinsic recording is available, rerun:

```bash
.venv/bin/python solve_caliscope_extrinsics.py \
  --workspace /home/cat/calibration \
  --frame-step 10 \
  --initial-nfev 120 \
  --final-nfev 80
```

If the overlap graph improves and RMSE drops substantially, rerun reconstruction
on a trial recording.
