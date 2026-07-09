# Multi-Camera Cage Tracking System Documentation

## Overview

This document summarizes the current multi-camera tracking setup, including the
hardware assembly, camera calibration workflow, reconstruction test, and current
status of the system.

The goal of this work is to build a synchronized multi-camera system around a
mouse cage so that 2D detections from multiple views can be triangulated into 3D
points using Caliscope.

## Hardware Setup

The physical rig currently consists of 10 cameras mounted around the cage.

Hardware work completed:

- Fixed 10 cameras onto custom mounts attached to the cage.
- Connected all cameras by USB to a mini PC.
- Assigned the cameras to physical cage positions such as front, back, right,
  and top views.
- Printed an ArUco/ChArUco calibration marker board.
- Attached the printed marker board to a custom 3D-printed holder so it can be
  moved through the camera volume during calibration.

The intended use of the marker holder is to move the calibration board through
overlapping camera views. This allows Caliscope to estimate the relative camera
positions and orientations.

## Camera Layout

The active camera system currently uses 10 cameras:

- `cam_0`: front 1
- `cam_1`: front 2
- `cam_2`: back 1
- `cam_3`: back 2
- `cam_4`: right 1
- `cam_5`: right 2
- `cam_8`: top 1
- `cam_9`: top 2
- `cam_10`: top 3
- `cam_11`: top 4


## Software Environment

The system is running from the repository:

`/home/cat/mouse-arducam`

The Caliscope workspace is:

`/home/cat/calibration`

The Caliscope launcher script is:

`run_caliscope.sh`

The main recording script used for calibration videos is:

`record_caliscope_intrinsics_v4l2.py`

A helper script was also added for headless extrinsic solving:

`solve_caliscope_extrinsics.py`

## Calibration Target

The calibration target is a printed ChArUco board. It is mounted on a custom
3D-printed holder so it can be moved steadily through the cage volume.

The active board configuration in Caliscope is:

- 12 columns
- 6 rows
- 5.4 cm square size
- `DICT_4X4_50` marker dictionary
- Inverted board enabled

The board is used for both intrinsic and extrinsic calibration.

## Calibration Workflow

The calibration workflow has three main stages:

1. Intrinsic calibration
2. Extrinsic calibration
3. Reconstruction test

### Intrinsic Calibration

Intrinsic calibration estimates the internal parameters of each camera, such as
focal length, optical center, and lens distortion.

A fresh intrinsic recording was made for the 10 active cameras at `1280x800`.
All cameras recorded successfully with no dropped frames and no decode errors.

Most cameras calibrated directly in Caliscope. Two cameras needed manual filtered
solves:

- `cam_8`: solved with approximately `1.04 px` RMSE.
- `cam_9`: solved with approximately `0.916 px` RMSE.

Intrinsics are now available for all 10 active cameras.

### Extrinsic Calibration

Extrinsic calibration estimates the camera positions and orientations relative to
one another. This requires the ChArUco board to be visible in overlapping views
between camera pairs.

Three extrinsic calibration passes were performed.

The first solve produced a poor result:

- Final reprojection RMSE: approximately `43 px`
- Volumetric scale RMSE: approximately `22,594 mm`

A second 90-second extrinsic recording was then made. All 10 cameras recorded
successfully with no dropped frames and no decode errors.

The second solve improved the result:

- Final reprojection RMSE: `27.216 px`
- Matched observations: `9,199 / 9,199`
- Volumetric scale RMSE: `997.48 mm`

This result is better than the first attempt, but it is still not accurate enough
for reliable 3D reconstruction.

A third 90-second extrinsic recording was then made with more deliberate focus on
the weak back/right/top overlap regions. This recording also completed
successfully across all 10 cameras with no dropped frames and no decode errors.

The third solve produced:

- Final reprojection RMSE: `32.488 px`
- Matched observations: `12,697 / 12,697`
- Volumetric scale RMSE: `68.53 mm`

Compared with the second solve, the pixel reprojection RMSE became worse
(`32.488 px` instead of `27.216 px`), but the physical scale accuracy improved
substantially (`68.53 mm` instead of `997.48 mm`). This suggests the third
recording has a more plausible 3D scale, although reprojection error is still
too high for the calibration to be considered final.

## Reconstruction Test

A short 15-second reconstruction test recording was made and processed in
Caliscope using the RTMPose-l Halpe26 model.

The reconstruction pipeline successfully produced:

- 2D keypoint CSV output
- 3D `xyz` CSV output
- labelled CSV output
- TRC output
- labelled preview videos

The test triangulated:

- 3,492 3D points
- across 216 synchronized frames

This confirms that the software pipeline can run from recording through 3D
output. However, the 3D output quality is limited by the current extrinsic
calibration accuracy.

## Current Status

Completed:

- Hardware camera mounting around the cage.
- USB connection of 10 cameras to the mini PC.
- Printed ChArUco/ArUco calibration board.
- Custom 3D-printed board holder.
- Camera mapping for the 10 active cameras.
- Intrinsic calibration for all 10 active cameras.
- Extrinsic calibration attempts.
- Caliscope reconstruction test.
- Export of 3D reconstruction files.

Current limitation:

The extrinsic calibration is improved but still not final. The most likely cause
is that the calibration board is still not visible in enough overlapping camera
pairs during the extrinsic recording.

The third recording improved several previously weak links, especially:

- `cam_3-cam_10`
- `cam_5-cam_10`
- `cam_10-cam_11`
- `cam_3-cam_11`
- `cam_2-cam_10`

Remaining weak overlap areas still involve some back, right, and top views,
especially:

- `cam_3-cam_5`
- `cam_4-cam_10`
- `cam_8-cam_10`
- `cam_9-cam_10`
- `cam_9-cam_11`
- `cam_4-cam_11`
- `cam_5-cam_11`

These camera pairs need more shared ChArUco board detections.

## Next Steps

The next recommended step is to inspect the Caliscope 3D view and reconstruction
output using the third extrinsic solve. This solve may behave better than the
second solve because the volumetric scale error is much lower.

If the 3D view or reconstruction still looks unstable, repeat the extrinsic
calibration recording with more deliberate board movement.

During the next recording:

- Move the board slowly.
- Pause in each overlap region.
- Make sure at least two cameras can see the board at the same time.
- Focus especially on the weak camera pairs listed above.
- Include back-to-top, right-to-top, and side-to-top transitions.

After recording, rerun the headless extrinsic solve and check whether:

- Reprojection RMSE decreases substantially from `32.488 px`.
- Volumetric scale RMSE remains low or improves from `68.53 mm`.
- The 3D visualization looks geometrically plausible.
- Reconstruction output is stable.

Once extrinsic quality improves, the system should be ready for more meaningful
trial recordings and 3D behavioral tracking tests.
