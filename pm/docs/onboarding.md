# Team Onboarding & Handoff

Welcome to the **Mouse-Arducam 3D Tracking Rig** project. This page is the single
starting point for anyone new. Read it top to bottom once, then use the tabs in this
app day-to-day.

---

## 1. What this project is (in one paragraph)

We built a **synchronized multi-camera system around a mouse cage**. Ten USB cameras
watch the arena from the front, back, right side, and top. Each camera produces a 2D
video. Software called **Caliscope** takes the 2D views from all cameras, figures out
exactly where each camera sits in space (**calibration**), and then combines the views
to reconstruct the animal's pose as **3D points over time**. The end goal is reliable
3D behavioral tracking of a mouse in the cage.

The whole pipeline already works end-to-end (record → calibrate → reconstruct 3D). The
**one thing still blocking good results is calibration accuracy** — see section 5.

---

## 2. The hardware (the rig)

- **10 active Arducam USB cameras**, fixed to custom 3D-printed mounts on the cage.
- **1 mini PC** (Ubuntu, Python 3.12) that all cameras plug into over USB. All recording
  and calibration runs on this machine.
- **A mouse cage / arena**, roughly **0.29 m wide × 0.38 m long**.
- **A printed ChArUco calibration board** on a 3D-printed holder, waved through the
  cameras' shared field of view to calibrate.

### Camera layout

| Camera | Position | Camera | Position |
|--------|----------|--------|----------|
| cam_0  | front 1  | cam_5  | right 2  |
| cam_1  | front 2  | cam_8  | top 1    |
| cam_2  | back 1   | cam_9  | top 2    |
| cam_3  | back 2   | cam_10 | top 3    |
| cam_4  | right 1  | cam_11 | top 4    |

> `cam_6` and `cam_7` are the **left-side** cameras. They are currently **not plugged
> in / inactive**. Bringing them online is a known task (it would improve calibration
> coverage). The live USB-to-position mapping lives in `v4l2_camera_positions.json`.

---

## 3. The software (where things live)

| Thing | Location |
|-------|----------|
| Code repo | `/home/cat/mouse-arducam` |
| Caliscope workspace | `/home/cat/calibration` |
| Caliscope launcher | `run_caliscope.sh` |
| Main recording script | `record_caliscope_intrinsics_v4l2.py` |
| Headless extrinsic solver | `solve_caliscope_extrinsics.py` |
| This PM app | `/home/cat/mouse-arducam/pm` |

The full technical narrative is in the **System Documentation** and **Calibration Log**
tabs (these render `documentation.md` and `9Jul.md` from the repo).

---

## 4. Key terms (glossary)

- **Intrinsic calibration** — measuring *one* camera's internal properties: focal length,
  optical center, lens distortion. Done once per camera. Ours are all solved.
- **Extrinsic calibration** — measuring *where each camera is* relative to the others
  (position + orientation). This is the hard part and the current bottleneck.
- **ChArUco board** — a checkerboard combined with ArUco markers. Waving it through the
  cameras gives the solver matched points it can use to place the cameras. Ours is
  12 cols × 6 rows, 5.4 cm squares, `DICT_4X4_50`, inverted.
- **Reprojection RMSE (px)** — after solving, take the known 3D board points, project them
  back into each camera, and measure the pixel error vs. what was actually detected.
  **Lower = better.** This is the headline calibration-quality number.
- **Volumetric scale RMSE (mm)** — how consistent the *real-world scale* of the solved
  scene is, in millimeters. **Lower = better.**
- **Overlap / shared poses** — two cameras can only be linked if they both see the board
  at the same moment. Weak links = not enough shared views = bad calibration.
- **Reconstruction** — running a pose model (RTMPose) on a recording and triangulating
  2D detections into 3D. Only as good as the extrinsic calibration underneath it.

---

## 5. The current bottleneck (read this before doing calibration work)

Extrinsic calibration is **improved but not good enough**. Three solves so far:

| Solve | Reprojection RMSE | Volumetric scale RMSE | Verdict |
|-------|-------------------|------------------------|---------|
| #1    | 43.25 px          | 22,594 mm              | Poor |
| #2    | 27.22 px          | 997.48 mm              | Better pixels, bad scale |
| #3 (current) | 32.49 px   | 68.53 mm               | Best scale, pixels still high |

The **Calibration** tab charts these over time so you can immediately see whether a new
solve is drifting up (worse) or down (better) — and whether it crosses the pass/warn
threshold lines.

**Root cause:** the ChArUco board isn't seen by *enough camera pairs at the same time*,
especially the back/right/top links. The fix is a better extrinsic recording, not new code.

### How to do a better extrinsic pass
1. Open the **Run Scripts** tab → run **Record extrinsic calibration video**.
2. While recording: move the board **slowly**, and **pause** in each region where two or
   more cameras can see it. Prioritize the weak links (listed in the Calibration Log tab).
3. Then run **Solve extrinsics (headless)** from the same tab.
4. When it finishes, open the **Calibration** tab and **Log run** with the resulting
   reprojection RMSE and volumetric scale RMSE (the solver prints both).
5. Check the drift charts. Goal: reprojection RMSE well below 32 px while keeping scale low.

---

## 6. How to use this PM app

- **Dashboard** — one-glance health: latest calibration status, open tasks, cost, prints due.
- **Tasks** — kanban board. Create/assign work, set priority & due date. Drag by editing status.
- **Calibration** — the heart of the tool: log every calibration run, watch drift, set
  pass/warn thresholds. Click **per-cam** on a run to see which cameras are worst.
- **Run Scripts** — launch the real recording/solve scripts on the mini PC and watch their
  live log. *(Recording scripts open the cameras; only run when you're aware of the rig.)*
- **Equipment & Cost** — the bill of materials with a live cost rollup. Fill in real prices.
- **3D Models** — every part that needs printing, its status, material, and file link.
- **Team** — people you can assign tasks to.

---

## 7. First-day checklist for a new team member

1. Read this page + skim the **System Documentation** tab.
2. Add yourself in the **Team** tab.
3. Look at the **Calibration** drift charts — understand where we are.
4. Pick up a task from the **Tasks** board (the critical one is re-recording extrinsics).
5. Ask the project lead for access to the mini PC and the Caliscope workspace.
