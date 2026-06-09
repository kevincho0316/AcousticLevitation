# Pipeline Instructions — Acoustic Levitation Measurement System

---

# 1. Intrinsic Calibration

One-time per camera. Only redo if you swap a lens, change resolution, or lose the calibration file.

## Capture tips

- **Cover the full field of view.** Tilt and slide the board to every corner and edge of the frame. Distortion is highest at the edges — if you only shoot center poses, it goes uncharacterized.
- **Vary distance and angle.** Include poses where the board is close (fills 80 % of frame) and far (fills 30 %), and tilt it ~30–45° on both axes. Flat-on poses add almost no information.
- **Lock focus and exposure before capturing.** If autofocus is on, focal length changes between shots and the calibration will be inconsistent.
- **Avoid motion blur.** Use a tripod or hold very still. Even a single blurry image raises the reprojection error noticeably.
- **Good lighting, no glare.** The board squares must be clearly distinguishable — diffuse lighting, no direct reflections on the paper.
- **Check corner detection live.** The calibration script rejects images where fewer than a threshold of corners are found. Aim for >80 % of the board's corners detected in every accepted image.
- **Target RMS < 0.5 px.** Values above 1.0 px indicate a problem with the dataset (motion blur, focus drift, too-similar poses, or a poorly printed board).

## After calibrating

Check the debug output: the script overlays detected corners on the images. If corners are being detected on the wrong squares, something is wrong with the board detection parameters (dict size, marker length).

The output YAML (e.g. `calibration/cam_1_intrinsics.yaml`) contains the camera matrix `K` and distortion coefficients. These values are used by every downstream stage — don't move or rename the file without updating `config/cameras.yaml`.

---

# 2. Box Calibration

Box calibration is the most critical and fragile part of this pipeline. Handle carefully.

**Initial RMS is everything.** Get it as low as possible before running bundle adjustment. Empirically, bundle adjustment cannot recover from a bad starting point — the optimizer converges to a local minimum that looks plausible but is wrong. Aim for initial reprojection RMS < 5 px before BA runs.

## Capture with a high-resolution camera

Input quality is critical. Use the highest resolution camera available.

We used 3 webcams (1080p Logitech Brio 100) to detect the levitating ball. That resolution was not enough for box calibration — it led to high initial RMS. We used a **phone camera** instead. This approach works for most people.

### Phone camera setup

Achieved intrinsics from our phone calibration:

```yaml
camera_matrix:
- - 3088.9321918709943
  - 0.0
  - 2019.0811998544625
- - 0.0
  - 3109.0744926332295
  - 1501.0563994606648
- - 0.0
  - 0.0
  - 1.0
distortion_coefficients:
- 0.26729805613843555
- -1.3473021544979038
- 0.00028953053254239736
- 0.0018386094646656987
- 2.178494251727113
```

**Camera matrix** — focal length and principal point. You do not need to manually set focus for this; it describes the lens geometry.

**Distortion coefficients** — lens barrel/pincushion distortion. The first two values (radial k1, k2) should dominate; large values here are normal for phone cameras.

### Phone-specific gotchas

- Many phone cameras have auto-rotation enabled — if the phone tilts, the image rotates too. This corrupts the calibration. Use a third-party camera app that locks orientation, or tape the phone to a rigid mount.
- Fix the phone position and move the box by hand when capturing. Moving the camera while the box is fixed worked significantly worse in our experience — the reprojection error was higher.

## Capture procedure

1. Set up the phone on a tripod or fixed mount, aimed at the box.
2. Capture 30–50 images from different angles: front, sides, diagonals, above.
3. **Keep markers crisp and visible.** Each image needs at least 3 markers visible (configured via `--min-markers`).
4. Every marker should fill a meaningful portion of the frame — too small and corner detection becomes noisy.
5. Include some images where the top markers are visible (view from slightly above).

## After running box calibration

Always check the debug images (`captures/boxConfig/debug/`). Each image shows the detected marker corners overlaid. Look for:
- Corners that snap accurately to the printed marker corners (not offset by a pixel or more)
- No missing markers that should have been detected
- No spurious detections on the box surface

The calibrated output writes to `config/box.config.yaml`. This file is used by every downstream stage. Keep a backup before re-running calibration.

---

# 3. Capture Session

## Before capturing

1. Power on the acoustic levitation rig and initialize it first. Wait for the trap to stabilize before capturing — a ball that is still settling will produce inconsistent detection.
2. Confirm all 3 webcams are connected and recognized: run `python -m capture.capture --list-cameras` to see device indices.
3. Check `config/cameras.yaml` device indices match the actual cameras. If you unplugged and re-plugged cameras, the indices may have shifted. Use `v4l2-ctl --list-devices` to verify.

## Capture procedure

1. Run Tab 3 · Capture in the GUI (or `python -m capture.capture --config config/cameras.yaml --output <session_dir>`).
2. Capture **50–60 frames per camera** with the ball levitating. More frames = better sub-pixel averaging, diminishing returns past ~200.
3. **Top markers must always be visible** in at least one camera view — the extrinsic solver needs them to determine the full 6-DOF pose. At least **3 markers total must be visible** per camera image.
4. After the ball-capture frames are done, **remove the ball** from the trap (turn off the levitation or physically remove it).
5. Capture **1 frame without the ball** per camera. This overwrites `frame_00.png` in each camera's folder.
6. Rename every `frame_00.png` to `background.png` in each camera subdirectory. This background image is used for ball detection via background subtraction.

   ```
   <session>/cam_1/background.png
   <session>/cam_2/background.png
   <session>/cam_3/background.png
   ```

---

# 4. Extrinsic Solve

Run Tab 5 · Extrinsic in the GUI (or `python -m extrinsic_solver.solve`).

This computes each camera's pose (position + orientation) in the box coordinate frame using the ArUco markers.

## If it fails with "not enough markers visible"

This usually means the webcams have too much reflection from the reflective panels or stands — marker detection fails on those frames.

**Workaround:**
1. Create a separate temporary session directory (e.g. `sessions/tmp_extrinsic/`).
2. Remove all reflective panels and stands from around the box.
3. Capture 1 frame per camera in this clean configuration.
4. Run the extrinsic solver on this temporary session — it will produce `extrinsics.json`.
5. Copy `extrinsics.json` into your main session directory.

---

# 5. Ball Detection

Run Tab 6 · Detect in the GUI.

**Select the second detection option** — this uses background subtraction with the `background.png` you captured earlier. It isolates the ball from background clutter before thresholding.

## Interactive click procedure

When you launch detection with the interactive flag, a window pops up for each camera showing candidate blobs. Click once on the blob that corresponds to the ball. The system uses your click to seed the ROI for subsequent frames.

- One click per camera is all that is required.
- If no blob is visible near the ball location, check that `background.png` exists and is from the same camera position.

---

# 6. Triangulation

Run Tab 7 · Triangulate in the GUI (or `python -m triangulation.triangulate`).

Takes the per-camera 2D ball positions and reconstructs the 3D position in the box coordinate frame. Outputs `<session>/triangulation.json`.

Check the reprojection residuals in the log — they should be < 1 px. If they're large, the extrinsic calibration or ball detection has an issue.

---

# 7. Error Propagation

Run Tab 8 · Error Prop in the GUI (or `python -m error_propagation.propagate`).

Quantifies the total 3D position uncertainty broken down by source (intrinsic residuals, marker position uncertainty, ArUco corner noise, extrinsic pose residual, ball detection noise). Outputs `<session>/error_budget.json`.

This step is optional for getting a 3D position, but required for the comparison step to report a meaningful Mahalanobis distance.

---

# 8. Comparison (skip for now)

Tab 9 · Compare is for comparing the measured ball position against the acoustic simulator's predicted trap location. Skip this step unless you have simulator output (`sim.py` / `summary.json`) ready.

---

# 9. 3D Scene Viewer

Run Tab 10 · Scene 3D in the GUI (or standalone: `python -m visualization.scene_3d --session <session_dir> --box-config config/box.config.yaml`).

Click **Refresh** after running any upstream stage — the plot reloads from disk and will show whichever data files exist:
- Box wireframe (always, from `box.config.yaml`)
- Colored marker quads (front=red, back=blue, left=green, right=orange, top=purple)
- Camera positions with view frustums (from `extrinsics.json`)
- Triangulated ball position as a yellow sphere (from `triangulation.json`)

## Navigating the 3D view

The scene renders in a matplotlib 3D axes embedded in the GUI.

| Action | Control |
|--------|---------|
| Rotate view | Left-click and drag |
| Zoom in / out | Scroll wheel |
| Pan | Middle-click and drag |
| Reset view | Click the home icon in the matplotlib toolbar |

Default view is from front-right-above (elevation 25°, azimuth 45°), which shows the front face, right face, and top of the box simultaneously.

**Box coordinate frame:**
- X → right (width)
- Y → up (height)
- Z → depth (front face at Z=0, back face at Z=depth)

If cameras and ball appear in reasonable positions relative to the box, the calibration chain is working correctly.
