# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Acoustic Levitation Measurement System

## Commands

```bash
pip install -r requirements.txt          # deps (opencv-contrib-python — must be contrib)

python gui.py                            # Tkinter GUI — every stage as a clickable tab
python run_pipeline.py --session <dir> --sim-output <path>   # all 5 stages, one command

# Individual stages — each module is a -m entrypoint, run from repo root:
python -m intrinsic_calibration.calibrate --camera-id <id> --images-dir <dir> ...
python -m box_calibration.calibrate --images-dir <dir> --intrinsics <yaml> --box-config config/box.yaml --output config/box.yaml
python -m capture.capture --list-cameras                     # discover device indices
python -m capture.capture --config config/cameras.yaml --output <session> --n-frames 200
python -m extrinsic_solver.solve --session <dir> --box-config config/box.yaml --cameras-config config/cameras.yaml --calibration-dir calibration
python -m ball_detector.detect --session <dir> --cameras-config config/cameras.yaml --calibration-dir calibration
python -m triangulation.triangulate --session <dir> --cameras-config config/cameras.yaml --calibration-dir calibration
python -m error_propagation.propagate --session <dir> --box-config config/box.yaml --cameras-config config/cameras.yaml --calibration-dir calibration
python -m comparison.compare --session <dir> --sim-output <path> --box-config config/box.yaml
```

Tests (synthetic, no hardware needed):

```bash
pytest synthetic_tests/tests/                       # all
pytest synthetic_tests/tests/test_triangulation.py  # one file
pytest synthetic_tests/tests/test_triangulation.py::test_name   # one test
```

There is no `pytest.ini`/`setup.py`. `synthetic_tests/tests/conftest.py` inserts the repo
root onto `sys.path`, so run `pytest` from the repo root. CLI modules also self-insert the
root via `sys.path.insert`, but `-m` from the root is the reliable invocation.

**Windows**: shell is PowerShell. `python -m capture.capture --list-cameras` uses
`cv2.CAP_DSHOW`; the Linux sysfs camera-name probes in `gui.py` silently fall back.

## Codebase Layout (as implemented)

The design spec below predates the code; actual layout differs. Each module is a flat
file, not a subpackage of files:

```
intrinsic_calibration/calibrate.py   box_calibration/        (calibrate.py + bundle.py,
capture/capture.py                     faces.py, init_*.py, box_fit.py, io_results.py —
extrinsic_solver/solve.py              full self-calibration via bundle adjustment)
ball_detector/detect.py              visualization/scene_3d.py  (3D scene for GUI tab)
triangulation/triangulate.py         synthetic_tests/synth/   (synthetic scene renderer)
error_propagation/propagate.py       synthetic_tests/tests/   (the actual test suite)
comparison/compare.py                gui.py, run_pipeline.py  (orchestrators, repo root)
common/                              sim.py                   (pre-existing trap simulator)
```

`debug_*.py` files in several modules are throwaway diagnostics, not pipeline stages.

### Pipeline data flow

`run_pipeline.py` calls each stage's `*_session()` function in-process (not via subprocess;
`gui.py` instead spawns `-m` subprocesses). Stages communicate through JSON files written
into the session directory — each stage reads the previous stage's output:

```
capture        → <session>/<cam_id>/frame_NNNN.png  + metadata
extrinsic_solver.solve_session       → <session>/extrinsics.json
ball_detector.detect_session         → <session>/ball_detections.json
triangulation.triangulate_session    → <session>/triangulation.json
error_propagation.propagate_session  → <session>/error_budget.json
comparison.compare_session           → <session>/comparison/
```

A stage will fail if its input JSON is absent — run stages in order, or use the full pipeline.

### Shared contracts (`common/`)

- `common/__init__.py` — every cross-stage dataclass: `CameraIntrinsics`, `CameraPose`,
  `BallDetection2D`, `TriangulationResult`, `ErrorSource`/`ErrorBudget`, `ComparisonResult`.
  Changing a field here ripples through every stage and the JSON files.
- `common/io_utils.py` — all YAML/JSON IO: `load_box_config`, `load_cameras_config`,
  `load_intrinsics`, `load_box_to_sim_transform`, numpy-aware JSON encoder.
- `common/se3_utils.py` — SE(3) Lie algebra (`_se3_log`, `_se3_exp`, `_average_se3`). Camera
  poses are averaged in the Lie algebra, never as naive matrix means.

### Config coupling

- `config/cameras.yaml` `serial_to_index` maps camera serial → OpenCV device index; the
  committed values are placeholders (all `2`) and must be set for real captures.
- `config/box.yaml` is both input and output of `box_calibration.calibrate` — it rewrites
  `corners_box_frame` (and adds `pose_*`, `reprojection_rms_px`) in place. It also holds
  `box_to_sim` (4×4), the transform `comparison` needs to put both points in one frame.
- `comparison` reads `newton_x/y/z` from the simulator's `summary.json` or
  `final_candidates_*.csv` produced by `sim.py`.

## Project Overview

This project builds a multi-camera measurement system to validate an acoustic levitation simulator against real-world experiments. A small white styrofoam ball is suspended in a static acoustic trap inside a black cuboid box. Multiple USB webcams observe the ball from different angles, and the system reconstructs the ball's 3D position to compare against the simulator's predicted trap location.

**Key constraint:** the ball is stationary (static trap). This drastically simplifies the system — no synchronization is required between cameras.

## System Architecture

### Physical Setup

- **Cuboid box** (precisely manufactured, e.g., 3D printed or CNC machined)
  - All surfaces matte black to maximize ball/marker contrast
  - 4 ArUco markers on the side faces (one per side)
  - Up to 4 ArUco markers on the top face
  - All marker positions in the box coordinate frame must be known with sub-millimeter accuracy
- **White styrofoam ball** suspended in the acoustic trap inside the box
- **Multiple USB webcams** on rigid tripods, positioned with at least 90° angular spread for good triangulation geometry
- **Diffuse, uniform lighting** to avoid specular highlights on the ball
- **Black background** outside the box for additional contrast

### Software Components

```
├── intrinsic_calibration/   # Per-camera lens calibration (one-time)
├── capture/                  # Multi-camera image capture
├── extrinsic_solver/         # Per-capture camera pose estimation via ArUco
├── ball_detector/            # Sub-pixel ball center detection
├── triangulation/            # 3D position reconstruction
├── error_propagation/        # Uncertainty quantification
├── comparison/               # Measured vs simulated trap position
└── common/                   # Shared utilities, data classes, IO
```

## Critical Design Decisions

### 1. No Synchronization Required

Because the ball is static, cameras do **not** need to capture simultaneously. Each camera captures independently. This eliminates the need for hardware triggers, LED sync pulses, or global shutter cameras.

**Implication:** any consumer USB webcam works. Rolling shutter is acceptable.

### 2. Temporal Averaging for Sub-pixel Precision

Even a stationary ball has micro-oscillations and the cameras have noise. The capture pipeline takes N frames per camera (typically 100–1000) and averages the detected ball center. This pushes per-camera 2D precision to roughly 1/10–1/100 of a pixel.

The standard deviation across frames is recorded and propagated as the per-camera measurement uncertainty.

### 3. Multi-Marker Board for Box Pose

All ArUco markers on the box are treated as a **single rigid body** using OpenCV's `aruco::Board` API. The box-frame coordinates of every marker are defined in a configuration file. Pose estimation uses every visible marker's corners simultaneously, which:

- Eliminates the planar pose ambiguity that affects single-marker pose estimation
- Allows extrinsic calibration even when some markers are occluded
- Improves rotational accuracy substantially because corner points span multiple non-coplanar surfaces

### 4. Two-Stage Camera Calibration

- **Intrinsic calibration** (one-time per camera): focal length, principal point, lens distortion. Done with a checkerboard or ChArUco target. Stored on disk per camera serial.
- **Extrinsic calibration** (every capture): camera pose in the box coordinate frame. Solved at runtime from the visible ArUco board.

### 5. Self-Calibration Option for Marker Positions

If physically measuring marker positions on the box to sub-millimeter accuracy is impractical, a self-calibration utility can refine the marker layout via bundle adjustment using multi-view captures of the box. The refined layout is then used as ground truth for all subsequent experiments.

## Module Specifications

### intrinsic_calibration/

**Purpose:** generate distortion profile per webcam.

**Inputs:** ChArUco board image set (~30–50 images per camera, varied poses).

**Outputs:** YAML/JSON file per camera containing:
- Camera matrix `K` (3×3)
- Distortion coefficients (5 or 8 parameters depending on model)
- Reprojection error statistics
- Image resolution

**Implementation notes:**
- Use `cv2.aruco.CharucoBoard_create` and `cv2.aruco.calibrateCameraCharuco`
- Reject images with high per-image reprojection error
- Support both pinhole + radial-tangential and rational/thin-prism models
- Each camera identified by USB serial or user-assigned ID

### capture/

**Purpose:** acquire N frames per camera from each connected webcam.

**Inputs:** number of cameras, frames per camera, exposure/focus settings.

**Outputs:** raw frames saved to disk, organized by camera ID and capture session.

**Implementation notes:**
- Disable autofocus and auto-exposure (lock these via `cv2.VideoCapture.set` properties)
- Tape over the autofocus mechanism if necessary; manual focus drift ruins intrinsic calibration
- Capture to a session folder with metadata (timestamp, camera settings, camera IDs)
- No synchronization logic required
- Verify each camera's resolution matches its intrinsic calibration

### extrinsic_solver/

**Purpose:** estimate each camera's pose in the box coordinate frame.

**Inputs:** captured frames, intrinsic parameters per camera, box marker layout config.

**Outputs:** 4×4 transformation matrix per camera (box → camera frame), with covariance.

**Implementation notes:**
- Undistort frames first using stored intrinsics
- Detect ArUco markers (`cv2.aruco.detectMarkers`)
- Use `cv2.aruco.estimatePoseBoard` with all detected markers from the box board
- Reject frames with too few visible markers (require ≥ 3 markers, ideally from ≥ 2 box faces)
- Compute reprojection error per camera; flag outliers
- Average pose across frames per camera (use Lie algebra averaging for SE(3), not naive matrix mean)

### ball_detector/

**Purpose:** locate the white ball center in each undistorted frame with sub-pixel precision.

**Inputs:** undistorted frames per camera.

**Outputs:** per-frame 2D ball center `(u, v)` plus per-camera averaged `(u̅, v̅)` and covariance `Σ_2D`.

**Implementation notes:**
- The setup is high-contrast (white ball, everything else black) so detection is straightforward
- Pipeline: threshold → connected components → keep largest blob → centroid + ellipse/circle fit
- For sub-pixel precision use one of:
  - Intensity-weighted centroid on the thresholded blob
  - Circle fit by least squares on edge pixels (Canny → fit)
  - 2D Gaussian fit on the blob's intensity profile
- Reject frames where: blob area is implausible, blob is at frame edge, or fit residual is too large
- Average accepted frames; record sample standard deviation as 2D measurement uncertainty per camera

### triangulation/

**Purpose:** reconstruct the 3D ball position in the box coordinate frame.

**Inputs:** per-camera averaged 2D position, intrinsics, extrinsics, 2D covariance per camera.

**Outputs:** 3D ball position `(X, Y, Z)` in box frame, with 3×3 covariance `Σ_3D`.

**Implementation notes:**
- Use the linear DLT method to get an initial estimate from all cameras
- Refine by nonlinear least squares minimizing reprojection error across all cameras (Levenberg–Marquardt)
- Weight each camera's residual by its 2D covariance (Mahalanobis distance, not Euclidean)
- Recover 3D covariance from the Jacobian at the optimum: `Σ_3D = (Jᵀ W J)⁻¹`
- Sanity check: reprojection residuals should be on the order of the per-camera 2D noise

### error_propagation/

**Purpose:** quantify total measurement uncertainty by source.

**Sources to track separately:**
1. Intrinsic calibration residual (typical: 0.1–0.5 px)
2. Marker position uncertainty on the box (manufacturing/printing)
3. ArUco corner detection noise (typical: 0.1–0.3 px)
4. Box pose estimation residual (extrinsic reprojection error)
5. Ball detection noise (averaged out by N frames; ∝ 1/√N)
6. Triangulation geometric dilution (depends on camera angular spread)

**Implementation notes:**
- For each source, propagate to 3D position via Jacobian (analytical or numerical)
- Combine into a total covariance by summing covariances (assuming independence)
- Report both per-source contribution and total. Identifying the dominant error source is important for further refinement.
- Validate via Monte Carlo: perturb each source by its sigma, re-run pipeline, compare to analytical propagation

### comparison/

**Purpose:** compare measured 3D position to simulator's predicted trap position.

**Inputs:** measured 3D position with covariance, simulator output (trap location in box frame).

**Outputs:** offset vector, Mahalanobis distance, pass/fail per configurable threshold, plots.

**Implementation notes:**
- Both positions must be in the same coordinate frame (box frame). Verify this explicitly.
- Compute residual `r = measured - simulated`
- Mahalanobis: `r.T @ inv(Σ_3D) @ r` — this is χ² distributed under the null hypothesis
- Visualize: 3D plot with measured (with error ellipsoid) and simulated points

## Hardware Recommendations

### Cameras

Since the ball is stationary, expensive machine-vision cameras are unnecessary. Priority order for webcam selection:

1. **Resolution**: 1080p minimum, 4K preferred (more pixels → better sub-pixel precision)
2. **Lens quality**: low distortion, fixed focus preferred over autofocus
3. **Manual control**: ability to lock exposure and focus via software (UVC controls or vendor SDK)
4. **Mount**: 1/4" tripod thread

**Recommended models** (any of these work):
- Logitech Brio 4K (consumer, 4K, ~$200)
- Logitech C922/C920 (consumer, 1080p, ~$70)
- Arducam OV9281 USB modules (industrial-ish, global shutter, ~$50)

Avoid cameras with strong fisheye distortion or built-in image processing that cannot be disabled.

### Box

- 3D printed or CNC machined for true orthogonality
- Matte black surface (paint or filament)
- Internal dimensions large enough to contain trap region with margin
- One face open or removable for ball insertion / transducer access

### Markers

- ArUco DICT_4X4_50 or DICT_5X5_100 (avoid DICT_4X4_1000+ unless needed; smaller dictionaries are more robust)
- Minimum size: marker side ≥ 50 px in image at typical camera distance
- Print on matte paper, laminate flat, then measure the actual printed size with calipers and put that into the config (printers commonly have 0.5–1% scaling error)
- Marker layout config defines each marker's 4 corner positions in the box frame to ≤ 0.1 mm if possible

### Lighting

- Diffuse panel or softbox, broad and uniform
- Avoid direct point sources that create specular spots on the ball
- Lock white balance on cameras (a varying white balance changes apparent ball brightness)

### Tripods

- Rigid, fully tightened, on a stable surface
- Position cameras with ≥ 90° angular spread (e.g., 4 cameras at ~90° around the box, or tetrahedral arrangement)
- Avoid configurations where 2+ cameras share nearly the same axis — depth uncertainty will explode

## Configuration Files

### `config/box.yaml`
```yaml
box_dimensions: [width_mm, depth_mm, height_mm]
markers:
  - id: 0
    face: front
    corners_box_frame:  # 4 × 3 in box coordinates, mm
      - [x0, y0, z0]
      - [x1, y1, z1]
      - [x2, y2, z2]
      - [x3, y3, z3]
  - id: 1
    ...
aruco_dictionary: DICT_4X4_50
```

### `config/cameras.yaml`
```yaml
cameras:
  - id: cam_left
    serial: "ABC123"
    intrinsics_file: calibration/cam_left_intrinsics.yaml
    capture_resolution: [1920, 1080]
    exposure: -6
    focus: 0
```

## Workflow

1. **One-time per camera**: run intrinsic calibration with ChArUco board → save profile
2. **Box construction**: build box, attach markers, measure marker positions → save `box.yaml`
   - Optional: run self-calibration to refine marker positions
3. **Per experiment session**:
   1. Position cameras around box, lock focus/exposure
   2. Capture N frames per camera
   3. Run extrinsic solver to get camera poses
   4. Run ball detector to get per-camera 2D positions
   5. Triangulate to get 3D position with covariance
   6. Compare to simulator output
   7. Report offset, uncertainty breakdown, pass/fail

## Validation Plan

Before trusting any measurement against the simulator, validate the measurement system itself:

1. **Reprojection consistency**: triangulate, then reproject the 3D point to every camera. Residuals should be < 1 px and consistent with the 2D noise level.
2. **Multi-view box-pose consistency**: all cameras should agree on the box pose (compare each camera's pose to the average; differences indicate intrinsic, marker layout, or board issues).
3. **Ground truth test**: place the ball at a known location (e.g., on a precision micrometer stage or 3D printed jig with known coordinates) and verify the system recovers that position within stated uncertainty.
4. **Subset test**: re-triangulate using only N–1 cameras. The result should still fall within the error ellipsoid from the full N-camera result.

Without these validations, simulator-vs-measurement mismatch is unattributable.

## Tech Stack

- **Language**: Python 3.10+
- **Core libraries**:
  - `opencv-contrib-python` (must be `contrib` for ArUco)
  - `numpy`, `scipy` (optimization, linear algebra)
  - `pyyaml` (configs)
  - `matplotlib`, `plotly` (visualization)
  - `pytest` (tests)
- **Optional**:
  - `pyusb` or vendor SDKs for advanced camera control
  - `numba` if hot loops need speeding up

## Open Questions / Future Work

- Marker layout: physical measurement vs. self-calibration — decision pending on achievable physical accuracy.
- Number of cameras: 3 minimum, 4 strongly recommended for redundancy. More is better but with diminishing returns past ~6.
- If experiments later become dynamic (moving ball), this whole no-sync simplification breaks. The codebase should keep the per-camera capture pipeline modular so a synchronized capture backend can be plugged in without rewriting downstream stages.
- Whether to ingest simulator output via file (offline) or live API.
