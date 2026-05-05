# Acoustic Levitation Measurement System

Multi-camera measurement system that validates an acoustic levitation simulator against real-world experiments. A white styrofoam ball suspended in a static acoustic trap inside a black box is observed by multiple USB webcams. The system reconstructs the ball's 3D position and compares it to the simulator's predicted trap location.

---

## System Architecture

```
AcousticLevitation/
├── config/
│   ├── box.yaml                  # Box dimensions, ArUco marker positions, box→sim transform
│   └── cameras.yaml              # Camera IDs, serials, intrinsics paths, capture settings
├── common/
│   ├── __init__.py               # Shared data classes
│   └── io_utils.py               # YAML/JSON I/O helpers
├── intrinsic_calibration/
│   └── calibrate.py              # Per-camera lens calibration via ChArUco board
├── capture/
│   └── capture.py                # Multi-camera frame capture
├── extrinsic_solver/
│   └── solve.py                  # Camera pose estimation via ArUco board
├── ball_detector/
│   └── detect.py                 # Sub-pixel ball center detection
├── triangulation/
│   └── triangulate.py            # 3D position reconstruction (DLT + LM)
├── error_propagation/
│   └── propagate.py              # Uncertainty quantification (6 sources + Monte Carlo)
├── comparison/
│   └── compare.py                # Measured vs. simulated trap position
├── run_pipeline.py               # Full pipeline runner (single command)
├── sim.py                        # Acoustic trap simulator (pre-existing)
└── requirements.txt
```

### Module Summary

| Module | File | Purpose |
|---|---|---|
| `config/` | `box.yaml` | Box dims, ArUco marker corners (mm, box frame), box→sim transform |
| `config/` | `cameras.yaml` | Camera IDs, serials, intrinsics paths, capture settings |
| `common/` | `__init__.py` | All shared data classes (`CameraIntrinsics`, `CameraPose`, `BallDetection2D`, `TriangulationResult`, `ErrorBudget`, `ComparisonResult`) |
| `common/` | `io_utils.py` | YAML/JSON I/O, intrinsics load/save, box config loader, numpy serializer |
| `intrinsic_calibration/` | `calibrate.py` | ChArUco detection (new + legacy API), per-image outlier rejection, saves YAML |
| `capture/` | `capture.py` | UVC autofocus/auto-exposure disable, N-frame capture per camera, metadata JSON |
| `extrinsic_solver/` | `solve.py` | ArUco board pose via `estimatePoseBoard`, SE(3) Lie algebra averaging over frames |
| `ball_detector/` | `detect.py` | Otsu threshold → largest blob → Canny edge circle fit (LSQ) → temporal averaging |
| `triangulation/` | `triangulate.py` | DLT init → LM refinement weighted by Mahalanobis, 3D covariance `(JᵀWJ)⁻¹` |
| `error_propagation/` | `propagate.py` | 6 error sources via Monte Carlo + analytical propagation; MC validation |
| `comparison/` | `compare.py` | Loads `newton_x/y/z` from `sim.py` output, sim→box frame transform, Mahalanobis, 3D+2D plots |
| `run_pipeline.py` | — | Runs all 5 stages in sequence with one CLI command |

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `opencv-contrib-python >= 4.7.0` — **must be contrib** for ArUco support
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `pyyaml >= 6.0`
- `matplotlib >= 3.7.0`

---

## Physical Setup

1. Build a matte-black cuboid box (3D printed or CNC machined).
2. Print ArUco markers (DICT_4X4_50 recommended). Measure actual printed size with calipers — printers have 0.5–1% scaling error.
3. Attach markers to box faces. Measure each marker's corner positions in the box coordinate frame to ≤ 0.1 mm.
4. Enter corner positions into `config/box.yaml` (in mm, box frame).
5. Place 3–4 USB webcams on rigid tripods with ≥ 90° angular spread around the box.
6. Set up diffuse, uniform lighting (softbox or diffuse panel). Lock white balance on all cameras.

---

## Workflow

### Step 0 — Configure

Edit `config/box.yaml`:
- Set `box_dimensions` (mm).
- Fill in `markers` — each marker's 4 corner positions in box frame (mm).
- Set `box_to_sim` transform (rotation + translation from box frame to simulator frame in meters). This is the physical placement of the box above the transducer array.

Edit `config/cameras.yaml`:
- Add one entry per camera with ID, serial, intrinsics file path, resolution, and exposure.
- Run `python -m capture.capture --list-cameras` to find device indices, then fill in `serial_to_index`.

---

### Step 1 — Intrinsic Calibration (one-time per camera)

Capture 30–50 images of a ChArUco board per camera at varied angles and distances. Then:

```bash
python -m intrinsic_calibration.calibrate \
    --camera-id cam_front \
    --images-dir images/cam_front_charuco/ \
    --output calibration/cam_front_intrinsics.yaml \
    --squares-x 9 \
    --squares-y 6 \
    --square-length 0.04 \
    --marker-length 0.02 \
    --dict DICT_5X5_100
```

| Argument | Default | Description |
|---|---|---|
| `--camera-id` | required | Camera identifier (must match cameras.yaml) |
| `--images-dir` | required | Directory of ChArUco calibration images |
| `--output` | required | Output path for intrinsics YAML |
| `--squares-x` | `9` | ChArUco board horizontal square count |
| `--squares-y` | `6` | ChArUco board vertical square count |
| `--square-length` | `0.04` | Square side length in meters |
| `--marker-length` | `0.02` | Embedded marker side length in meters |
| `--dict` | `DICT_5X5_100` | ArUco dictionary name |
| `--max-reproj-px` | `1.0` | Per-image reprojection error threshold for outlier rejection |

Repeat for every camera. Saves `calibration/<camera_id>_intrinsics.yaml` per camera.

---

### Step 2 — Discover Camera Device Indices

```bash
python -m capture.capture --list-cameras
```

Prints available device indices and resolutions. Fill in `serial_to_index` in `cameras.yaml`.

---

### Step 3 — Capture Session

Position cameras around the box. Lock focus and exposure. Run the levitator to suspend the ball. Then:

```bash
python -m capture.capture \
    --config config/cameras.yaml \
    --output sessions/session_001 \
    --n-frames 200
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/cameras.yaml` | Cameras config file |
| `--output` | `sessions/session_001` | Session output directory |
| `--n-frames` | from config | Override frames_per_camera |

Output structure:
```
sessions/session_001/
├── cam_front/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── cam_right/
│   └── ...
└── metadata.json
```

---

### Step 4 — Run Full Pipeline (recommended)

Once calibration is done and frames are captured, run all remaining stages with one command:

```bash
python run_pipeline.py \
    --session sessions/session_001 \
    --sim-output simulation_outputs/hardware_trap_runs/attempt_004/summary.json \
    --box-config config/box.yaml \
    --cameras-config config/cameras.yaml \
    --calibration-dir calibration \
    --threshold-mm 2.0 \
    --sim-rank 1
```

| Argument | Default | Description |
|---|---|---|
| `--session` | required | Session directory |
| `--sim-output` | required | `summary.json` or `final_candidates_*.csv` from `sim.py` |
| `--box-config` | `config/box.yaml` | Box configuration |
| `--cameras-config` | `config/cameras.yaml` | Cameras configuration |
| `--calibration-dir` | `calibration` | Directory with intrinsics YAML files |
| `--threshold-mm` | `2.0` | Pass/fail Euclidean distance threshold (mm) |
| `--sim-rank` | `1` | Which sim.py candidate rank to compare against |
| `--skip-error-propagation` | off | Skip Monte Carlo error propagation (faster) |
| `--n-mc` | `500` | Monte Carlo trial count for error propagation |
| `--min-markers` | `3` | Min ArUco markers per frame for pose acceptance |
| `--max-reproj-px` | `2.0` | Max reprojection error (px) for pose frame rejection |
| `--min-ball-area` | `50` | Min blob area (px²) for ball detection |
| `--max-ball-area` | `50000` | Max blob area (px²) for ball detection |

Pipeline stages run in order:
1. **Extrinsic solver** → `session/extrinsics.json`
2. **Ball detector** → `session/ball_detections.json`
3. **Triangulation** → `session/triangulation.json`
4. **Error propagation** → `session/error_budget.json`
5. **Comparison** → `session/comparison/`

---

### Step 4 (alternative) — Run Stages Individually

**Extrinsic solver:**
```bash
python -m extrinsic_solver.solve \
    --session sessions/session_001 \
    --box-config config/box.yaml \
    --cameras-config config/cameras.yaml \
    --calibration-dir calibration \
    --min-markers 3 \
    --max-reproj-px 2.0
```

**Ball detector:**
```bash
python -m ball_detector.detect \
    --session sessions/session_001 \
    --cameras-config config/cameras.yaml \
    --calibration-dir calibration \
    --min-area 50 \
    --max-area 50000 \
    --max-fit-residual 3.0
```

**Triangulation:**
```bash
python -m triangulation.triangulate \
    --session sessions/session_001 \
    --cameras-config config/cameras.yaml \
    --calibration-dir calibration
```

**Error propagation:**
```bash
python -m error_propagation.propagate \
    --session sessions/session_001 \
    --box-config config/box.yaml \
    --cameras-config config/cameras.yaml \
    --calibration-dir calibration \
    --n-mc 500
```

**Comparison:**
```bash
python -m comparison.compare \
    --session sessions/session_001 \
    --sim-output simulation_outputs/hardware_trap_runs/attempt_004/summary.json \
    --box-config config/box.yaml \
    --threshold-mm 2.0 \
    --sim-rank 1
```

---

## Output Files

After a full pipeline run, the session directory contains:

```
sessions/session_001/
├── extrinsics.json          # T_cam_box (4×4) per camera, reprojection error
├── ball_detections.json     # Per-camera averaged 2D center + covariance
├── triangulation.json       # 3D position (m, box frame) + 3×3 covariance
├── error_budget.json        # Per-source uncertainty + Monte Carlo validation
└── comparison/
    ├── comparison_result.json   # Offset, Mahalanobis distance, pass/fail
    ├── comparison_3d.png        # 3D plot with error ellipsoid
    ├── comparison_xy.png        # XY projection
    ├── comparison_xz.png        # XZ projection
    └── comparison_yz.png        # YZ projection
```

### comparison_result.json structure

```json
{
  "measured_position_box_mm": [x, y, z],
  "simulated_position_box_mm": [x, y, z],
  "simulated_position_sim_mm": [x, y, z],
  "offset_mm": [dx, dy, dz],
  "euclidean_offset_mm": 1.23,
  "mahalanobis_distance": 2.45,
  "chi2_dof": 3,
  "passed": true,
  "threshold_mm": 2.0,
  "sim_candidate_rank": 1
}
```

The Mahalanobis distance is χ²(3) distributed under the null hypothesis (measured = simulated). The 95% critical value is 2.80.

---

## Error Budget

Six sources are propagated independently to 3D position uncertainty:

| Source | Description |
|---|---|
| `intrinsic_calibration` | Lens calibration residual reprojected to 3D |
| `marker_position` | Manufacturing/printing uncertainty of ArUco marker corners |
| `aruco_corner_detection` | Noise in detected marker corners (~0.2 px) |
| `box_pose_estimation` | Extrinsic reprojection error → T_cam_box uncertainty |
| `ball_detection` | Ball center noise (averaged over N frames, ∝ 1/√N) |
| `triangulation_geometry` | Geometric dilution from camera angular arrangement (GDOP) |

Total covariance = sum of independent source covariances + triangulation covariance.

Monte Carlo validation (N trials, default 500) compares the analytical total against empirical covariance from random perturbations. Frobenius ratio < 0.5 indicates good agreement.

---

## Simulator Output Format

The comparison module reads trap positions from `sim.py` output files:

- **`summary.json`**: reads `ideal_final_candidates[rank-1].newton_x/y/z` (meters, sim frame)
- **`final_candidates_*.csv`**: reads row where `rank == N`, columns `newton_x`, `newton_y`, `newton_z`

The rank-1 candidate is the strongest predicted trap (lowest Gor'kov potential, fully refined by Newton's method).

---

## Coordinate Frames

| Frame | Origin | Units | Used by |
|---|---|---|---|
| **Box frame** | Front-bottom-left corner of box | meters | All measurement stages |
| **Simulator frame** | Center of transducer array, z up | meters | `sim.py` output |
| **Camera frame** | Camera optical center | meters | Projection matrices |

The `box_to_sim` section in `box.yaml` defines the 4×4 SE(3) transform from box frame to simulator frame. **Edit this to match the physical placement of the box above the transducer array before comparing results.**

---

## Validation Checklist

Before trusting measurements against the simulator:

- [ ] Reprojection consistency: triangulate, reproject to every camera. Residuals < 1 px.
- [ ] Multi-view box-pose consistency: all cameras agree on box pose (compare each to the mean).
- [ ] Ground truth test: place ball at a known location (micrometer stage or printed jig) and verify recovery within stated uncertainty.
- [ ] Subset test: re-triangulate using N−1 cameras. Result falls within the full N-camera error ellipsoid.

---

## Design Decisions

**No synchronization required.** Ball is stationary (static trap), so cameras capture independently. Rolling shutter and consumer webcams are acceptable.

**Temporal averaging for sub-pixel precision.** N frames per camera are averaged. Per-camera 2D precision ≈ σ_single / √N. Standard deviation across frames is propagated as 2D measurement uncertainty.

**Multi-marker board eliminates planar ambiguity.** All ArUco markers on the box are treated as a single rigid body via `cv2.aruco.estimatePoseBoard`. Markers spanning multiple non-coplanar faces remove the rotation ambiguity that affects single-marker estimation.

**SE(3) Lie algebra averaging.** Pose matrices are averaged in the Lie algebra (not by naive matrix mean) to stay on the SE(3) manifold.

**Mahalanobis-weighted triangulation.** LM refinement weights each camera's residual by its 2D covariance (inverse), not Euclidean distance. Cameras with more frames (lower noise) contribute more.

**3D covariance from Jacobian.** `Σ_3D = (JᵀWJ)⁻¹` at the LM optimum gives a principled uncertainty estimate that reflects both 2D noise levels and geometric dilution.
