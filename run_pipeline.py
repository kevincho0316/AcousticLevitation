"""
Full measurement pipeline runner.

Runs all stages in sequence for a captured session and produces a final
comparison report against the simulator's predicted trap location.

Stages:
  1. extrinsic_solver  — estimate camera poses from ArUco board
  2. ball_detector     — detect ball centers in all camera frames
  3. triangulation     — reconstruct 3D position
  4. error_propagation — quantify uncertainty (optional, slower)
  5. comparison        — compare to simulator output

Usage:
    python run_pipeline.py \\
        --session sessions/session_001 \\
        --sim-output simulation_outputs/hardware_trap_runs/attempt_004/summary.json \\
        [--box-config config/box.yaml] \\
        [--cameras-config config/cameras.yaml] \\
        [--calibration-dir calibration] \\
        [--threshold-mm 2.0] \\
        [--sim-rank 1] \\
        [--skip-error-propagation] \\
        [--n-mc 500]

Prerequisites:
  - Intrinsic calibration files must exist in --calibration-dir.
  - Frames must be captured to --session/<camera_id>/frame_NNNN.png.
  - box.yaml and cameras.yaml must be configured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full acoustic levitation measurement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session", type=Path, required=True,
                   help="Session directory containing per-camera frame subdirectories")
    p.add_argument("--sim-output", type=Path, required=True,
                   help="Simulator output: summary.json or final_candidates_*.csv from sim.py")
    p.add_argument("--box-config", type=Path, default=Path("config/box.yaml"))
    p.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    p.add_argument("--threshold-mm", type=float, default=2.0,
                   help="Pass/fail Euclidean distance threshold (mm)")
    p.add_argument("--sim-rank", type=int, default=1,
                   help="Simulator candidate rank to compare against")
    p.add_argument("--skip-error-propagation", action="store_true",
                   help="Skip Monte Carlo error propagation (faster, less informative)")
    p.add_argument("--n-mc", type=int, default=500,
                   help="Number of Monte Carlo trials for error propagation")
    p.add_argument("--min-markers", type=int, default=3,
                   help="Minimum ArUco markers per frame for pose estimation")
    p.add_argument("--max-reproj-px", type=float, default=2.0,
                   help="Max reprojection error (px) for pose frame rejection")
    p.add_argument("--min-ball-area", type=int, default=50,
                   help="Min blob area (px²) for ball detection")
    p.add_argument("--max-ball-area", type=int, default=50_000,
                   help="Max blob area (px²) for ball detection")
    p.add_argument("--interactive", action="store_true",
                   help="Show numbered blob selection UI for ball detection")
    return p.parse_args()


def _stage(name: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Stage: {name}")
    print(f"{'─'*60}")


def main() -> None:
    args = _parse_args()

    session = args.session
    if not session.exists():
        sys.exit(f"Session directory not found: {session}")
    if not args.sim_output.exists():
        sys.exit(f"Simulator output not found: {args.sim_output}")

    t0 = time.monotonic()

    # ── Stage 1: Extrinsic solver ──────────────────────────────────────────
    _stage("Extrinsic solver (camera pose estimation via ArUco)")
    from extrinsic_solver.solve import solve_session
    solve_session(
        session_dir=session,
        box_config_path=args.box_config,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=session / "extrinsics.json",
        min_markers=args.min_markers,
        max_reproj_px=args.max_reproj_px,
    )

    # ── Stage 2: Ball detector ─────────────────────────────────────────────
    _stage("Ball detector (sub-pixel 2D center per camera)")
    from ball_detector.detect import detect_session
    detect_session(
        session_dir=session,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=session / "ball_detections.json",
        min_area=args.min_ball_area,
        max_area=args.max_ball_area,
        interactive=args.interactive,
    )

    # ── Stage 3: Triangulation ─────────────────────────────────────────────
    _stage("Triangulation (DLT + Levenberg–Marquardt, weighted by 2D covariance)")
    from triangulation.triangulate import triangulate_session
    triangulate_session(
        session_dir=session,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=session / "triangulation.json",
    )

    # ── Stage 4: Error propagation ─────────────────────────────────────────
    if not args.skip_error_propagation:
        _stage("Error propagation (Monte Carlo + analytical, per-source budget)")
        from error_propagation.propagate import propagate_session
        propagate_session(
            session_dir=session,
            box_config_path=args.box_config,
            cameras_config_path=args.cameras_config,
            calibration_dir=args.calibration_dir,
            output_path=session / "error_budget.json",
            n_mc=args.n_mc,
        )
    else:
        print("\nSkipping error propagation (--skip-error-propagation).")

    # ── Stage 5: Comparison ────────────────────────────────────────────────
    _stage("Comparison (measured vs. simulated trap position)")
    from comparison.compare import compare_session
    result = compare_session(
        session_dir=session,
        sim_output_path=args.sim_output,
        box_config_path=args.box_config,
        output_dir=session / "comparison",
        threshold_mm=args.threshold_mm,
        sim_rank=args.sim_rank,
    )

    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f} s")
    print(f"Final result: {'PASS ✓' if result.passed else 'FAIL ✗'}")
    print(f"  Offset: {float(result.offset_vector_box.dot(result.offset_vector_box)**0.5 * 1000):.3f} mm")
    print(f"  Mahalanobis: {result.mahalanobis_distance:.3f}")
    print(f"  Output: {session / 'comparison'}")


if __name__ == "__main__":
    main()
