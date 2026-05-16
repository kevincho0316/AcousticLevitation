"""
Box marker calibration via bundle adjustment.

Phases:
  1. Parse YAML → BoxModel with nominal SE(3) poses.
  2. Detect ArUco markers on undistorted images.
  3. Seed camera poses per image via solvePnP.
  4. Joint bundle adjustment (TRF + Huber, sparse Jacobian).
  5. Outlier rejection + re-solve (inside bundle.py).
  6. Report stats, write enriched sidecar YAML, optional debug images.

Usage:
    python -m box_calibration.calibrate \\
        --images-dir  data/box_calib/ \\
        --intrinsics  calibration/cam_front_intrinsics.yaml \\
        --box-config  config/box.yaml \\
        --output      config/box.yaml \\
        [--min-markers 2] \\
        [--max-reproj-px 1.5] \\
        [--sigma-r-tilt-deg 10] \\
        [--sigma-r-yaw-deg 45] \\
        [--huber-scale 1.0] \\
        [--cross-validate] \\
        [--debug-dir debug/box_cal/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io_utils import load_box_config, load_intrinsics, load_yaml

from .bundle import cross_validate, run_bundle_adjustment
from .detect import detect_images
from .faces import build_box_model
from .init_poses import init_camera_poses
from .io_results import save_3d_plot, save_debug_overlays, write_output_yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine box marker positions via bundle adjustment")
    p.add_argument("--images-dir",        type=Path, required=True)
    p.add_argument("--intrinsics",         type=Path, required=True)
    p.add_argument("--box-config",         type=Path, default=Path("config/box.yaml"))
    p.add_argument("--output",             type=Path, default=Path("config/box.yaml"))
    p.add_argument("--min-markers",        type=int,   default=2,
                   help="Minimum markers per image (default 2)")
    p.add_argument("--max-reproj-px",      type=float, default=1.5,
                   help="Warn if final RMS exceeds this (default 1.5 px)")
    p.add_argument("--sigma-r-tilt-deg",   type=float, default=10.0,
                   help="Prior sigma for marker tilt (rotvec_x/y) in degrees (default 10°)")
    p.add_argument("--sigma-r-yaw-deg",    type=float, default=45.0,
                   help="Prior sigma for marker yaw (rotvec_z) in degrees (default 45°)")
    p.add_argument("--huber-scale",        type=float, default=1.0,
                   help="Huber loss scale in pixels (default 1.0)")
    p.add_argument("--cross-validate",     action="store_true",
                   help="Hold out last image and report holdout reprojection RMS")
    p.add_argument("--debug-dir",          type=Path, default=None,
                   help="Save debug overlays and 3D plot here")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                    "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
        for p in args.images_dir.glob(ext)
    )
    if not image_paths:
        print(f"ERROR: no images in {args.images_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images in {args.images_dir}")

    intrinsics = load_intrinsics(args.intrinsics)
    box_cfg = load_box_config(args.box_config)

    # Phase 1: nominal box geometry.
    box_model = build_box_model(box_cfg)
    print(f"\nBox model: {len(box_model.ids)} markers — ids {box_model.ids}")

    # Priors.
    sigma_t = float(box_cfg.get("marker_position_uncertainty_mm", 1.0)) / 1000.0
    sigma_r = np.array([
        np.radians(args.sigma_r_tilt_deg),
        np.radians(args.sigma_r_tilt_deg),
        np.radians(args.sigma_r_yaw_deg),
    ])
    print(f"Priors: sigma_t={sigma_t*1000:.1f} mm, "
          f"sigma_r_tilt={args.sigma_r_tilt_deg:.1f}°, "
          f"sigma_r_yaw={args.sigma_r_yaw_deg:.1f}°")

    # Phase 2: detect.
    print("\n--- Phase 2: Detection ---")
    detections = detect_images(
        image_paths,
        K=intrinsics.K,
        dist=intrinsics.dist,
        aruco_dict_name=box_cfg.get("aruco_dictionary", "DICT_4X4_50"),
        valid_ids=set(box_model.ids),
        min_markers=args.min_markers,
    )

    # Phase 3: seed camera poses using existing corners_box_frame_m.
    print("\n--- Phase 3: Initial camera poses ---")
    corners_box_m: dict[int, np.ndarray] = {
        int(m["id"]): m["corners_box_frame_m"]
        for m in box_cfg["markers"]
        if "corners_box_frame_m" in m
    }
    init_poses = init_camera_poses(detections, box_model, corners_box_m, intrinsics.K)

    # Filter images where pose init failed.
    valid_pairs = [(d, p) for d, p in zip(detections, init_poses) if p is not None]
    if len(valid_pairs) < len(detections):
        n_dropped = len(detections) - len(valid_pairs)
        print(f"  Dropped {n_dropped} images with failed pose init")
    detections = [d for d, _ in valid_pairs]
    init_poses = [p for _, p in valid_pairs]

    if len(detections) == 0:
        print("ERROR: no usable images after pose init. Aborting.")
        sys.exit(1)

    # Optional cross-validation before the main solve.
    if args.cross_validate:
        print("\n--- Cross-validation ---")
        cross_validate(detections, box_model, intrinsics.K, sigma_t, sigma_r,
                       huber_scale=args.huber_scale)

    # Phase 4+5: bundle adjustment + outlier rejection.
    print("\n--- Phase 4: Bundle adjustment ---")
    result = run_bundle_adjustment(
        detections=detections,
        box_model=box_model,
        init_poses=init_poses,
        K=intrinsics.K,
        sigma_t=sigma_t,
        sigma_r=sigma_r,
        huber_scale=args.huber_scale,
    )

    # Phase 5: report stats.
    print(f"\n--- Results ---")
    print(f"  Final RMS: {result.final_rms:.3f} px  ({len(result.detection_list)} observations)")
    if result.final_rms > args.max_reproj_px:
        print(f"  WARN: RMS exceeds --max-reproj-px={args.max_reproj_px}. "
              "Inspect debug images.")

    print("\n  Per-marker:")
    n_markers = len(box_model.ids)
    mk_offs = result.x[6 * result.n_cams :].reshape(n_markers, 6)
    for i, mid in enumerate(box_model.ids):
        t = mk_offs[i, 3:] * 1000.0   # mm
        r = np.degrees(mk_offs[i, :3])
        disp = float(np.linalg.norm(t))
        rot_mag = float(np.degrees(np.linalg.norm(mk_offs[i, :3])))
        flag = ""
        if disp > 3 * sigma_t * 1000:
            flag += "  *** large translation — check pasting"
        if rot_mag > 10.0:
            flag += "  *** large rotation — check face assignment"
        print(f"    id={mid:2d} ({box_model.faces[i]:6s}): "
              f"t=[{t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f}]mm "
              f"r=[{r[0]:+.1f},{r[1]:+.1f},{r[2]:+.1f}]°  "
              f"|t|={disp:.2f}mm  rms={result.per_marker_rms[i]:.3f}px  "
              f"n_obs={result.n_obs_per_marker[i]}{flag}")

    print("\n  Per-image RMS:")
    for cam_idx, (path, _, _) in enumerate(detections):
        print(f"    [{cam_idx:2d}] {path.name}: {result.per_image_rms[cam_idx]:.3f} px")

    # Phase 6: write output.
    print("\n--- Phase 6: Output ---")
    write_output_yaml(args.box_config, box_model, result, args.output)

    if args.debug_dir is not None:
        save_debug_overlays(detections, result, box_model, intrinsics.K, Path(args.debug_dir))
        save_3d_plot(box_model, result, Path(args.debug_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
