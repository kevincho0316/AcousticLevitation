"""Full self-calibration of box marker layout via bundle adjustment.

Trusts ONLY from box.yaml:
  - box_dimensions (informational, used for viz)
  - marker_side_mm (fixes metric scale per marker)
  - marker IDs (which IDs to expect)

Each marker's 6-DOF pose in the box frame is recovered from the images.
One marker (anchor) is fixed at identity to remove gauge ambiguity → box
frame = anchor marker's local frame. User aligns to physical box later
via box_to_sim in the YAML if desired.

Phases:
  1. Detect ArUco markers on undistorted images.
  2. Per (image, marker) IPPE_SQUARE pose candidates.
  3. Co-visibility graph + anchor pick.
  4. BFS init marker poses (disambiguate IPPE candidates by joint reproj).
  5. Init camera poses via solvePnP using initial marker corners.
  6. Bundle adjustment (reprojection only, anchor pinned).
  7. Outlier rejection + re-solve.
  8. Output enriched YAML + debug overlays + 3D plot.

Usage:
    python -m box_calibration.calibrate \\
        --images-dir  data/box_calib/ \\
        --intrinsics  calibration/cam_front_intrinsics.yaml \\
        --box-config  config/box.yaml \\
        --output      config/box.yaml \\
        [--anchor-marker-id ID] \\
        [--min-markers 2] \\
        [--max-reproj-px 1.5] \\
        [--huber-scale 1.0] \\
        [--max-initial-reproj-px 40] \\
        [--force-output] \\
        [--debug-dir debug/box_cal/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from common.io_utils import load_box_config, load_intrinsics

from .bundle import run_bundle_adjustment
from .detect import detect_images
from .faces import marker_corners_mkr_frame
from .init_graph import (
    build_covis_graph,
    disambiguate_by_nominal,
    init_marker_poses,
    per_marker_ippe,
    pick_anchor,
)
from .faces import build_box_model
from .init_poses import init_camera_poses
from .box_fit import apply_box_fit, fit_box_frame_with_labels
from .io_results import save_3d_plot, save_debug_overlays, write_output_yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-calibrate box marker layout via bundle adjustment "
                    "(no nominal layout trust, no priors)."
    )
    p.add_argument("--images-dir",        type=Path, required=True)
    p.add_argument("--intrinsics",         type=Path, required=True)
    p.add_argument("--box-config",         type=Path, default=Path("config/box.yaml"))
    p.add_argument("--output",             type=Path, default=Path("config/box.yaml"))
    p.add_argument("--anchor-marker-id",   type=int,   default=None,
                   help="Marker ID to pin at box-frame origin. "
                        "Default: marker with most observations.")
    p.add_argument("--min-markers",        type=int,   default=2,
                   help="Minimum markers per image (default 2)")
    p.add_argument("--max-reproj-px",      type=float, default=1.5,
                   help="Warn/fail if final RMS exceeds this (default 1.5 px)")
    p.add_argument("--huber-scale",        type=float, default=1.0,
                   help="Huber loss scale in pixels (default 1.0)")
    p.add_argument("--max-initial-reproj-px", type=float, default=40.0,
                   help="Abort if initial bundle RMS is above this (default 40 px)")
    p.add_argument("--force-output",        action="store_true",
                   help="Write YAML even when final RMS exceeds --max-reproj-px")
    p.add_argument("--debug-dir",          type=Path, default=None,
                   help="Save debug overlays and 3D plot here")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_paths_by_key: dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        for p in args.images_dir.glob(ext):
            image_paths_by_key.setdefault(str(p.resolve()).casefold(), p)
    image_paths = sorted(image_paths_by_key.values())
    if not image_paths:
        print(f"ERROR: no images in {args.images_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images in {args.images_dir}")

    intrinsics = load_intrinsics(args.intrinsics)
    box_cfg = load_box_config(args.box_config)

    marker_ids = [int(m["id"]) for m in box_cfg["markers"]]
    marker_side_m = float(box_cfg["marker_side_mm"]) / 1000.0
    dims = box_cfg["box_dimensions"]
    print(f"\nBox dims: {dims['width_mm']}×{dims['height_mm']}×{dims['depth_mm']} mm")
    print(f"Marker side: {box_cfg['marker_side_mm']} mm")
    print(f"Marker IDs to track: {marker_ids}")

    # Phase 1: detect.
    print("\n--- Phase 1: Detection ---")
    detections = detect_images(
        image_paths,
        K=intrinsics.K,
        dist=intrinsics.dist,
        aruco_dict_name=box_cfg.get("aruco_dictionary", "DICT_4X4_50"),
        valid_ids=set(marker_ids),
        min_markers=args.min_markers,
    )

    # Phase 2: per-marker IPPE candidates.
    print("\n--- Phase 2: Per-marker IPPE poses ---")
    per_img_cands = per_marker_ippe(detections, intrinsics.K, marker_side_m)
    n_dets = sum(len(d) for d in per_img_cands)
    print(f"  IPPE candidates computed for {n_dets} (image, marker) detections")

    # Phase 2.5: resolve the IPPE square-flip ambiguity per image using the
    # nominal marker orientations from box.yaml face labels. Without this, the
    # two near-equal IPPE candidates (ambiguity ratio ~1 for near-frontal flat
    # markers) get mixed across images by the BFS index, producing a garbage
    # marker layout and a huge initial bundle reprojection.
    print("\n--- Phase 2.5: IPPE flip disambiguation (nominal face normals) ---")
    box_model = build_box_model(box_cfg)
    nominal_R_by_id = {
        mid: box_model.nominal_poses[i][:3, :3] for i, mid in enumerate(box_model.ids)
    }
    n_amb_before = sum(1 for per in per_img_cands for cl in per.values() if len(cl) > 1)
    per_img_cands = disambiguate_by_nominal(per_img_cands, nominal_R_by_id)
    print(f"  Resolved {n_amb_before} ambiguous (image, marker) candidates to a single flip")

    # Phase 3: co-visibility + anchor.
    print("\n--- Phase 3: Co-visibility graph + anchor ---")
    observed_ids, adj, n_obs = build_covis_graph(per_img_cands)
    if not observed_ids:
        print("ERROR: no markers observed. Aborting.")
        sys.exit(1)

    try:
        anchor_id = pick_anchor(n_obs, override=args.anchor_marker_id)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"  Anchor marker: id={anchor_id}  (n_obs={n_obs[anchor_id]})")
    print(f"  Observed markers (n_obs): {dict(sorted(n_obs.items()))}")

    # Phase 4: BFS init marker poses.
    print("\n--- Phase 4: BFS marker pose init ---")
    init_mk_poses = init_marker_poses(
        per_img_cands, detections, adj, anchor_id, intrinsics.K, marker_side_m,
    )
    placed_ids = sorted(init_mk_poses.keys())
    print(f"  Placed {len(placed_ids)}/{len(observed_ids)} markers: {placed_ids}")
    missing = sorted(set(observed_ids) - set(placed_ids))
    if missing:
        print(f"  WARNING: dropped disconnected markers: {missing}")

    # Narrow marker_ids down to the connected component, preserving original order.
    marker_ids = [mid for mid in marker_ids if mid in init_mk_poses]
    if anchor_id not in marker_ids:
        print(f"ERROR: anchor id={anchor_id} not in connected marker set. Aborting.")
        sys.exit(1)

    # Phase 5: seed camera poses.
    print("\n--- Phase 5: Initial camera poses ---")
    corners_mkr = marker_corners_mkr_frame(marker_side_m)
    corners_hom = np.hstack([corners_mkr, np.ones((4, 1))])  # (4,4) rows
    corners_box_m: dict[int, np.ndarray] = {}
    for mid in marker_ids:
        T = init_mk_poses[mid]
        corners_box_m[mid] = (T @ corners_hom.T).T[:, :3]

    # Build a minimal BoxModel-shim for init_camera_poses (it only uses .ids).
    class _Shim:
        pass
    shim = _Shim()
    shim.ids = marker_ids

    init_poses = init_camera_poses(detections, shim, corners_box_m, intrinsics.K)

    valid_pairs = [(d, p) for d, p in zip(detections, init_poses) if p is not None]
    if len(valid_pairs) < len(detections):
        n_dropped = len(detections) - len(valid_pairs)
        print(f"  Dropped {n_dropped} images with failed pose init")
    detections = [d for d, _ in valid_pairs]
    init_poses = [p for _, p in valid_pairs]

    if len(detections) == 0:
        print("ERROR: no usable images after pose init. Aborting.")
        sys.exit(1)

    # Phase 6+7: bundle adjustment + outlier rejection.
    print("\n--- Phase 6: Bundle adjustment ---")
    try:
        result = run_bundle_adjustment(
            detections=detections,
            marker_ids=marker_ids,
            init_marker_poses=init_mk_poses,
            init_camera_poses=init_poses,
            anchor_id=anchor_id,
            K=intrinsics.K,
            marker_side_m=marker_side_m,
            huber_scale=args.huber_scale,
            max_initial_rms_px=args.max_initial_reproj_px,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    # Stats (pre-fit reprojection).
    print(f"\n--- Results ---")
    print(f"  Final RMS: {result.final_rms:.3f} px  ({len(result.detection_list)} observations)")
    rms_too_high = result.final_rms > args.max_reproj_px
    if rms_too_high:
        print(f"  WARN: RMS exceeds --max-reproj-px={args.max_reproj_px}.")

    # Phase 6.5: Box-frame fit with face labels from box.yaml.
    print("\n--- Phase 6.5: Box-frame fit (markers on faces) ---")
    print("  Using face labels from box.yaml (fixed assignments).")

    face_by_id = {int(m["id"]): m["face"] for m in box_cfg["markers"] if "face" in m}
    missing_faces = [mid for mid in result.marker_ids if mid not in face_by_id]
    if missing_faces:
        print(f"ERROR: box.yaml missing 'face' key for markers: {missing_faces}")
        sys.exit(2)
    face_labels = [face_by_id[mid] for mid in result.marker_ids]

    T_align, face_assigns, fit_info = fit_box_frame_with_labels(
        result.marker_poses,
        face_labels=face_labels,
        W_mm=float(dims["width_mm"]),
        H_mm=float(dims["height_mm"]),
        D_mm=float(dims["depth_mm"]),
        marker_side_m=marker_side_m,
    )
    apply_box_fit(result, T_align)
    print(f"  Fit final cost: {fit_info['final_cost']:.3e}")
    print(f"  Plane residual: rms={fit_info['rms_plane_mm']:.2f} mm  "
          f"max={fit_info['max_plane_mm']:.2f} mm")
    print(f"  Normal residual: rms={fit_info['rms_normal_deg']:.2f}°  "
          f"max={fit_info['max_normal_deg']:.2f}°")

    print("\n  Per-marker (box frame):")
    for i, mid in enumerate(result.marker_ids):
        T = result.marker_poses[i]
        t_mm = T[:3, 3] * 1000.0
        import cv2 as _cv2
        rvec, _ = _cv2.Rodrigues(T[:3, :3])
        r = np.degrees(rvec.ravel())
        print(f"    id={mid:2d} [{face_assigns[i]:>6s}]: "
              f"t=[{t_mm[0]:+8.2f},{t_mm[1]:+8.2f},{t_mm[2]:+8.2f}]mm "
              f"r=[{r[0]:+7.1f},{r[1]:+7.1f},{r[2]:+7.1f}]°  "
              f"plane={fit_info['plane_resid_mm'][i]:+5.2f}mm  "
              f"n_resid={fit_info['normal_resid_deg'][i]:.2f}°  "
              f"rms={result.per_marker_rms[i]:.3f}px  "
              f"n_obs={result.n_obs_per_marker[i]}")

    # Attach face labels + fit info so downstream (YAML write, plot) can use them.
    result.face_assigns = face_assigns
    result.fit_info = fit_info

    print("\n  Per-image RMS:")
    for cam_idx, (path, _, _) in enumerate(detections):
        print(f"    [{cam_idx:2d}] {path.name}: {result.per_image_rms[cam_idx]:.3f} px")

    # Output.
    print("\n--- Phase 7: Output ---")
    if rms_too_high and not args.force_output:
        print("  ERROR: calibration failed quality gate; YAML output skipped. "
              "Use --force-output to write anyway.")
    else:
        write_output_yaml(args.box_config, box_cfg, result, marker_side_m, args.output)

    if args.debug_dir is not None:
        save_debug_overlays(
            detections,
            result,
            intrinsics.K,
            marker_side_m,
            box_cfg,
            Path(args.debug_dir),
        )
        save_3d_plot(result, box_cfg, marker_side_m, Path(args.debug_dir))

    if rms_too_high and not args.force_output:
        sys.exit(2)

    print("\nDone.")


if __name__ == "__main__":
    main()
