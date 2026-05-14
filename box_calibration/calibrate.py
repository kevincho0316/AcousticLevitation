"""
Box marker calibration via bundle adjustment.

Jointly optimizes per-image camera poses and per-marker center offsets + in-plane
rotation so that ArUco corner reprojection error is minimized.  Each marker is
constrained to lie on its declared face by the parameterization.

Usage:
    python -m box_calibration.calibrate \\
        --images-dir  data/box_calib/ \\
        --intrinsics  calibration/cam_front_intrinsics.yaml \\
        --box-config  config/box.yaml \\
        --output      config/box.yaml \\
        [--min-markers  3] \\
        [--max-reproj-px 1.5] \\
        [--debug-dir  debug/box_cal/]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CameraIntrinsics
from common.io_utils import _FACE_AXES, load_box_config, load_intrinsics, load_yaml, save_yaml
from common.se3_utils import _se3_exp, _se3_log
from extrinsic_solver.solve import _build_board, _detect_markers

# face → (coordinate axis index, expected value fn taking W,D,H)
_FACE_PLANE: dict[str, tuple[int, str]] = {
    "front":  (2, "0"),
    "back":   (2, "D"),
    "right":  (0, "W"),
    "left":   (0, "0"),
    "top":    (1, "H"),
    "bottom": (1, "0"),
}


# ── Marker geometry ───────────────────────────────────────────────────────────

def _default_center_m(marker: dict, box_cfg: dict) -> np.ndarray:
    """Initial marker center in box frame (meters)."""
    dims = box_cfg["box_dimensions"]
    W = float(dims["width_mm"]) / 1000.0
    D = float(dims["depth_mm"]) / 1000.0
    H = float(dims["height_mm"]) / 1000.0

    if "corners_box_frame" in marker:
        return (np.array(marker["corners_box_frame"], dtype=np.float64) / 1000.0).mean(axis=0)

    face = marker["face"]
    if "center_box_mm" in marker:
        return np.array(marker["center_box_mm"], dtype=np.float64) / 1000.0
    return _FACE_AXES[face]["c"](W, D, H)


def _corners_from_params(
    face: str, default_center: np.ndarray, params: np.ndarray, s: float
) -> np.ndarray:
    """(4,3) corner positions in box frame from optimization params (du, dv, angle)."""
    du, dv, angle = float(params[0]), float(params[1]), float(params[2])
    r_vec = _FACE_AXES[face]["r"].astype(np.float64)
    u_vec = _FACE_AXES[face]["u"].astype(np.float64)

    center = default_center + du * r_vec + dv * u_vec
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    r_rot = cos_a * r_vec + sin_a * u_vec
    u_rot = -sin_a * r_vec + cos_a * u_vec

    h = s / 2.0
    return np.array([
        center - h * r_rot + h * u_rot,   # 0: top-left
        center + h * r_rot + h * u_rot,   # 1: top-right
        center + h * r_rot - h * u_rot,   # 2: bottom-right
        center - h * r_rot - h * u_rot,   # 3: bottom-left
    ], dtype=np.float64)


# ── Image processing ──────────────────────────────────────────────────────────

def _build_detections(
    image_paths: list[Path],
    intrinsics: CameraIntrinsics,
    box_cfg: dict,
    min_markers: int,
) -> tuple[np.ndarray, list, list[np.ndarray]]:
    """
    Undistort images, detect ArUco markers, initialize camera poses.

    Returns:
        cam_xis      — (N_accepted, 6) se3 log of initial T_cam_box per image
        detections   — list of (cam_idx, marker_local_idx, obs_corners_4x2)
        undist_imgs  — undistorted BGR images, one per accepted image
    """
    board, aruco_dict = _build_board(box_cfg)
    K = intrinsics.K.astype(np.float32)
    dist = intrinsics.dist.astype(np.float32)
    zero_dist = np.zeros(5, dtype=np.float32)

    id_to_idx = {int(m["id"]): i for i, m in enumerate(box_cfg["markers"])}

    try:
        det_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        det_params = cv2.aruco.DetectorParameters_create()

    cam_xis: list[np.ndarray] = []
    detections: list[tuple[int, int, np.ndarray]] = []
    undist_imgs: list[np.ndarray] = []
    n_rejected = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            n_rejected += 1
            continue

        img_ud = cv2.undistort(img, K, dist)
        gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)
        corners, ids = _detect_markers(gray, aruco_dict, det_params)

        if ids is None or len(ids) < min_markers:
            n_rejected += 1
            continue

        n_valid, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board, K, zero_dist, None, None
        )
        if n_valid == 0:
            n_rejected += 1
            continue

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()

        cam_idx = len(cam_xis)
        cam_xis.append(_se3_log(T))
        undist_imgs.append(img_ud)

        for corner_set, mid in zip(corners, ids.ravel()):
            mid = int(mid)
            if mid not in id_to_idx:
                continue
            detections.append((cam_idx, id_to_idx[mid], corner_set.reshape(4, 2).astype(np.float64)))

    n_accepted = len(cam_xis)
    print(f"  Images: {n_accepted} accepted, {n_rejected} skipped")
    if n_accepted == 0:
        raise RuntimeError(
            "No images produced a valid initial pose. "
            "Check that enough markers are visible and box.yaml is correct."
        )

    return np.array(cam_xis, dtype=np.float64), detections, undist_imgs


# ── Residuals ─────────────────────────────────────────────────────────────────

def _residuals_fn(
    x: np.ndarray,
    detections: list,
    K_f32: np.ndarray,
    marker_info: list[tuple[str, np.ndarray, float]],
    n_cams: int,
    n_markers: int,
) -> np.ndarray:
    """Flat reprojection-error residuals for least_squares."""
    cam_xis = x[: n_cams * 6].reshape(n_cams, 6)
    marker_params = x[n_cams * 6 :].reshape(n_markers, 3)

    zero_dist = np.zeros(5, dtype=np.float32)

    # Precompute refined corners for all markers
    mk_corners = [
        _corners_from_params(face, center, marker_params[i], s).astype(np.float32)
        for i, (face, center, s) in enumerate(marker_info)
    ]

    res: list[float] = []
    for cam_idx, mk_idx, obs in detections:
        T = _se3_exp(cam_xis[cam_idx])
        rvec, _ = cv2.Rodrigues(T[:3, :3].astype(np.float32))
        tvec = T[:3, 3].astype(np.float32)

        proj, _ = cv2.projectPoints(mk_corners[mk_idx], rvec, tvec, K_f32, zero_dist)
        res.extend((obs - proj.reshape(4, 2)).ravel())

    return np.array(res, dtype=np.float64)


# ── Debug visualization ───────────────────────────────────────────────────────

def _save_debug_images(
    undist_imgs: list[np.ndarray],
    image_paths: list[Path],
    detections: list,
    x: np.ndarray,
    K_f32: np.ndarray,
    marker_info: list[tuple[str, np.ndarray, float]],
    n_cams: int,
    n_markers: int,
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    zero_dist = np.zeros(5, dtype=np.float32)

    cam_xis = x[: n_cams * 6].reshape(n_cams, 6)
    marker_params = x[n_cams * 6 :].reshape(n_markers, 3)
    mk_corners = [
        _corners_from_params(face, center, marker_params[i], s).astype(np.float32)
        for i, (face, center, s) in enumerate(marker_info)
    ]

    # Group detections by camera
    cam_dets: dict[int, list] = {i: [] for i in range(n_cams)}
    for cam_idx, mk_idx, obs in detections:
        cam_dets[cam_idx].append((mk_idx, obs))

    for cam_idx, img in enumerate(undist_imgs):
        vis = img.copy()
        T = _se3_exp(cam_xis[cam_idx])
        rvec, _ = cv2.Rodrigues(T[:3, :3].astype(np.float32))
        tvec = T[:3, 3].astype(np.float32)

        for mk_idx, obs in cam_dets[cam_idx]:
            for pt in obs.reshape(4, 2):
                cv2.circle(vis, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

            proj, _ = cv2.projectPoints(mk_corners[mk_idx], rvec, tvec, K_f32, zero_dist)
            for pt in proj.reshape(4, 2):
                cv2.circle(vis, tuple(pt.astype(int)), 5, (0, 0, 255), 2)

        stem = image_paths[cam_idx].stem if cam_idx < len(image_paths) else f"img{cam_idx:03d}"
        cv2.imwrite(str(debug_dir / f"debug_{stem}.jpg"), vis)

    print(f"  Debug images → {debug_dir}/  (green=detected, red=reprojected refined)")


# ── Main calibration routine ──────────────────────────────────────────────────

def calibrate_box_markers(
    image_paths: list[Path],
    intrinsics: CameraIntrinsics,
    box_cfg: dict,
    min_markers: int = 3,
    max_reproj_px: float = 1.5,
    debug_dir: Path | None = None,
) -> dict:
    """
    Run bundle adjustment to refine marker center positions.

    Returns a deep copy of box_cfg with each marker's ``corners_box_frame``
    set to the optimized value (list of 4×[x,y,z] in mm).
    """
    # Validate that all markers have a face declaration
    for m in box_cfg["markers"]:
        if "face" not in m:
            raise ValueError(
                f"Marker {m.get('id', '?')} has no 'face' key. "
                "Box calibration requires 'face' to constrain each marker to its plane."
            )

    s = box_cfg["marker_side_m"]
    marker_info = [
        (m["face"], _default_center_m(m, box_cfg), s)
        for m in box_cfg["markers"]
    ]
    n_markers = len(box_cfg["markers"])
    K_f32 = intrinsics.K.astype(np.float32)

    cam_xis, detections, undist_imgs = _build_detections(
        image_paths, intrinsics, box_cfg, min_markers
    )
    n_cams = len(cam_xis)

    x0 = np.concatenate([cam_xis.ravel(), np.zeros(n_markers * 3)])

    res0 = _residuals_fn(x0, detections, K_f32, marker_info, n_cams, n_markers)
    print(f"  Initial RMS reprojection error: {np.sqrt(np.mean(res0**2)):.3f} px")

    result = least_squares(
        _residuals_fn,
        x0,
        args=(detections, K_f32, marker_info, n_cams, n_markers),
        method="lm",
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=20000,
    )

    final_rms = float(np.sqrt(np.mean(result.fun**2)))
    print(f"  Final  RMS reprojection error: {final_rms:.3f} px")
    if final_rms > max_reproj_px:
        print(
            f"  WARN: Final RMS {final_rms:.3f} px exceeds max_reproj_px={max_reproj_px}. "
            "Results may be unreliable — inspect debug images."
        )

    optimized_mk_params = result.x[n_cams * 6 :].reshape(n_markers, 3)

    # Report per-marker displacement
    print("\n  Per-marker displacement from initial position:")
    dims = box_cfg["box_dimensions"]
    W = float(dims["width_mm"]) / 1000.0
    D = float(dims["depth_mm"]) / 1000.0
    H = float(dims["height_mm"]) / 1000.0
    face_plane_vals = {"0": 0.0, "W": W, "D": D, "H": H}

    refined_cfg = copy.deepcopy(box_cfg)
    for i, m in enumerate(refined_cfg["markers"]):
        face = m["face"]
        default_center = marker_info[i][1]
        params = optimized_mk_params[i]
        du, dv, angle = params
        disp_mm = float(np.linalg.norm([du, dv])) * 1000.0
        print(
            f"    id={m['id']:2d} ({face:6s}): "
            f"du={du*1000:+.2f}mm  dv={dv*1000:+.2f}mm  "
            f"angle={np.degrees(angle):+.2f}°  |disp|={disp_mm:.2f}mm"
        )
        if disp_mm > 5.0:
            print(f"             WARN: large displacement — check detection quality")

        corners = _corners_from_params(face, default_center, params, s)

        # Geometric sanity: on face plane
        axis_idx, plane_key = _FACE_PLANE[face]
        expected = face_plane_vals[plane_key]
        max_dev = float(np.max(np.abs(corners[:, axis_idx] - expected)))
        if max_dev > 1e-6:
            print(f"             WARN: corners deviate {max_dev*1000:.4f}mm from {face} face plane")

        # Geometric sanity: inside box
        if not (
            np.all(corners[:, 0] >= -0.001) and np.all(corners[:, 0] <= W + 0.001)
            and np.all(corners[:, 1] >= -0.001) and np.all(corners[:, 1] <= H + 0.001)
            and np.all(corners[:, 2] >= -0.001) and np.all(corners[:, 2] <= D + 0.001)
        ):
            print(f"             WARN: corners outside box bounding box")

        # Write refined corners (mm) back; load_box_config checks this key first
        m["corners_box_frame"] = (corners * 1000.0).tolist()
        m["corners_box_frame_m"] = corners  # keep resolved meters version in-memory

    if debug_dir is not None:
        _save_debug_images(
            undist_imgs, image_paths, detections, result.x,
            K_f32, marker_info, n_cams, n_markers, Path(debug_dir)
        )

    return refined_cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine box marker positions via bundle adjustment")
    p.add_argument("--images-dir",    type=Path, required=True,
                   help="Directory containing calibration images (jpg/png)")
    p.add_argument("--intrinsics",    type=Path, required=True,
                   help="Camera intrinsics YAML from intrinsic_calibration")
    p.add_argument("--box-config",    type=Path, default=Path("config/box.yaml"))
    p.add_argument("--output",        type=Path, default=Path("config/box.yaml"),
                   help="Destination for refined box.yaml (can be same as --box-config)")
    p.add_argument("--min-markers",   type=int,   default=3)
    p.add_argument("--max-reproj-px", type=float, default=1.5)
    p.add_argument("--debug-dir",     type=Path,  default=None,
                   help="Save annotated images here (green=detected, red=reprojected)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    image_paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in args.images_dir.glob(ext)
    )
    if not image_paths:
        print(f"ERROR: no images found in {args.images_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images in {args.images_dir}")

    intrinsics = load_intrinsics(args.intrinsics)
    box_cfg = load_box_config(args.box_config)

    print(f"\nRunning bundle adjustment …")
    refined_cfg = calibrate_box_markers(
        image_paths=image_paths,
        intrinsics=intrinsics,
        box_cfg=box_cfg,
        min_markers=args.min_markers,
        max_reproj_px=args.max_reproj_px,
        debug_dir=args.debug_dir,
    )

    # Load raw YAML (no numpy arrays) and patch only corners_box_frame
    raw_cfg = load_yaml(args.box_config)
    id_to_corners = {
        int(m["id"]): m["corners_box_frame"]
        for m in refined_cfg["markers"]
    }
    for raw_m in raw_cfg["markers"]:
        mid = int(raw_m["id"])
        if mid in id_to_corners:
            raw_m["corners_box_frame"] = id_to_corners[mid]

    save_yaml(raw_cfg, args.output)
    print(f"\nRefined config written to {args.output}")


if __name__ == "__main__":
    main()
