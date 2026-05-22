"""
3D ball position reconstruction via multi-view triangulation.

Algorithm:
  1. DLT (Direct Linear Transform) for an initial estimate from all cameras.
  2. Nonlinear refinement by Levenberg–Marquardt, minimizing weighted reprojection
     error. Each camera's residual is weighted by the inverse of its 2D covariance
     (Mahalanobis distance).
  3. 3D covariance is recovered from the Jacobian at the optimum:
         Σ_3D = (Jᵀ W J)⁻¹
     where W is the block-diagonal weight matrix from 2D covariances.

Usage:
    python -m triangulation.triangulate \\
        --session sessions/session_001 \\
        --cameras-config config/cameras.yaml \\
        --calibration-dir calibration/ \\
        --output sessions/session_001/triangulation.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BallDetection2D, CameraIntrinsics, CameraPose, TriangulationResult
from common.io_utils import load_cameras_config, load_intrinsics, load_json, save_json


# ── Projection helpers ────────────────────────────────────────────────────────

def _projection_matrix(intrinsics: CameraIntrinsics, pose: CameraPose) -> np.ndarray:
    """Return 3×4 projection matrix P = K @ T_cam_box[:3, :]."""
    return intrinsics.K @ pose.T_cam_box[:3, :]


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D point X (3,) through P (3×4) → (2,) pixel coordinates."""
    Xh = np.append(X, 1.0)
    ph = P @ Xh
    return ph[:2] / ph[2]


# ── DLT ──────────────────────────────────────────────────────────────────────

def _dlt_triangulate(cameras: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """DLT from a list of (P 3×4, uv 2) pairs. Returns 3D point in box frame."""
    rows: list[np.ndarray] = []
    for P, uv in cameras:
        u, v = float(uv[0]), float(uv[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    return X_h[:3] / X_h[3]


# ── Weighted nonlinear refinement ─────────────────────────────────────────────

def _residuals_weighted(
    X: np.ndarray,
    Ps: list[np.ndarray],
    uvs: list[np.ndarray],
    Sigma_invs: list[np.ndarray],
) -> np.ndarray:
    """Weighted reprojection residuals for LM. Returns flat array of whitened residuals."""
    res: list[np.ndarray] = []
    for P, uv, Sigma_inv in zip(Ps, uvs, Sigma_invs):
        proj = _project(P, X)
        diff = proj - uv
        # Whiten: L^T @ diff where L L^T = Sigma_inv (Cholesky of inverse)
        try:
            L = np.linalg.cholesky(Sigma_inv)
            whitened = L.T @ diff
        except np.linalg.LinAlgError:
            # Fallback: use diagonal scaling
            whitened = diff * np.sqrt(np.diag(Sigma_inv))
        res.extend(whitened.tolist())
    return np.array(res)


def _triangulate_lm(
    Ps: list[np.ndarray],
    uvs: list[np.ndarray],
    Sigma_invs: list[np.ndarray],
    X0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """LM refinement. Returns (X_refined, covariance_3D)."""
    result = least_squares(
        _residuals_weighted,
        X0,
        args=(Ps, uvs, Sigma_invs),
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    X_opt = result.x

    # 3D covariance from Jacobian: Σ_3D = (Jᵀ W J)⁻¹
    # Since residuals are already whitened, JᵀJ ≈ JᵀWJ
    J = result.jac  # (2*N_cameras, 3)
    JtJ = J.T @ J
    try:
        cov = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JtJ)

    return X_opt, cov


# ── Main triangulation ────────────────────────────────────────────────────────

def triangulate(
    detections: dict[str, BallDetection2D],
    poses: dict[str, CameraPose],
    intrinsics_map: dict[str, CameraIntrinsics],
) -> TriangulationResult:
    if len(detections) < 2:
        raise RuntimeError(f"Need ≥ 2 cameras for triangulation; got {len(detections)}.")

    camera_ids = [cid for cid in detections if cid in poses and cid in intrinsics_map]
    if len(camera_ids) < 2:
        raise RuntimeError("Fewer than 2 cameras have both detections and poses.")

    Ps: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    Sigma_invs: list[np.ndarray] = []
    P_uv_list: list[tuple[np.ndarray, np.ndarray]] = []

    for cid in camera_ids:
        intr = intrinsics_map[cid]
        pose = poses[cid]
        det = detections[cid]

        P = _projection_matrix(intr, pose)
        uv = det.center
        Sigma = det.covariance.copy()  # (2,2) covariance of the mean

        # Inflate 2D weight by pose uncertainty: Σ_eff = Σ_2D + J_pose Σ_pose J_pose^T.
        if pose.pose_covariance is not None:
            import cv2 as _cv2
            K = intr.K.astype(np.float64)
            R = pose.T_cam_box[:3, :3]
            t = pose.T_cam_box[:3, 3]
            rvec_pose, _ = _cv2.Rodrigues(R)
            tvec_pose = t.copy()
            # Back-project uv at the camera–box translation distance to get a
            # representative 3D point for computing the Jacobian.
            Z_approx = float(np.linalg.norm(t))
            uv_h = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
            X_approx = (R.T @ (Z_approx * uv_h - t)).reshape(1, 3).astype(np.float64)
            zeros_dist = np.zeros(5, dtype=np.float64)
            _, jac_pnp = _cv2.projectPoints(X_approx, rvec_pose, tvec_pose, K, zeros_dist)
            J_pose_2x6 = jac_pnp[0, :, :6].astype(np.float64)  # (2,6) — columns: rvec then tvec
            Sigma = Sigma + J_pose_2x6 @ pose.pose_covariance @ J_pose_2x6.T

        # Regularize to ensure invertibility.
        Sigma_reg = Sigma + np.eye(2) * 1e-6
        Sigma_inv = np.linalg.inv(Sigma_reg)

        Ps.append(P)
        uvs.append(uv)
        Sigma_invs.append(Sigma_inv)
        P_uv_list.append((P, uv))

    X0 = _dlt_triangulate(P_uv_list)
    X_opt, cov = _triangulate_lm(Ps, uvs, Sigma_invs, X0)

    # Compute per-camera reprojection residuals at the optimum.
    residuals: dict[str, np.ndarray] = {}
    for cid, P, uv in zip(camera_ids, Ps, uvs):
        proj = _project(P, X_opt)
        residuals[cid] = proj - uv

    mean_reproj = float(np.mean([np.linalg.norm(r) for r in residuals.values()]))
    print(f"Triangulation: {len(camera_ids)} cameras, mean reprojection error = {mean_reproj:.4f} px")
    for cid, r in residuals.items():
        print(f"  {cid}: ({r[0]:+.4f}, {r[1]:+.4f}) px")

    return TriangulationResult(
        position_box=X_opt,
        covariance_box=cov,
        reprojection_residuals=residuals,
        n_cameras=len(camera_ids),
    )


# ── Load poses from JSON ──────────────────────────────────────────────────────

def load_poses_json(path: Path) -> dict[str, CameraPose]:
    data = load_json(path)
    poses: dict[str, CameraPose] = {}
    for cam_id, entry in data["poses"].items():
        pose_cov = entry.get("pose_covariance")
        poses[cam_id] = CameraPose(
            camera_id=cam_id,
            T_cam_box=np.array(entry["T_cam_box"]),
            reprojection_error=entry["reprojection_error_px"],
            n_markers_used=0,
            n_frames_used=entry.get("n_frames_used", 0),
            pose_covariance=np.array(pose_cov) if pose_cov is not None else None,
        )
    return poses


def load_detections_json(path: Path) -> dict[str, BallDetection2D]:
    data = load_json(path)
    detections: dict[str, BallDetection2D] = {}
    for cam_id, entry in data["detections"].items():
        detections[cam_id] = BallDetection2D(
            camera_id=cam_id,
            center=np.array(entry["center_uv"]),
            covariance=np.array(entry["covariance_uv"]),
            n_frames_accepted=entry["n_frames_accepted"],
            n_frames_rejected=entry["n_frames_rejected"],
        )
    return detections


# ── Session triangulation ─────────────────────────────────────────────────────

def triangulate_session(
    session_dir: Path,
    cameras_config_path: Path,
    calibration_dir: Path,
    output_path: Path,
) -> TriangulationResult:
    cam_cfg = load_cameras_config(cameras_config_path)

    intrinsics_map: dict[str, CameraIntrinsics] = {}
    for cam in cam_cfg["cameras"]:
        cam_id = cam["id"]
        intr_path = calibration_dir / Path(cam["intrinsics_file"]).name
        if intr_path.exists():
            intrinsics_map[cam_id] = load_intrinsics(intr_path)

    poses = load_poses_json(session_dir / "extrinsics.json")
    detections = load_detections_json(session_dir / "ball_detections.json")

    result = triangulate(detections, poses, intrinsics_map)

    result_json = {
        "position_box_m": result.position_box.tolist(),
        "covariance_box_m2": result.covariance_box.tolist(),
        "n_cameras": result.n_cameras,
        "reprojection_residuals_px": {
            k: v.tolist() for k, v in result.reprojection_residuals.items()
        },
    }
    save_json(result_json, output_path)
    print(f"\nTriangulation result saved to {output_path}")
    print(f"  Position (box frame, mm): {result.position_box * 1000}")
    print(f"  Std dev (mm): {np.sqrt(np.diag(result.covariance_box)) * 1000}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-view ball triangulation")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (args.session / "triangulation.json")
    triangulate_session(
        session_dir=args.session,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=output,
    )


if __name__ == "__main__":
    main()
