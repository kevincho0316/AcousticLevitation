"""
Uncertainty quantification for the 3D ball position measurement.

Five error sources, propagated independently to 3D via Monte Carlo Jacobian;
total covariance is the sum (assuming independence between sources).

Sources:
  1. intrinsic_calibration  — residual reprojection error from lens calibration
  2. marker_position        — manufacturing/printing uncertainty of ArUco corners
  3. aruco_corner_detection — noise in detected marker corners
  4. box_pose_estimation    — extrinsic reprojection error → uncertainty in T_cam_box
  5. ball_detection_and_geometry — 2D ball center noise × triangulation GDOP
                                   (encoded in triangulation covariance Σ_3D)

Sources 1–4 are propagated numerically (perturb the relevant quantity — both
translation AND rotation — re-triangulate, measure the resulting 3D shift).
Source 5 comes directly from the triangulation covariance (JᵀWJ)⁻¹.

Two independent Monte Carlo validations are run:
  • mc_validate_triangulation — perturbs only 2D ball centers; validates source 5.
  • mc_validate_total         — perturbs all sources; validates the full budget.

Usage:
    python -m error_propagation.propagate \\
        --session sessions/session_001 \\
        --box-config config/box.yaml \\
        --cameras-config config/cameras.yaml \\
        --calibration-dir calibration/ \\
        --output sessions/session_001/error_budget.json \\
        [--n-mc 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    BallDetection2D, CameraIntrinsics, CameraPose,
    ErrorBudget, ErrorSource, TriangulationResult,
)
from common.io_utils import (
    load_box_config, load_cameras_config, load_intrinsics, load_json, save_json,
)
from common.se3_utils import _se3_exp
from triangulation.triangulate import (
    _dlt_triangulate, _projection_matrix, _triangulate_lm, load_detections_json, load_poses_json,
)


# ── Numerical Jacobian helper ─────────────────────────────────────────────────

def _perturb_and_triangulate(
    Ps: list[np.ndarray],
    uvs: list[np.ndarray],
    Sigma_invs: list[np.ndarray],
    X0: np.ndarray,
) -> np.ndarray:
    """Re-triangulate with given projection matrices and observations."""
    P_uv = list(zip(Ps, uvs))
    X_dlt = _dlt_triangulate(P_uv)
    X_opt, _ = _triangulate_lm(Ps, uvs, Sigma_invs, X_dlt)
    return X_opt


def _perturb_pose(T_cam_box: np.ndarray, sigma_t: float, sigma_r: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Apply random SE(3) perturbation: translation σ_t (m) + rotation σ_r (rad).

    _se3_exp convention: xi = [u (translation); omega (rotation)].
    """
    xi = np.zeros(6)
    xi[:3] = rng.normal(0, sigma_t, size=3)   # u — translation
    xi[3:] = rng.normal(0, sigma_r, size=3)   # omega — rotation
    return _se3_exp(xi) @ T_cam_box


# ── Source 1: Intrinsic calibration residual ──────────────────────────────────

def _source_intrinsic(
    intrinsics_map: dict[str, CameraIntrinsics],
    poses: dict[str, CameraPose],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> ErrorSource:
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    Ps_nom = [_projection_matrix(intrinsics_map[c], poses[c]) for c in camera_ids]
    uvs = [detections[c].center for c in camera_ids]
    Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]

    shifts: list[np.ndarray] = []
    for _ in range(n_mc):
        uvs_perturbed: list[np.ndarray] = []
        for c, intr in [(c, intrinsics_map[c]) for c in camera_ids]:
            sigma_px = intr.reprojection_error / np.sqrt(2.0)
            noise = rng.normal(0, sigma_px, size=2)
            uvs_perturbed.append(detections[c].center + noise)
        X_perturbed = _perturb_and_triangulate(Ps_nom, uvs_perturbed, Sigma_invs, X_nominal)
        shifts.append(X_perturbed - X_nominal)

    cov = np.cov(np.array(shifts).T, ddof=1) if len(shifts) >= 2 else np.zeros((3, 3))
    return ErrorSource(
        name="intrinsic_calibration",
        covariance_box=cov,
        description="Lens calibration residual reprojected to 3D",
    )


# ── Source 2: Marker position uncertainty ────────────────────────────────────

def _source_marker_position(
    box_cfg: dict,
    intrinsics_map: dict[str, CameraIntrinsics],
    poses: dict[str, CameraPose],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> ErrorSource:
    sigma_m = float(box_cfg.get("marker_position_uncertainty_m", 5e-4))
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    uvs = [detections[c].center for c in camera_ids]
    Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]

    shifts: list[np.ndarray] = []
    for _ in range(n_mc):
        Ps_perturbed: list[np.ndarray] = []
        for c in camera_ids:
            dist_approx = float(np.linalg.norm(poses[c].T_cam_box[:3, 3]))
            # Translation sigma from marker position error; rotation ≈ sigma_m / dist.
            sigma_r = sigma_m / max(dist_approx, 1e-3)
            T_pert = _perturb_pose(poses[c].T_cam_box, sigma_m, sigma_r, rng)
            Ps_perturbed.append(intrinsics_map[c].K @ T_pert[:3, :])
        X_perturbed = _perturb_and_triangulate(Ps_perturbed, uvs, Sigma_invs, X_nominal)
        shifts.append(X_perturbed - X_nominal)

    cov = np.cov(np.array(shifts).T, ddof=1) if len(shifts) >= 2 else np.zeros((3, 3))
    return ErrorSource(
        name="marker_position",
        covariance_box=cov,
        description=f"ArUco marker corner position uncertainty (σ = {sigma_m*1000:.2f} mm)",
    )


# ── Source 3: ArUco corner detection noise ────────────────────────────────────

def _source_aruco_corner(
    poses: dict[str, CameraPose],
    intrinsics_map: dict[str, CameraIntrinsics],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
    sigma_aruco_px: float = 0.2,
) -> ErrorSource:
    """ArUco corner detection noise propagated through pose → projection."""
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    uvs = [detections[c].center for c in camera_ids]
    Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]

    shifts: list[np.ndarray] = []
    for _ in range(n_mc):
        Ps_perturbed: list[np.ndarray] = []
        for c in camera_ids:
            focal = float(intrinsics_map[c].K[0, 0])
            dist_approx = float(np.linalg.norm(poses[c].T_cam_box[:3, 3]))
            # Translation: corner pixel noise → pose translation noise.
            sigma_t = sigma_aruco_px * dist_approx / focal
            # Rotation: corner pixel noise → angular noise.
            sigma_r = sigma_aruco_px / focal
            T_pert = _perturb_pose(poses[c].T_cam_box, sigma_t, sigma_r, rng)
            Ps_perturbed.append(intrinsics_map[c].K @ T_pert[:3, :])
        X_perturbed = _perturb_and_triangulate(Ps_perturbed, uvs, Sigma_invs, X_nominal)
        shifts.append(X_perturbed - X_nominal)

    cov = np.cov(np.array(shifts).T, ddof=1) if len(shifts) >= 2 else np.zeros((3, 3))
    return ErrorSource(
        name="aruco_corner_detection",
        covariance_box=cov,
        description=f"ArUco corner detection noise (σ ≈ {sigma_aruco_px} px)",
    )


# ── Source 4: Box pose estimation residual ────────────────────────────────────

def _perturb_pose_from_cov(T_cam_box: np.ndarray, pose_cov_6x6: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Sample SE(3) perturbation from real 6×6 pose covariance (rvec/tvec ordering)."""
    xi_rv_tv = rng.multivariate_normal(np.zeros(6), pose_cov_6x6)
    # solvePnP Jacobian columns: first 3 = rvec, next 3 = tvec.
    # _se3_exp convention: xi[:3] = translation (u), xi[3:] = rotation (omega).
    xi = np.concatenate([xi_rv_tv[3:6], xi_rv_tv[0:3]])   # reorder to [tvec, rvec]
    return _se3_exp(xi) @ T_cam_box


def _source_box_pose(
    poses: dict[str, CameraPose],
    intrinsics_map: dict[str, CameraIntrinsics],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> ErrorSource:
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    uvs = [detections[c].center for c in camera_ids]
    Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]

    shifts: list[np.ndarray] = []
    for _ in range(n_mc):
        Ps_perturbed: list[np.ndarray] = []
        for c in camera_ids:
            pose = poses[c]
            if pose.pose_covariance is not None:
                T_pert = _perturb_pose_from_cov(pose.T_cam_box, pose.pose_covariance, rng)
            else:
                # Heuristic fallback when no real covariance is available.
                focal = float(intrinsics_map[c].K[0, 0])
                dist_approx = float(np.linalg.norm(pose.T_cam_box[:3, 3]))
                reproj = pose.reprojection_error
                sigma_t = reproj * dist_approx / focal
                sigma_r = reproj / focal
                T_pert = _perturb_pose(pose.T_cam_box, sigma_t, sigma_r, rng)
            Ps_perturbed.append(intrinsics_map[c].K @ T_pert[:3, :])
        X_perturbed = _perturb_and_triangulate(Ps_perturbed, uvs, Sigma_invs, X_nominal)
        shifts.append(X_perturbed - X_nominal)

    cov = np.cov(np.array(shifts).T, ddof=1) if len(shifts) >= 2 else np.zeros((3, 3))
    return ErrorSource(
        name="box_pose_estimation",
        covariance_box=cov,
        description="Camera pose estimation residual (extrinsic reprojection error)",
    )


# ── Source 5: Ball detection noise + geometric dilution ──────────────────────

def _source_ball_detection_and_geometry(
    triangulation_result: TriangulationResult,
) -> ErrorSource:
    """Ball detection noise amplified by triangulation geometry (GDOP).

    The triangulation covariance (JᵀWJ)⁻¹ already encodes both the 2D
    ball-center noise (Σ_2D = sample_cov / N) and the geometric dilution
    from the camera arrangement.  This is reported as one combined source.
    """
    return ErrorSource(
        name="ball_detection_and_geometry",
        covariance_box=triangulation_result.covariance_box.copy(),
        description=(
            "Ball center 2D noise (averaged over N frames) × triangulation GDOP "
            "(both encoded in Σ_3D = (JᵀWJ)⁻¹)"
        ),
    )


# ── Monte Carlo validation ────────────────────────────────────────────────────

def mc_validate_triangulation(
    intrinsics_map: dict[str, CameraIntrinsics],
    poses: dict[str, CameraPose],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    triangulation_cov: np.ndarray,
    n_mc: int = 500,
    seed: int = 42,
) -> dict:
    """Validate source 5: perturb only 2D ball centers; compare to triangulation Σ_3D."""
    rng = np.random.default_rng(seed)
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]

    mc_positions: list[np.ndarray] = []
    for _ in range(n_mc):
        uvs_p: list[np.ndarray] = []
        for c in camera_ids:
            det = detections[c]
            noise = rng.multivariate_normal(np.zeros(2), det.covariance)
            uvs_p.append(det.center + noise)
        Ps = [_projection_matrix(intrinsics_map[c], poses[c]) for c in camera_ids]
        X_p = _perturb_and_triangulate(Ps, uvs_p, Sigma_invs, X_nominal)
        mc_positions.append(X_p)

    mc_cov = np.cov(np.array(mc_positions).T, ddof=1)
    mc_std = np.sqrt(np.diag(mc_cov)) * 1000
    anal_std = np.sqrt(np.diag(triangulation_cov)) * 1000
    frob = float(np.linalg.norm(mc_cov - triangulation_cov) / np.linalg.norm(triangulation_cov))

    print(f"\nMC validation — triangulation covariance ({n_mc} trials, 2D noise only):")
    print(f"  MC std (mm):         {mc_std}")
    print(f"  Analytical std (mm): {anal_std}")
    print(f"  Frobenius ratio:     {frob:.4f}  (< 0.3 = good agreement)")

    return {
        "label": "triangulation_covariance",
        "n_trials": n_mc,
        "mc_covariance_m2": mc_cov.tolist(),
        "reference_covariance_m2": triangulation_cov.tolist(),
        "mc_std_mm": mc_std.tolist(),
        "reference_std_mm": anal_std.tolist(),
        "frobenius_ratio": frob,
    }


def mc_validate_total(
    box_cfg: dict,
    intrinsics_map: dict[str, CameraIntrinsics],
    poses: dict[str, CameraPose],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    total_cov: np.ndarray,
    sigma_aruco_px: float = 0.2,
    n_mc: int = 500,
    seed: int = 99,
) -> dict:
    """Validate full budget: perturb all sources simultaneously; compare to total Σ."""
    rng = np.random.default_rng(seed)
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    sigma_m = float(box_cfg.get("marker_position_uncertainty_m", 5e-4))

    mc_positions: list[np.ndarray] = []
    for _ in range(n_mc):
        # Perturb 2D centers (ball detection noise).
        uvs_p: list[np.ndarray] = []
        Ps_p: list[np.ndarray] = []
        for c in camera_ids:
            det = detections[c]
            intr = intrinsics_map[c]
            pose = poses[c]
            focal = float(intr.K[0, 0])
            dist_approx = float(np.linalg.norm(pose.T_cam_box[:3, 3]))
            reproj = pose.reprojection_error

            # 2D ball center noise.
            noise_uv = rng.multivariate_normal(np.zeros(2), det.covariance)
            uvs_p.append(det.center + noise_uv)

            # Intrinsic calibration noise (perturbs observation).
            sigma_intr = intr.reprojection_error / np.sqrt(2.0)
            uvs_p[-1] = uvs_p[-1] + rng.normal(0, sigma_intr, size=2)

            # Pose perturbation: use real covariance if available, else combine heuristic sources.
            if pose.pose_covariance is not None:
                T_pert = _perturb_pose_from_cov(pose.T_cam_box, pose.pose_covariance, rng)
            else:
                sigma_t_mk = sigma_m
                sigma_r_mk = sigma_m / max(dist_approx, 1e-3)
                sigma_t_ac = sigma_aruco_px * dist_approx / focal
                sigma_r_ac = sigma_aruco_px / focal
                sigma_t_bp = reproj * dist_approx / focal
                sigma_r_bp = reproj / focal
                sigma_t_total = np.sqrt(sigma_t_mk**2 + sigma_t_ac**2 + sigma_t_bp**2)
                sigma_r_total = np.sqrt(sigma_r_mk**2 + sigma_r_ac**2 + sigma_r_bp**2)
                T_pert = _perturb_pose(pose.T_cam_box, sigma_t_total, sigma_r_total, rng)
            Ps_p.append(intr.K @ T_pert[:3, :])

        Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]
        X_p = _perturb_and_triangulate(Ps_p, uvs_p, Sigma_invs, X_nominal)
        mc_positions.append(X_p)

    mc_cov = np.cov(np.array(mc_positions).T, ddof=1)
    mc_std = np.sqrt(np.diag(mc_cov)) * 1000
    anal_std = np.sqrt(np.diag(total_cov)) * 1000
    frob = float(np.linalg.norm(mc_cov - total_cov) / np.linalg.norm(total_cov))

    print(f"\nMC validation — total budget ({n_mc} trials, all sources):")
    print(f"  MC std (mm):         {mc_std}")
    print(f"  Analytical std (mm): {anal_std}")
    print(f"  Frobenius ratio:     {frob:.4f}  (< 0.5 = acceptable agreement)")

    return {
        "label": "total_budget",
        "n_trials": n_mc,
        "mc_covariance_m2": mc_cov.tolist(),
        "reference_covariance_m2": total_cov.tolist(),
        "mc_std_mm": mc_std.tolist(),
        "reference_std_mm": anal_std.tolist(),
        "frobenius_ratio": frob,
    }


# ── Budget assembly ───────────────────────────────────────────────────────────

def propagate_errors(
    triangulation_result: TriangulationResult,
    poses: dict[str, CameraPose],
    intrinsics_map: dict[str, CameraIntrinsics],
    detections: dict[str, BallDetection2D],
    box_cfg: dict,
    n_mc: int = 500,
    seed: int = 0,
) -> ErrorBudget:
    X = triangulation_result.position_box
    rng = np.random.default_rng(seed)

    print("\nPropagating errors …")
    sources: list[ErrorSource] = []

    src1 = _source_intrinsic(intrinsics_map, poses, detections, X, n_mc, rng)
    sources.append(src1)
    print(f"  intrinsic_calibration std (mm):         {np.sqrt(np.diag(src1.covariance_box))*1000}")

    src2 = _source_marker_position(box_cfg, intrinsics_map, poses, detections, X, n_mc, rng)
    sources.append(src2)
    print(f"  marker_position std (mm):               {np.sqrt(np.diag(src2.covariance_box))*1000}")

    src3 = _source_aruco_corner(poses, intrinsics_map, detections, X, n_mc, rng)
    sources.append(src3)
    print(f"  aruco_corner_detection std (mm):        {np.sqrt(np.diag(src3.covariance_box))*1000}")

    src4 = _source_box_pose(poses, intrinsics_map, detections, X, n_mc, rng)
    sources.append(src4)
    print(f"  box_pose_estimation std (mm):           {np.sqrt(np.diag(src4.covariance_box))*1000}")

    src5 = _source_ball_detection_and_geometry(triangulation_result)
    sources.append(src5)
    print(f"  ball_detection_and_geometry std (mm):   {np.sqrt(np.diag(src5.covariance_box))*1000}")

    # Total: five independent sources.
    total_cov = (
        src1.covariance_box
        + src2.covariance_box
        + src3.covariance_box
        + src4.covariance_box
        + triangulation_result.covariance_box
    )
    print(f"\nTotal uncertainty std (mm): {np.sqrt(np.diag(total_cov))*1000}")

    return ErrorBudget(sources=sources, total_covariance=total_cov)


# ── Session-level propagation ─────────────────────────────────────────────────

def propagate_session(
    session_dir: Path,
    box_config_path: Path,
    cameras_config_path: Path,
    calibration_dir: Path,
    output_path: Path,
    n_mc: int = 500,
) -> ErrorBudget:
    box_cfg = load_box_config(box_config_path)
    cam_cfg = load_cameras_config(cameras_config_path)

    intrinsics_map: dict[str, CameraIntrinsics] = {}
    for cam in cam_cfg["cameras"]:
        cam_id = cam["id"]
        intr_path = calibration_dir / Path(cam["intrinsics_file"]).name
        if intr_path.exists():
            intrinsics_map[cam_id] = load_intrinsics(intr_path)

    poses = load_poses_json(session_dir / "extrinsics.json")
    detections = load_detections_json(session_dir / "ball_detections.json")

    tri_data = load_json(session_dir / "triangulation.json")
    tri_result = TriangulationResult(
        position_box=np.array(tri_data["position_box_m"]),
        covariance_box=np.array(tri_data["covariance_box_m2"]),
        reprojection_residuals={
            k: np.array(v) for k, v in tri_data["reprojection_residuals_px"].items()
        },
        n_cameras=tri_data["n_cameras"],
    )

    budget = propagate_errors(tri_result, poses, intrinsics_map, detections, box_cfg, n_mc=n_mc)

    mc_tri = mc_validate_triangulation(
        intrinsics_map, poses, detections,
        tri_result.position_box, tri_result.covariance_box, n_mc=n_mc,
    )
    mc_tot = mc_validate_total(
        box_cfg, intrinsics_map, poses, detections,
        tri_result.position_box, budget.total_covariance, n_mc=n_mc,
    )

    result_json = {
        "total_covariance_m2": budget.total_covariance.tolist(),
        "total_std_mm": (np.sqrt(np.diag(budget.total_covariance)) * 1000).tolist(),
        "sources": [
            {
                "name": s.name,
                "description": s.description,
                "covariance_m2": s.covariance_box.tolist(),
                "std_mm": (np.sqrt(np.diag(s.covariance_box)) * 1000).tolist(),
            }
            for s in budget.sources
        ],
        "monte_carlo_validation": [mc_tri, mc_tot],
    }
    save_json(result_json, output_path)
    print(f"\nError budget saved to {output_path}")
    return budget


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error propagation and uncertainty quantification")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--box-config", type=Path, default=Path("config/box.yaml"))
    p.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--n-mc", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (args.session / "error_budget.json")
    propagate_session(
        session_dir=args.session,
        box_config_path=args.box_config,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=output,
        n_mc=args.n_mc,
    )


if __name__ == "__main__":
    main()
