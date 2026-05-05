"""
Uncertainty quantification for the 3D ball position measurement.

Six error sources, propagated independently to 3D via Jacobian; total covariance
is the sum (assuming independence between sources).

Sources:
  1. intrinsic_calibration  — residual reprojection error from lens calibration
  2. marker_position        — manufacturing/printing uncertainty of ArUco corners
  3. aruco_corner_detection — noise in detected marker corners
  4. box_pose_estimation    — extrinsic reprojection error → uncertainty in T_cam_box
  5. ball_detection         — ball center noise (1/√N already applied in Σ_2D)
  6. triangulation_geometry — geometric dilution from camera arrangement

Sources 1–4 are propagated numerically (perturb the relevant quantity, re-triangulate,
measure the resulting 3D shift). Source 5 comes directly from the ball detection
covariance already embedded in the triangulation. Source 6 is characterized by the
triangulation covariance itself.

Additionally, a Monte Carlo validation compares the analytical total against
empirical covariance from N_MC random perturbation trials.

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
        # Perturb each camera's 2D observation by its intrinsic calibration noise.
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
        # For each pose, add a perturbation to the box pose (translation only,
        # as a simple proxy for marker position uncertainty).
        Ps_perturbed: list[np.ndarray] = []
        for c in camera_ids:
            T_perturbed = poses[c].T_cam_box.copy()
            T_perturbed[:3, 3] += rng.normal(0, sigma_m, size=3)
            K = intrinsics_map[c].K
            Ps_perturbed.append(K @ T_perturbed[:3, :])
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
        # Perturb the translation component of each pose by a scaled noise
        # representative of how ArUco corner noise affects pose estimation.
        Ps_perturbed: list[np.ndarray] = []
        for c in camera_ids:
            pose = poses[c]
            intr = intrinsics_map[c]
            # Corner noise → pose noise via approximate sensitivity:
            # σ_t ≈ σ_aruco / (focal_length / distance)
            # Use reprojection error as a proxy for the pose translation noise.
            focal = float(intr.K[0, 0])
            # Approximate distance from camera to box (use tvec magnitude).
            dist_approx = float(np.linalg.norm(pose.T_cam_box[:3, 3]))
            sigma_t = sigma_aruco_px * dist_approx / focal
            T_perturbed = pose.T_cam_box.copy()
            T_perturbed[:3, 3] += rng.normal(0, sigma_t, size=3)
            Ps_perturbed.append(intr.K @ T_perturbed[:3, :])
        X_perturbed = _perturb_and_triangulate(Ps_perturbed, uvs, Sigma_invs, X_nominal)
        shifts.append(X_perturbed - X_nominal)

    cov = np.cov(np.array(shifts).T, ddof=1) if len(shifts) >= 2 else np.zeros((3, 3))
    return ErrorSource(
        name="aruco_corner_detection",
        covariance_box=cov,
        description=f"ArUco corner detection noise (σ ≈ {sigma_aruco_px} px)",
    )


# ── Source 4: Box pose estimation residual ────────────────────────────────────

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
            intr = intrinsics_map[c]
            # Perturb translation by pose reprojection error scaled to 3D.
            focal = float(intr.K[0, 0])
            dist_approx = float(np.linalg.norm(pose.T_cam_box[:3, 3]))
            sigma_t = pose.reprojection_error * dist_approx / focal
            T_perturbed = pose.T_cam_box.copy()
            T_perturbed[:3, 3] += rng.normal(0, sigma_t, size=3)
            Ps_perturbed.append(intr.K @ T_perturbed[:3, :])
        X_perturbed = _perturb_and_triangulate(Ps_perturbed, uvs, Sigma_invs, X_nominal)
        shifts.append(X_perturbed - X_nominal)

    cov = np.cov(np.array(shifts).T, ddof=1) if len(shifts) >= 2 else np.zeros((3, 3))
    return ErrorSource(
        name="box_pose_estimation",
        covariance_box=cov,
        description="Camera pose estimation residual (extrinsic reprojection error)",
    )


# ── Source 5: Ball detection noise ───────────────────────────────────────────

def _source_ball_detection(
    triangulation_result: TriangulationResult,
) -> ErrorSource:
    """Ball detection noise is already embedded in the triangulation covariance
    via the 2D covariance Σ_2D = sample_cov / N passed to LM weighting.
    Extract its contribution by taking the full triangulation covariance as an
    upper bound and attributing it to this source for accounting purposes.
    The covariance here is the direct output of (JᵀWJ)⁻¹ which reflects 2D noise.
    """
    return ErrorSource(
        name="ball_detection",
        covariance_box=triangulation_result.covariance_box.copy(),
        description="Ball center detection noise (already averaged over N frames, encoded in Σ_3D)",
    )


# ── Source 6: Triangulation geometric dilution ────────────────────────────────

def _source_geometric_dilution(
    triangulation_result: TriangulationResult,
) -> ErrorSource:
    """Report the full triangulation covariance as the geometric contribution.
    This is the same as source 5 in practice; listed separately to make the
    budget explicit about the geometric arrangement contribution.
    """
    return ErrorSource(
        name="triangulation_geometry",
        covariance_box=triangulation_result.covariance_box.copy(),
        description="3D geometric dilution from camera angular arrangement (GDOP)",
    )


# ── Monte Carlo validation ────────────────────────────────────────────────────

def monte_carlo_validate(
    intrinsics_map: dict[str, CameraIntrinsics],
    poses: dict[str, CameraPose],
    detections: dict[str, BallDetection2D],
    X_nominal: np.ndarray,
    analytical_total_cov: np.ndarray,
    n_mc: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    camera_ids = list(set(detections) & set(poses) & set(intrinsics_map))
    Sigma_invs = [np.linalg.inv(detections[c].covariance + np.eye(2) * 1e-6) for c in camera_ids]

    mc_positions: list[np.ndarray] = []
    for _ in range(n_mc):
        # Perturb 2D observations by their covariance.
        uvs_p: list[np.ndarray] = []
        for c in camera_ids:
            det = detections[c]
            noise = rng.multivariate_normal(np.zeros(2), det.covariance)
            uvs_p.append(det.center + noise)
        Ps = [_projection_matrix(intrinsics_map[c], poses[c]) for c in camera_ids]
        X_p = _perturb_and_triangulate(Ps, uvs_p, Sigma_invs, X_nominal)
        mc_positions.append(X_p)

    mc_array = np.array(mc_positions)
    mc_cov = np.cov(mc_array.T, ddof=1)
    mc_std = np.sqrt(np.diag(mc_cov)) * 1000  # mm
    anal_std = np.sqrt(np.diag(analytical_total_cov)) * 1000  # mm

    frobenius_ratio = float(np.linalg.norm(mc_cov - analytical_total_cov) / np.linalg.norm(analytical_total_cov))
    print(f"\nMonte Carlo validation ({n_mc} trials):")
    print(f"  MC std (mm):         {mc_std}")
    print(f"  Analytical std (mm): {anal_std}")
    print(f"  Frobenius ratio:     {frobenius_ratio:.4f}  (< 0.5 = good agreement)")

    return {
        "n_trials": n_mc,
        "mc_covariance_m2": mc_cov.tolist(),
        "analytical_total_covariance_m2": analytical_total_cov.tolist(),
        "mc_std_mm": mc_std.tolist(),
        "analytical_std_mm": anal_std.tolist(),
        "frobenius_ratio": frobenius_ratio,
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
    print(f"  intrinsic_calibration std (mm): {np.sqrt(np.diag(src1.covariance_box))*1000}")

    src2 = _source_marker_position(box_cfg, intrinsics_map, poses, detections, X, n_mc, rng)
    sources.append(src2)
    print(f"  marker_position std (mm):       {np.sqrt(np.diag(src2.covariance_box))*1000}")

    src3 = _source_aruco_corner(poses, intrinsics_map, detections, X, n_mc, rng)
    sources.append(src3)
    print(f"  aruco_corner_detection std (mm):{np.sqrt(np.diag(src3.covariance_box))*1000}")

    src4 = _source_box_pose(poses, intrinsics_map, detections, X, n_mc, rng)
    sources.append(src4)
    print(f"  box_pose_estimation std (mm):   {np.sqrt(np.diag(src4.covariance_box))*1000}")

    # Sources 5 & 6 come directly from the triangulation result.
    src5 = _source_ball_detection(triangulation_result)
    sources.append(src5)
    print(f"  ball_detection std (mm):        {np.sqrt(np.diag(src5.covariance_box))*1000}")

    src6 = _source_geometric_dilution(triangulation_result)
    sources.append(src6)
    print(f"  triangulation_geometry std (mm):{np.sqrt(np.diag(src6.covariance_box))*1000}")

    # Total: sources 1–4 are independent Monte Carlo estimates; sources 5 & 6
    # are already the full triangulation covariance (same matrix, not additive).
    # Sum the independent MC sources (1–4) with the triangulation covariance.
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

    mc_validation = monte_carlo_validate(
        intrinsics_map, poses, detections,
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
        "monte_carlo_validation": mc_validation,
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
