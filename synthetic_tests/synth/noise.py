"""Noise models for synthetic test perturbations."""
from __future__ import annotations

import numpy as np

from common import BallDetection2D, CameraIntrinsics, CameraPose


def noisy_ball_detection(
    scene,
    cam_id: str,
    sigma_px: float = 0.3,
    n_frames: int = 100,
    rng: np.random.Generator | None = None,
) -> BallDetection2D:
    """Return a BallDetection2D at the true projected ball center plus Gaussian noise."""
    import cv2
    from synthetic_tests.synth.renderer import _project

    if rng is None:
        rng = np.random.default_rng()

    cam = next(c for c in scene.cameras if c["id"] == cam_id)
    K = cam["intrinsics"].K
    T = cam["T_cam_box"]
    uv_true, _ = _project(K, T, scene.ball_position_box)

    # Simulate n_frames detections with per-frame noise.
    centers = rng.normal(uv_true, sigma_px, size=(n_frames, 2))
    mean_center = centers.mean(axis=0)
    cov_mean = np.cov(centers.T, ddof=1) / n_frames   # covariance of the mean

    return BallDetection2D(
        camera_id=cam_id,
        center=mean_center,
        covariance=cov_mean,
        n_frames_accepted=n_frames,
        n_frames_rejected=0,
        per_frame_centers=centers,
    )


def perturb_pose(
    pose: CameraPose,
    sigma_t_m: float = 1e-4,
    sigma_r_rad: float = 5e-4,
    rng: np.random.Generator | None = None,
) -> CameraPose:
    """Return a copy of pose with small random translation and rotation perturbation."""
    if rng is None:
        rng = np.random.default_rng()
    T = pose.T_cam_box.copy()
    T[:3, 3] += rng.normal(0, sigma_t_m, 3)
    # Small-angle rotation perturbation via skew-symmetric matrix.
    omega = rng.normal(0, sigma_r_rad, 3)
    theta = np.linalg.norm(omega)
    if theta > 1e-12:
        K_skew = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ])
        R_perturb = np.eye(3) + np.sin(theta) / theta * K_skew + \
                    (1 - np.cos(theta)) / theta**2 * (K_skew @ K_skew)
        T[:3, :3] = R_perturb @ T[:3, :3]
    return CameraPose(
        camera_id=pose.camera_id,
        T_cam_box=T,
        reprojection_error=pose.reprojection_error,
        n_markers_used=pose.n_markers_used,
        n_frames_used=pose.n_frames_used,
    )
