"""
Test triangulation.triangulate against synthetic perfect + noisy observations.

Smoke: perfect 2D observations → 3D error < 0.01 mm
Noisy: σ=0.3 px ball-centre noise, 100-frame average → 3D error < 0.5 mm
Grid: 5×5×5 ball positions inside box, nominal noise → error < 1.0 mm everywhere
Covariance: recovered Σ_3D is positive-definite and diagonal std matches MC estimate
"""
from __future__ import annotations

import numpy as np
import pytest

from synthetic_tests.synth.cameras import make_default_scene
from synthetic_tests.synth.noise import noisy_ball_detection
from synthetic_tests.synth.renderer import _project
from common import BallDetection2D, CameraIntrinsics, CameraPose
from triangulation.triangulate import triangulate


def _perfect_detections(scene) -> dict:
    detections = {}
    for cam in scene.cameras:
        K = cam["intrinsics"].K
        T = cam["T_cam_box"]
        uv, Z = _project(K, T, scene.ball_position_box)
        assert Z > 0
        detections[cam["id"]] = BallDetection2D(
            camera_id=cam["id"],
            center=uv,
            covariance=np.eye(2) * 1e-8,
            n_frames_accepted=1,
            n_frames_rejected=0,
        )
    return detections


def _scene_to_maps(scene):
    poses = {c["id"]: scene.to_camera_pose(c["id"]) for c in scene.cameras}
    intrinsics = {c["id"]: c["intrinsics"] for c in scene.cameras}
    return poses, intrinsics


def test_triangulation_smoke(default_scene):
    """Perfect 2D observations → 3D position error < 0.01 mm."""
    detections = _perfect_detections(default_scene)
    poses, intrinsics = _scene_to_maps(default_scene)
    result = triangulate(detections, poses, intrinsics)
    err_mm = np.linalg.norm(result.position_box - default_scene.ball_position_box) * 1000
    assert err_mm < 0.01, f"smoke: error {err_mm:.4f} mm"


def test_triangulation_noisy(default_scene):
    """σ=0.3 px noise, 100-frame average → error < 0.5 mm."""
    rng = np.random.default_rng(0)
    detections = {
        c["id"]: noisy_ball_detection(default_scene, c["id"], sigma_px=0.3,
                                       n_frames=100, rng=rng)
        for c in default_scene.cameras
    }
    poses, intrinsics = _scene_to_maps(default_scene)
    result = triangulate(detections, poses, intrinsics)
    err_mm = np.linalg.norm(result.position_box - default_scene.ball_position_box) * 1000
    assert err_mm < 0.5, f"noisy: error {err_mm:.3f} mm"


@pytest.mark.parametrize("ix,iy,iz", [
    (i, j, k) for i in range(3) for j in range(2) for k in range(3)
])
def test_triangulation_grid(ix, iy, iz):
    """3D error < 1.0 mm at grid positions throughout the box volume."""
    W, D, H = 0.12, 0.12, 0.06
    xs = np.linspace(0.02, W - 0.02, 3)
    ys = np.linspace(0.01, H - 0.01, 2)
    zs = np.linspace(0.02, D - 0.02, 3)
    ball_pos = np.array([xs[ix], ys[iy], zs[iz]])

    scene = make_default_scene(ball_position_box=ball_pos)
    rng = np.random.default_rng(ix * 100 + iy * 10 + iz)
    detections = {
        c["id"]: noisy_ball_detection(scene, c["id"], sigma_px=0.3,
                                       n_frames=100, rng=rng)
        for c in scene.cameras
    }
    poses, intrinsics = _scene_to_maps(scene)
    result = triangulate(detections, poses, intrinsics)
    err_mm = np.linalg.norm(result.position_box - ball_pos) * 1000
    assert err_mm < 1.0, f"grid ({ix},{iy},{iz}): error {err_mm:.3f} mm"


def test_triangulation_covariance_positive_definite(default_scene):
    """Recovered Σ_3D is symmetric and positive definite."""
    rng = np.random.default_rng(7)
    detections = {
        c["id"]: noisy_ball_detection(default_scene, c["id"], sigma_px=0.3,
                                       n_frames=100, rng=rng)
        for c in default_scene.cameras
    }
    poses, intrinsics = _scene_to_maps(default_scene)
    result = triangulate(detections, poses, intrinsics)
    cov = result.covariance_box
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T, atol=1e-12), "covariance not symmetric"
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), f"covariance not PD, min eigval={eigvals.min():.2e}"


def test_triangulation_covariance_mc_agreement(default_scene):
    """MC empirical covariance Frobenius ratio vs Σ_3D < 0.5 (500 trials)."""
    rng = np.random.default_rng(99)
    sigma_px = 0.3
    n_frames = 100
    n_mc = 500
    poses, intrinsics = _scene_to_maps(default_scene)

    # Nominal result for reference covariance.
    det_nom = {
        c["id"]: noisy_ball_detection(default_scene, c["id"], sigma_px=sigma_px,
                                       n_frames=n_frames, rng=np.random.default_rng(0))
        for c in default_scene.cameras
    }
    res_nom = triangulate(det_nom, poses, intrinsics)
    analytical_cov = res_nom.covariance_box

    mc_positions = []
    for _ in range(n_mc):
        det = {
            c["id"]: noisy_ball_detection(default_scene, c["id"], sigma_px=sigma_px,
                                           n_frames=n_frames, rng=rng)
            for c in default_scene.cameras
        }
        res = triangulate(det, poses, intrinsics)
        mc_positions.append(res.position_box)

    mc_cov = np.cov(np.array(mc_positions).T, ddof=1)
    ratio = np.linalg.norm(mc_cov - analytical_cov) / np.linalg.norm(analytical_cov)
    assert ratio < 0.5, f"MC vs analytical Frobenius ratio {ratio:.3f} ≥ 0.5"
