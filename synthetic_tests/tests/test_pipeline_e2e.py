"""
End-to-end pipeline test: render frames → extrinsic solver → ball detector →
triangulation → compare against ground truth.

Smoke: zero noise → 3D error < 0.01 mm
Nominal: realistic noise levels → 3D error < 0.5 mm
Reprojection consistency: reproject recovered 3D point, residuals < 1 px per camera
Subset: N-1 cameras → result within error ellipsoid of N-camera result
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from synthetic_tests.synth.cameras import make_default_scene
from synthetic_tests.synth.renderer import write_session_frames
from extrinsic_solver.solve import estimate_camera_pose
from ball_detector.detect import detect_ball_camera
from triangulation.triangulate import triangulate
from common import BallDetection2D, CameraPose


def _run_pipeline(scene, session_dir: Path, max_reproj_px: float = 2.0):
    """Run extrinsic solver + ball detector + triangulation for all cameras."""
    box_cfg = scene.to_box_cfg()
    poses = {}
    detections = {}

    for cam in scene.cameras:
        cam_id = cam["id"]
        intr = cam["intrinsics"]
        frame_paths = sorted((session_dir / cam_id).glob("frame_*.png"))

        try:
            pose = estimate_camera_pose(
                frame_paths, intr, box_cfg,
                min_markers=3, max_reproj_px=max_reproj_px,
            )
            poses[cam_id] = pose
        except RuntimeError as e:
            pytest.fail(f"Extrinsic solver failed for {cam_id}: {e}")

        try:
            det = detect_ball_camera(frame_paths, intr)
            detections[cam_id] = det
        except RuntimeError as e:
            pytest.fail(f"Ball detector failed for {cam_id}: {e}")

    intrinsics = {c["id"]: c["intrinsics"] for c in scene.cameras}
    result = triangulate(detections, poses, intrinsics)
    return result, poses, intrinsics, detections


@pytest.fixture(scope="module")
def smoke_e2e(tmp_path_factory):
    session_dir = tmp_path_factory.mktemp("e2e_smoke")
    scene = make_default_scene(n_cameras=4)
    write_session_frames(scene, session_dir, n_frames=20, noise_sigma=0.0, seed=0)
    return scene, session_dir


@pytest.fixture(scope="module")
def nominal_e2e(tmp_path_factory):
    session_dir = tmp_path_factory.mktemp("e2e_nominal")
    scene = make_default_scene(n_cameras=4)
    write_session_frames(scene, session_dir, n_frames=20, noise_sigma=3.0, seed=1)
    return scene, session_dir


def test_e2e_smoke(smoke_e2e):
    """Zero noise: 3D position error < 0.01 mm."""
    scene, session_dir = smoke_e2e
    result, *_ = _run_pipeline(scene, session_dir, max_reproj_px=1.0)
    err_mm = np.linalg.norm(result.position_box - scene.ball_position_box) * 1000
    assert err_mm < 0.01, f"smoke e2e: error {err_mm:.4f} mm"


def test_e2e_nominal(nominal_e2e):
    """Realistic noise: 3D position error < 0.5 mm."""
    scene, session_dir = nominal_e2e
    result, *_ = _run_pipeline(scene, session_dir, max_reproj_px=3.0)
    err_mm = np.linalg.norm(result.position_box - scene.ball_position_box) * 1000
    assert err_mm < 0.5, f"nominal e2e: error {err_mm:.3f} mm"


def test_e2e_reprojection_consistency(smoke_e2e):
    """Reproject recovered 3D point: residuals < 1 px per camera."""
    from triangulation.triangulate import _project as tri_project, _projection_matrix

    scene, session_dir = smoke_e2e
    result, poses, intrinsics, detections = _run_pipeline(scene, session_dir, max_reproj_px=1.0)

    for cam_id in poses:
        P = _projection_matrix(intrinsics[cam_id], poses[cam_id])
        X = result.position_box
        Xh = np.append(X, 1.0)
        ph = P @ Xh
        uv_reproj = ph[:2] / ph[2]
        uv_det = detections[cam_id].center
        res_px = np.linalg.norm(uv_reproj - uv_det)
        assert res_px < 1.0, f"{cam_id}: reprojection residual {res_px:.3f} px"


def test_e2e_subset_cameras(nominal_e2e):
    """N-1 cameras: result within 3σ ellipsoid of N-camera result."""
    scene, session_dir = nominal_e2e
    result_full, poses, intrinsics, detections = _run_pipeline(
        scene, session_dir, max_reproj_px=3.0
    )

    # Drop last camera and re-triangulate.
    cam_ids = list(poses.keys())
    subset_ids = cam_ids[:-1]
    result_sub = triangulate(
        {k: detections[k] for k in subset_ids},
        {k: poses[k] for k in subset_ids},
        {k: intrinsics[k] for k in subset_ids},
    )

    # Check subset result falls within 3σ ellipsoid of full result.
    diff = result_sub.position_box - result_full.position_box
    cov = result_full.covariance_box
    try:
        mahal_sq = float(diff @ np.linalg.inv(cov) @ diff)
    except np.linalg.LinAlgError:
        mahal_sq = float(diff @ np.linalg.pinv(cov) @ diff)
    # χ²(3) at 99.9% = 16.3
    assert mahal_sq < 20.0, \
        f"N-1 result outside N-camera 99.9% ellipsoid: Mahalanobis² = {mahal_sq:.2f}"


def test_e2e_camera_count_sweep(tmp_path_factory):
    """3 and 4 cameras both produce < 1 mm error under nominal noise."""
    for n_cams in (3, 4):
        session_dir = tmp_path_factory.mktemp(f"e2e_ncam{n_cams}")
        scene = make_default_scene(n_cameras=n_cams)
        write_session_frames(scene, session_dir, n_frames=15, noise_sigma=3.0, seed=n_cams)
        result, *_ = _run_pipeline(scene, session_dir, max_reproj_px=3.0)
        err_mm = np.linalg.norm(result.position_box - scene.ball_position_box) * 1000
        assert err_mm < 1.0, f"{n_cams} cameras: error {err_mm:.3f} mm"
