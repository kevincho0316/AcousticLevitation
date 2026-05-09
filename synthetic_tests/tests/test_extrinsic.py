"""
Test extrinsic_solver.solve against synthetic marker images.

Smoke: 10 identical noise-free frames → rotation error < 0.5°, translation error < 1 mm
Noisy: σ=3 pixel image noise → rotation < 1.0°, translation < 3 mm
Multi-face: verifies solver accepts frames only when ≥ 2 faces visible
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from synthetic_tests.synth.cameras import make_default_scene
from synthetic_tests.synth.renderer import render_frame, write_session_frames
from extrinsic_solver.solve import estimate_camera_pose


def _rotation_error_deg(T_est: np.ndarray, T_true: np.ndarray) -> float:
    R_err = T_est[:3, :3] @ T_true[:3, :3].T
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _translation_error_mm(T_est: np.ndarray, T_true: np.ndarray) -> float:
    return float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000)


@pytest.fixture(scope="module")
def smoke_session(tmp_path_factory):
    """Write 10 noise-free frames to a tmp session dir."""
    session_dir = tmp_path_factory.mktemp("smoke_session")
    scene = make_default_scene(n_cameras=4)
    write_session_frames(scene, session_dir, n_frames=10, noise_sigma=0.0, seed=0)
    return scene, session_dir


@pytest.fixture(scope="module")
def noisy_session(tmp_path_factory):
    """Write 10 noisy frames (σ=3) to a tmp session dir."""
    session_dir = tmp_path_factory.mktemp("noisy_session")
    scene = make_default_scene(n_cameras=4)
    write_session_frames(scene, session_dir, n_frames=10, noise_sigma=3.0, seed=1)
    return scene, session_dir


def _run_solver(scene, session_dir, cam_id, max_reproj_px=2.0):
    cam = next(c for c in scene.cameras if c["id"] == cam_id)
    frame_paths = sorted((session_dir / cam_id).glob("frame_*.png"))
    box_cfg = scene.to_box_cfg()
    return estimate_camera_pose(
        frame_paths, cam["intrinsics"], box_cfg,
        min_markers=3, max_reproj_px=max_reproj_px,
    )


@pytest.mark.parametrize("cam_id", [f"cam_{i}" for i in range(4)])
def test_extrinsic_smoke(smoke_session, cam_id):
    """Noise-free: rotation < 0.5°, translation < 1 mm."""
    scene, session_dir = smoke_session
    pose = _run_solver(scene, session_dir, cam_id, max_reproj_px=1.0)
    T_true = scene.get_T_cam_box(cam_id)
    rot_err = _rotation_error_deg(pose.T_cam_box, T_true)
    t_err_mm = _translation_error_mm(pose.T_cam_box, T_true)
    assert rot_err < 0.5, f"{cam_id}: rotation error {rot_err:.3f}°"
    assert t_err_mm < 1.0, f"{cam_id}: translation error {t_err_mm:.3f} mm"


@pytest.mark.parametrize("cam_id", [f"cam_{i}" for i in range(4)])
def test_extrinsic_noisy(noisy_session, cam_id):
    """σ=3 image noise: rotation < 1.0°, translation < 3 mm."""
    scene, session_dir = noisy_session
    pose = _run_solver(scene, session_dir, cam_id, max_reproj_px=3.0)
    T_true = scene.get_T_cam_box(cam_id)
    rot_err = _rotation_error_deg(pose.T_cam_box, T_true)
    t_err_mm = _translation_error_mm(pose.T_cam_box, T_true)
    assert rot_err < 1.0, f"{cam_id}: rotation error {rot_err:.3f}°"
    assert t_err_mm < 3.0, f"{cam_id}: translation error {t_err_mm:.3f} mm"


def test_extrinsic_multi_face_required(smoke_session):
    """Solver raises RuntimeError when only 1 face visible (single side marker only)."""
    from extrinsic_solver.solve import _build_board, _detect_markers
    scene, session_dir = smoke_session
    # Use a box_cfg with markers only on one face → face_set check fails.
    one_face_cfg = scene.to_box_cfg()
    one_face_cfg["markers"] = [m for m in one_face_cfg["markers"] if m["face"] == "front"]
    cam = scene.cameras[0]
    frame_paths = sorted((session_dir / cam["id"]).glob("frame_*.png"))
    with pytest.raises(RuntimeError):
        estimate_camera_pose(
            frame_paths, cam["intrinsics"], one_face_cfg,
            min_markers=1, max_reproj_px=5.0,
        )


def test_extrinsic_reprojection_error_low(smoke_session):
    """Reported reprojection error ≤ 1.0 px for all cameras on noise-free frames."""
    scene, session_dir = smoke_session
    for cam in scene.cameras:
        pose = _run_solver(scene, session_dir, cam["id"], max_reproj_px=1.0)
        assert pose.reprojection_error <= 1.0, \
            f"{cam['id']}: reproj error {pose.reprojection_error:.3f} px"
