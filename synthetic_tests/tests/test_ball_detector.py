"""
Test ball_detector.detect against synthetic images.

Smoke: zero noise  → centre recovered to < 0.5 px
Noise: σ=3 pixel noise on image → centre recovered to < 1.5 px
Radius sweep: ball appears 8–40 px radius → all detected
"""
from __future__ import annotations

import numpy as np
import pytest

from synthetic_tests.synth.cameras import make_default_scene
from synthetic_tests.synth.renderer import render_frame, _project
from ball_detector.detect import detect_ball_frame


def _true_uv(scene, cam_id: str) -> np.ndarray:
    cam = next(c for c in scene.cameras if c["id"] == cam_id)
    uv, _ = _project(cam["intrinsics"].K, cam["T_cam_box"], scene.ball_position_box)
    return uv


def test_ball_detector_smoke(default_scene):
    """Zero noise: every camera detects ball centre within 0.5 px."""
    for cam in default_scene.cameras:
        cam_id = cam["id"]
        img = render_frame(default_scene, cam_id, noise_sigma=0.0)
        result = detect_ball_frame(img, cam["intrinsics"])
        assert result is not None, f"{cam_id}: detection failed on noise-free image"
        uv_true = _true_uv(default_scene, cam_id)
        err = np.linalg.norm(np.array(result) - uv_true)
        assert err < 0.5, f"{cam_id}: error {err:.3f} px exceeds 0.5 px threshold"


def test_ball_detector_noisy_image(default_scene):
    """σ=3 pixel image noise: detection error < 1.5 px."""
    rng = np.random.default_rng(42)
    for cam in default_scene.cameras:
        cam_id = cam["id"]
        img = render_frame(default_scene, cam_id, noise_sigma=3.0, rng=rng)
        result = detect_ball_frame(img, cam["intrinsics"])
        assert result is not None, f"{cam_id}: detection failed with σ=3 noise"
        uv_true = _true_uv(default_scene, cam_id)
        err = np.linalg.norm(np.array(result) - uv_true)
        assert err < 1.5, f"{cam_id}: error {err:.3f} px with noise"


@pytest.mark.parametrize("ball_radius_m", [0.003, 0.006, 0.010, 0.016])
def test_ball_detector_radius_sweep(ball_radius_m):
    """Detection succeeds across typical ball radii (different projected sizes)."""
    scene = make_default_scene()
    scene.ball_radius_m = ball_radius_m
    cam = scene.cameras[0]
    cam_id = cam["id"]
    img = render_frame(scene, cam_id, noise_sigma=0.0)
    result = detect_ball_frame(img, cam["intrinsics"])
    assert result is not None, f"radius {ball_radius_m*1000:.0f} mm: detection failed"


def test_ball_detector_off_centre(default_scene):
    """Ball not at box centre is still detected correctly."""
    import copy
    scene = copy.copy(default_scene)
    scene.ball_position_box = np.array([0.07, 0.025, 0.05])  # offset from centre
    cam = scene.cameras[0]
    cam_id = cam["id"]
    img = render_frame(scene, cam_id, noise_sigma=0.0)
    result = detect_ball_frame(img, cam["intrinsics"])
    assert result is not None
    uv_true = _true_uv(scene, cam_id)
    err = np.linalg.norm(np.array(result) - uv_true)
    assert err < 0.5
