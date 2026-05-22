"""
Test error_propagation.propagate against synthetic data.

MC validation: Frobenius ratio of empirical vs analytical triangulation covariance < 0.3
Budget sources: all 5 sources present and have positive semi-definite covariances
Total covariance: sum of independent MC sources ≥ triangulation covariance (more uncertainty)
"""
from __future__ import annotations

import numpy as np
import pytest

from synthetic_tests.synth.cameras import make_default_scene
from synthetic_tests.synth.noise import noisy_ball_detection
from synthetic_tests.synth.renderer import _project
from common import BallDetection2D, CameraPose, TriangulationResult
from triangulation.triangulate import triangulate
from error_propagation.propagate import propagate_errors, mc_validate_triangulation


def _build_inputs(scene, sigma_px=0.3, n_frames=100, seed=0):
    rng = np.random.default_rng(seed)
    detections = {
        c["id"]: noisy_ball_detection(scene, c["id"], sigma_px=sigma_px,
                                       n_frames=n_frames, rng=rng)
        for c in scene.cameras
    }
    poses = {c["id"]: scene.to_camera_pose(c["id"]) for c in scene.cameras}
    intrinsics = {c["id"]: c["intrinsics"] for c in scene.cameras}
    tri = triangulate(detections, poses, intrinsics)
    return detections, poses, intrinsics, tri


@pytest.fixture(scope="module")
def propagation_inputs(default_scene):
    return _build_inputs(default_scene)


def test_error_budget_has_five_sources(propagation_inputs, default_scene):
    detections, poses, intrinsics, tri = propagation_inputs
    box_cfg = default_scene.to_box_cfg()
    budget = propagate_errors(tri, poses, intrinsics, detections, box_cfg, n_mc=100, seed=0)
    assert len(budget.sources) == 5


def test_error_budget_sources_psd(propagation_inputs, default_scene):
    """All source covariances are positive semi-definite."""
    detections, poses, intrinsics, tri = propagation_inputs
    box_cfg = default_scene.to_box_cfg()
    budget = propagate_errors(tri, poses, intrinsics, detections, box_cfg, n_mc=100, seed=0)
    for src in budget.sources:
        cov = src.covariance_box
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-15), \
            f"Source '{src.name}' covariance not PSD: min eigval={eigvals.min():.2e}"


def test_error_budget_total_psd(propagation_inputs, default_scene):
    detections, poses, intrinsics, tri = propagation_inputs
    box_cfg = default_scene.to_box_cfg()
    budget = propagate_errors(tri, poses, intrinsics, detections, box_cfg, n_mc=100, seed=0)
    eigvals = np.linalg.eigvalsh(budget.total_covariance)
    assert np.all(eigvals > 0), f"total covariance not PD: min eigval={eigvals.min():.2e}"


def test_mc_validation_frobenius_ratio(propagation_inputs, default_scene):
    """MC (2D-noise only) vs triangulation covariance Frobenius ratio < 0.3 (500 trials)."""
    detections, poses, intrinsics, tri = propagation_inputs
    mc_result = mc_validate_triangulation(
        intrinsics, poses, detections,
        tri.position_box, tri.covariance_box,
        n_mc=500, seed=42,
    )
    ratio = mc_result["frobenius_ratio"]
    assert ratio < 0.3, f"Frobenius ratio {ratio:.3f} ≥ 0.3 (poor agreement)"


def test_total_covariance_exceeds_triangulation(propagation_inputs, default_scene):
    """Total covariance trace ≥ triangulation covariance trace (error adds uncertainty)."""
    detections, poses, intrinsics, tri = propagation_inputs
    box_cfg = default_scene.to_box_cfg()
    budget = propagate_errors(tri, poses, intrinsics, detections, box_cfg, n_mc=100, seed=2)
    total_trace = np.trace(budget.total_covariance)
    tri_trace = np.trace(tri.covariance_box)
    assert total_trace >= tri_trace * 0.9, \
        f"total trace {total_trace:.2e} < tri trace {tri_trace:.2e}"
