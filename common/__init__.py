"""Shared data classes for the acoustic levitation measurement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    camera_id: str
    K: np.ndarray           # (3,3) camera matrix
    dist: np.ndarray        # distortion coefficients (5 or 8 elements)
    resolution: tuple[int, int]  # (width, height)
    reprojection_error: float    # mean reprojection error in pixels


@dataclass
class CameraPose:
    camera_id: str
    T_cam_box: np.ndarray        # (4,4) SE(3): box frame → camera frame
    reprojection_error: float    # mean reprojection error in pixels
    n_markers_used: int
    n_frames_used: int
    pose_covariance: Optional[np.ndarray] = None  # (6,6) if estimated


@dataclass
class BallDetection2D:
    camera_id: str
    center: np.ndarray           # (2,) averaged center [u, v] in pixels
    covariance: np.ndarray       # (2,2) covariance of the mean
    n_frames_accepted: int
    n_frames_rejected: int
    per_frame_centers: Optional[np.ndarray] = None  # (N, 2) raw detections


@dataclass
class TriangulationResult:
    position_box: np.ndarray            # (3,) meters, box frame
    covariance_box: np.ndarray          # (3,3) meters², box frame
    reprojection_residuals: dict[str, np.ndarray]  # camera_id → (2,) px
    n_cameras: int


@dataclass
class ErrorSource:
    name: str
    covariance_box: np.ndarray   # (3,3) contribution in meters²
    description: str = ""


@dataclass
class ErrorBudget:
    sources: list[ErrorSource]
    total_covariance: np.ndarray  # (3,3) sum of source covariances


@dataclass
class ComparisonResult:
    measured_position_box: np.ndarray   # (3,) meters, box frame
    measured_covariance_box: np.ndarray # (3,3) meters², box frame
    simulated_position_box: np.ndarray  # (3,) meters, box frame (transformed)
    simulated_position_sim: np.ndarray  # (3,) meters, sim frame (original)
    offset_vector_box: np.ndarray       # (3,) measured - simulated, meters
    mahalanobis_distance: float
    chi2_dof: int                        # degrees of freedom = 3
    passed: bool
    threshold_mm: float
    sim_candidate_rank: int
