from __future__ import annotations

import numpy as np

from box_calibration.box_fit import fit_box_frame_with_labels
from box_calibration.faces import nominal_pose, marker_corners_mkr_frame
from common.se3_utils import _se3_exp


def _corners(T: np.ndarray, marker_side_m: float) -> np.ndarray:
    local = marker_corners_mkr_frame(marker_side_m)
    hom = np.hstack([local, np.ones((4, 1))])
    return (T @ hom.T).T[:, :3]


def test_fit_box_frame_with_labels_solves_global_box_pose_only():
    W_mm, H_mm, D_mm = 80.0, 30.0, 80.0
    marker_side_m = 0.015

    nominal = np.stack([
        nominal_pose("front", np.array([40.0, 15.0, 0.0]) / 1000.0),
        nominal_pose("right", np.array([80.0, 15.0, 40.0]) / 1000.0),
        nominal_pose("left",  np.array([0.0, 15.0, 40.0]) / 1000.0),
        nominal_pose("top",   np.array([20.0, 30.0, 20.0]) / 1000.0),
        nominal_pose("top",   np.array([60.0, 30.0, 60.0]) / 1000.0),
    ])
    face_labels = ["front", "right", "left", "top", "top"]

    T_align_true = _se3_exp(np.array([0.08, -0.04, 0.06, 0.03, -0.02, 0.04]))
    T_ba_from_box = np.linalg.inv(T_align_true)
    marker_poses_ba = np.stack([T_ba_from_box @ T for T in nominal])

    T_align, _, info = fit_box_frame_with_labels(
        marker_poses_ba,
        face_labels=face_labels,
        W_mm=W_mm,
        H_mm=H_mm,
        D_mm=D_mm,
        marker_side_m=marker_side_m,
    )

    recovered = np.stack([T_align @ T for T in marker_poses_ba])
    assert np.allclose(recovered, nominal, atol=1e-7)
    assert info["rms_plane_mm"] < 1e-6
    assert info["max_plane_mm"] < 1e-5

    front = _corners(recovered[0], marker_side_m)
    right = _corners(recovered[1], marker_side_m)
    left = _corners(recovered[2], marker_side_m)
    top0 = _corners(recovered[3], marker_side_m)
    top1 = _corners(recovered[4], marker_side_m)
    assert np.allclose(front[:, 2], 0.0, atol=1e-8)
    assert np.allclose(right[:, 0], W_mm / 1000.0, atol=1e-8)
    assert np.allclose(left[:, 0], 0.0, atol=1e-8)
    assert np.allclose(top0[:, 1], H_mm / 1000.0, atol=1e-8)
    assert np.allclose(top1[:, 1], H_mm / 1000.0, atol=1e-8)
