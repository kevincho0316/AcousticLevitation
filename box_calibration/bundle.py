"""Bundle adjustment for full self-calibration of marker poses + camera poses.

No nominal layout, no priors. One marker (anchor_idx) is held fixed at
identity to remove gauge ambiguity. Marker_side_m fixes scale through each
marker's known corner geometry.

Parameter vector layout:
  x = [ cam_0_xi(6), ..., cam_{N-1}_xi(6),
        mk_xi_for_each_free_marker(6) × (M-1) ]

  cam_i_xi:    se3 log-map of T_cam_box
  mk_j_xi:     se3 log-map of T_box_marker (anchor excluded)

Residuals: pure reprojection. 8 × n_detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from common.se3_utils import _se3_log

from .faces import marker_corners_mkr_frame


# ── Vectorized math (kept from previous bundle.py) ────────────────────────────

def _se3_exp_batch(xis: np.ndarray) -> np.ndarray:
    n = len(xis)
    u = xis[:, :3]
    omega = xis[:, 3:]
    theta = np.linalg.norm(omega, axis=1)
    small = theta < 1e-8
    ts = np.where(small, 1.0, theta)

    ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]
    O = np.zeros((n, 3, 3))
    O[:, 0, 1] = -oz;  O[:, 0, 2] =  oy
    O[:, 1, 0] =  oz;  O[:, 1, 2] = -ox
    O[:, 2, 0] = -oy;  O[:, 2, 1] =  ox

    I3 = np.eye(3)
    OO = O @ O
    s = np.sin(ts)[:, None, None]
    c = np.cos(ts)[:, None, None]
    ts2 = ts[:, None, None] ** 2
    ts3 = ts[:, None, None] ** 3

    R = I3 + (s / ts[:, None, None]) * O + ((1 - c) / ts2) * OO
    V = I3 + ((1 - c) / ts2) * O + ((ts[:, None, None] - s) / ts3) * OO
    R[small] = I3
    V[small] = I3

    T = np.zeros((n, 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = (V @ u[:, :, None])[:, :, 0]
    T[:, 3, 3] = 1.0
    return T


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BundleResult:
    x: np.ndarray
    n_cams: int
    n_markers: int
    anchor_idx: int                       # index into marker_ids list
    marker_ids: list[int]
    marker_poses: np.ndarray              # (n_markers, 4, 4) T_box_marker, anchor = I
    detection_list: list                  # (cam_idx, mk_idx, obs(4,2))
    per_marker_rms: np.ndarray
    per_image_rms: np.ndarray
    n_obs_per_marker: np.ndarray
    final_rms: float


# ── Detection list helpers ────────────────────────────────────────────────────

def _build_detection_list(
    detections: list,
    marker_ids: list[int],
    valid_marker_set: set[int],
) -> list[tuple[int, int, np.ndarray]]:
    id_to_idx = {mid: i for i, mid in enumerate(marker_ids)}
    det_list = []
    for cam_idx, (_, det, _) in enumerate(detections):
        for mid, px in det.items():
            if mid in valid_marker_set:
                det_list.append((cam_idx, id_to_idx[mid], px))
    return det_list


def _det_arrays(det_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_idxs = np.array([d[0] for d in det_list], dtype=np.int32)
    mk_idxs  = np.array([d[1] for d in det_list], dtype=np.int32)
    obs_all  = np.stack([d[2] for d in det_list])
    return cam_idxs, mk_idxs, obs_all


def _build_sparsity(
    n_cams: int,
    n_markers: int,
    anchor_idx: int,
    det_list: list,
) -> "lil_matrix":
    """Sparsity for reprojection-only Jacobian.

    Free marker params packed contiguously skipping anchor. We index by
    free_marker_slot(mk_idx) = mk_idx if mk_idx < anchor_idx else mk_idx - 1.
    """
    n_free_mk = n_markers - 1
    n_res = 8 * len(det_list)
    n_params = 6 * (n_cams + n_free_mk)

    J = lil_matrix((n_res, n_params), dtype=np.int8)

    for di, (cam_idx, mk_idx, _) in enumerate(det_list):
        rows = slice(8 * di, 8 * di + 8)
        J[rows, 6 * cam_idx : 6 * cam_idx + 6] = 1
        if mk_idx == anchor_idx:
            continue
        slot = mk_idx if mk_idx < anchor_idx else mk_idx - 1
        col = 6 * n_cams + 6 * slot
        J[rows, col : col + 6] = 1

    return J.tocsr()


# ── Pose unpacking ────────────────────────────────────────────────────────────

def _unpack(
    x: np.ndarray,
    n_cams: int,
    n_markers: int,
    anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_cams (n_cams,4,4), T_marks (n_markers,4,4))."""
    cam_xis = x[: 6 * n_cams].reshape(n_cams, 6)
    T_cams = _se3_exp_batch(cam_xis)

    n_free_mk = n_markers - 1
    mk_xis_free = x[6 * n_cams :].reshape(n_free_mk, 6)

    mk_xis = np.zeros((n_markers, 6))
    free_slot = 0
    for i in range(n_markers):
        if i == anchor_idx:
            continue
        mk_xis[i] = mk_xis_free[free_slot]
        free_slot += 1

    T_marks = _se3_exp_batch(mk_xis)
    T_marks[anchor_idx] = np.eye(4)
    return T_cams, T_marks


# ── Residuals ─────────────────────────────────────────────────────────────────

def _residuals(
    x: np.ndarray,
    cam_idxs: np.ndarray,
    mk_idxs: np.ndarray,
    obs_all: np.ndarray,
    corners_hom: np.ndarray,             # (4,4) [corners|1]^T
    fx: float, fy: float, cx: float, cy: float,
    n_cams: int,
    n_markers: int,
    anchor_idx: int,
) -> np.ndarray:
    T_cams, T_marks = _unpack(x, n_cams, n_markers, anchor_idx)
    T_cam_mk = T_cams[cam_idxs] @ T_marks[mk_idxs]
    pts_hom = T_cam_mk @ corners_hom
    z = pts_hom[:, 2, :]
    x_p = fx * pts_hom[:, 0, :] / z + cx
    y_p = fy * pts_hom[:, 1, :] / z + cy
    proj = np.stack([x_p, y_p], axis=2)
    return (obs_all - proj).reshape(-1)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _compute_stats(
    x: np.ndarray,
    det_list: list,
    corners_mkr: np.ndarray,
    K: np.ndarray,
    n_cams: int,
    n_markers: int,
    anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_idxs, mk_idxs, obs_all = _det_arrays(det_list)
    corners_hom = np.hstack([corners_mkr, np.ones((4, 1))]).T

    T_cams, T_marks = _unpack(x, n_cams, n_markers, anchor_idx)
    T_cam_mk = T_cams[cam_idxs] @ T_marks[mk_idxs]
    pts_hom = T_cam_mk @ corners_hom
    z = pts_hom[:, 2, :]
    x_p = K[0, 0] * pts_hom[:, 0, :] / z + K[0, 2]
    y_p = K[1, 1] * pts_hom[:, 1, :] / z + K[1, 2]
    proj = np.stack([x_p, y_p], axis=2)
    sq_per_det = np.sum((obs_all - proj) ** 2, axis=(1, 2))

    mk_sq  = np.bincount(mk_idxs,  weights=sq_per_det, minlength=n_markers)
    mk_cnt = np.bincount(mk_idxs,  minlength=n_markers) * 8
    cam_sq = np.bincount(cam_idxs, weights=sq_per_det, minlength=n_cams)
    cam_cnt= np.bincount(cam_idxs, minlength=n_cams) * 8

    per_mk  = np.sqrt(mk_sq  / np.maximum(mk_cnt,  1))
    per_cam = np.sqrt(cam_sq / np.maximum(cam_cnt, 1))
    return per_mk, per_cam, mk_cnt // 8, T_marks


# ── Bundle adjustment ─────────────────────────────────────────────────────────

def run_bundle_adjustment(
    detections: list,
    marker_ids: list[int],
    init_marker_poses: dict[int, np.ndarray],
    init_camera_poses: list[np.ndarray | None],
    anchor_id: int,
    K: np.ndarray,
    marker_side_m: float,
    huber_scale: float = 1.0,
    max_initial_rms_px: float = 40.0,
    outlier_threshold_px: float = 2.0,
    outlier_factor: float = 5.0,
) -> BundleResult:
    n_cams    = len(detections)
    n_markers = len(marker_ids)
    anchor_idx = marker_ids.index(anchor_id)
    valid_set = set(marker_ids)
    corners_mkr = marker_corners_mkr_frame(marker_side_m)

    det_list = _build_detection_list(detections, marker_ids, valid_set)
    print(f"  Observations: {len(det_list)} marker detections from {n_cams} images")

    # Build x0.
    cam_xis_init = np.zeros((n_cams, 6))
    for i, pose in enumerate(init_camera_poses):
        if pose is not None:
            cam_xis_init[i] = _se3_log(pose)

    n_free_mk = n_markers - 1
    mk_xis_free_init = np.zeros((n_free_mk, 6))
    free_slot = 0
    for i, mid in enumerate(marker_ids):
        if i == anchor_idx:
            continue
        T_box_mk = init_marker_poses.get(mid)
        if T_box_mk is not None:
            mk_xis_free_init[free_slot] = _se3_log(T_box_mk)
        free_slot += 1

    x0 = np.concatenate([cam_xis_init.ravel(), mk_xis_free_init.ravel()])

    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    def _make_args(dl):
        cam_idxs, mk_idxs, obs_all = _det_arrays(dl)
        corners_hom = np.hstack([corners_mkr, np.ones((4, 1))]).T
        return (cam_idxs, mk_idxs, obs_all, corners_hom,
                fx, fy, cx, cy, n_cams, n_markers, anchor_idx)

    args0 = _make_args(det_list)
    rms_init = float(np.sqrt(np.mean(_residuals(x0, *args0) ** 2)))
    print(f"  Initial reprojection RMS: {rms_init:.3f} px")
    if rms_init > max_initial_rms_px:
        raise RuntimeError(
            f"Initial reprojection RMS {rms_init:.3f} px exceeds "
            f"--max-initial-reproj-px={max_initial_rms_px}. "
            "Check marker side, intrinsics, image resolution, and that the "
            "co-visibility graph init produced sensible relative poses."
        )

    def _solve(x_start, dl, label: str):
        args_l = _make_args(dl)
        sp = _build_sparsity(n_cams, n_markers, anchor_idx, dl)
        n_reproj = 8 * len(dl)

        try:
            from tqdm import tqdm
            pbar = tqdm(total=50000, desc=f"  {label}", unit="nfev",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
                        dynamic_ncols=True, leave=False)
        except ImportError:
            pbar = None

        def _tracked(x, *a):
            r = _residuals(x, *a)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(rms=f"{np.sqrt(np.mean(r[:n_reproj]**2)):.3f}px",
                                 refresh=False)
            return r

        try:
            res = least_squares(
                _tracked, x_start, args=args_l,
                method="trf", loss="huber", f_scale=huber_scale,
                jac_sparsity=sp,
                ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=50000,
            )
        finally:
            if pbar is not None:
                pbar.close()
        return res

    result = _solve(x0, det_list, "BA pass 1")
    n_det = len(det_list)
    rms1 = float(np.sqrt(np.mean(result.fun[:8*n_det] ** 2)))
    print(f"  BA pass 1 RMS: {rms1:.3f} px ({n_det} observations)")

    per_det_rms = np.sqrt(np.mean(result.fun[:8*n_det].reshape(-1, 8) ** 2, axis=1))
    med = float(np.median(per_det_rms))
    thresh = max(outlier_threshold_px, outlier_factor * med)
    mask = per_det_rms <= thresh
    n_removed = int(np.sum(~mask))

    if n_removed > 0:
        det_list = [d for d, keep in zip(det_list, mask) if keep]
        print(f"  Outlier rejection: removed {n_removed} (thresh={thresh:.2f} px), "
              f"{len(det_list)} remaining")
        result = _solve(result.x, det_list, "BA pass 2")
        print(f"  BA pass 2 RMS: {np.sqrt(np.mean(result.fun[:8*len(det_list)]**2)):.3f} px")

    x_final = result.x
    final_rms = float(np.sqrt(np.mean(result.fun[:8*len(det_list)] ** 2)))
    per_mk, per_cam, n_obs, T_marks_final = _compute_stats(
        x_final, det_list, corners_mkr, K, n_cams, n_markers, anchor_idx,
    )

    return BundleResult(
        x=x_final, n_cams=n_cams, n_markers=n_markers,
        anchor_idx=anchor_idx, marker_ids=marker_ids,
        marker_poses=T_marks_final,
        detection_list=det_list,
        per_marker_rms=per_mk, per_image_rms=per_cam,
        n_obs_per_marker=n_obs, final_rms=final_rms,
    )
