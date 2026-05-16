"""Bundle adjustment: jointly optimize camera poses and per-marker 6-DOF offsets.

Parameter vector layout:
  x = [cam_0_xi(6), ..., cam_{N-1}_xi(6),
        mk_0_off(6), ..., mk_{M-1}_off(6)]

  cam_i_xi:  se3 log-map of T_cam_box  (see common/se3_utils.py convention)
  mk_i_off:  [rotvec(3), t(3)] in marker local frame
              T_box_mk_refined = T_box_mk_nominal @ T_offset

Residuals layout:
  [reproj(8*n_det), prior_t(3*n_mk), prior_r(3*n_mk)]

Priors:
  translation:  t_off / sigma_t   (isotropic, 3 residuals per marker)
  rotation:     rotvec / sigma_r  (per-axis, 3 residuals per marker)

Performance:
  _residuals is fully vectorized — no Python loops over detections.
  _se3_exp_batch and _rodrigues_batch replace per-element scalar ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from common.se3_utils import _se3_log
from .faces import BoxModel


# ── Vectorized math ───────────────────────────────────────────────────────────

def _se3_exp_batch(xis: np.ndarray) -> np.ndarray:
    """(N,6) twist vectors [u,omega] → (N,4,4) SE(3) matrices."""
    n = len(xis)
    u = xis[:, :3]
    omega = xis[:, 3:]
    theta = np.linalg.norm(omega, axis=1)          # (N,)
    small = theta < 1e-8
    ts = np.where(small, 1.0, theta)               # safe theta

    ox, oy, oz = omega[:, 0], omega[:, 1], omega[:, 2]
    O = np.zeros((n, 3, 3))                        # skew-symmetric (N,3,3)
    O[:, 0, 1] = -oz;  O[:, 0, 2] =  oy
    O[:, 1, 0] =  oz;  O[:, 1, 2] = -ox
    O[:, 2, 0] = -oy;  O[:, 2, 1] =  ox

    I3 = np.eye(3)
    OO = O @ O                                     # (N,3,3)
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


def _rodrigues_batch(rvecs: np.ndarray) -> np.ndarray:
    """(N,3) rotation vectors → (N,3,3) rotation matrices."""
    n = len(rvecs)
    angles = np.linalg.norm(rvecs, axis=1)         # (N,)
    small = angles < 1e-10
    as_ = np.where(small, 1.0, angles)
    axes = rvecs / as_[:, None]                    # (N,3) unit axes

    c = np.cos(as_)
    s = np.sin(as_)
    omc = 1.0 - c                                  # one-minus-cos
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]

    R = np.empty((n, 3, 3))
    R[:, 0, 0] = c + omc * x * x
    R[:, 0, 1] = omc * x * y - s * z
    R[:, 0, 2] = omc * x * z + s * y
    R[:, 1, 0] = omc * y * x + s * z
    R[:, 1, 1] = c + omc * y * y
    R[:, 1, 2] = omc * y * z - s * x
    R[:, 2, 0] = omc * z * x - s * y
    R[:, 2, 1] = omc * z * y + s * x
    R[:, 2, 2] = c + omc * z * z
    R[small] = np.eye(3)
    return R


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BundleResult:
    x: np.ndarray
    n_cams: int
    n_markers: int
    detection_list: list                 # (cam_idx, mk_idx, obs(4,2))
    per_marker_rms: np.ndarray          # (n_markers,) px
    per_image_rms: np.ndarray           # (n_cams,) px
    offsets_t_m: np.ndarray            # (n_markers,3) translation in marker frame
    offsets_r_rad: np.ndarray          # (n_markers,3) rotvec in marker frame
    n_obs_per_marker: np.ndarray       # (n_markers,) corner-set count
    final_rms: float


def make_T_offset(offset: np.ndarray) -> np.ndarray:
    """Single 4×4 T_offset from [rotvec(3), t(3)]. Used outside the hot path."""
    R_off = _rodrigues_batch(offset[None, :3])[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_off
    T[:3, 3] = offset[3:]
    return T


# ── Detection list helpers ────────────────────────────────────────────────────

def _build_detection_list(
    detections: list,
    box_model: BoxModel,
) -> list[tuple[int, int, np.ndarray]]:
    """Flatten to (cam_idx, mk_idx, obs_corners(4,2)) list."""
    id_to_idx = {mid: i for i, mid in enumerate(box_model.ids)}
    det_list = []
    for cam_idx, (_, det, _) in enumerate(detections):
        for mid, px in det.items():
            if mid in id_to_idx:
                det_list.append((cam_idx, id_to_idx[mid], px))
    return det_list


def _det_arrays(det_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute index arrays and stacked observations from det_list.

    Returns cam_idxs(n,), mk_idxs(n,), obs_all(n,4,2) — built once per solve.
    """
    cam_idxs = np.array([d[0] for d in det_list], dtype=np.int32)
    mk_idxs  = np.array([d[1] for d in det_list], dtype=np.int32)
    obs_all  = np.stack([d[2] for d in det_list])       # (n,4,2)
    return cam_idxs, mk_idxs, obs_all


def _build_sparsity(n_cams: int, n_markers: int, det_list: list):
    n_reproj = 8 * len(det_list)
    n_prior = 6 * n_markers
    n_res = n_reproj + n_prior
    n_params = 6 * (n_cams + n_markers)

    J = lil_matrix((n_res, n_params), dtype=np.int8)

    for di, (cam_idx, mk_idx, _) in enumerate(det_list):
        rows = slice(8 * di, 8 * di + 8)
        J[rows, 6 * cam_idx : 6 * cam_idx + 6] = 1
        col_mk = 6 * n_cams + 6 * mk_idx
        J[rows, col_mk : col_mk + 6] = 1

    for i in range(n_markers):
        for j in range(3):
            J[n_reproj + 3 * i + j,               6 * n_cams + 6 * i + 3 + j] = 1
            J[n_reproj + 3*n_markers + 3*i + j,   6 * n_cams + 6 * i + j]     = 1

    return J.tocsr()


# ── Vectorized residuals ──────────────────────────────────────────────────────

def _residuals(
    x: np.ndarray,
    cam_idxs: np.ndarray,    # (n_det,)
    mk_idxs: np.ndarray,     # (n_det,)
    obs_all: np.ndarray,     # (n_det,4,2)
    nom_poses: np.ndarray,   # (n_markers,4,4)
    corners_hom: np.ndarray, # (4,4)  precomputed [corners|1]^T
    fx: float, fy: float, cx: float, cy: float,
    sigma_t: float,
    sigma_r: np.ndarray,     # (3,)
    n_cams: int,
    n_markers: int,
) -> np.ndarray:
    cam_xis = x[: 6 * n_cams].reshape(n_cams, 6)
    mk_offs = x[6 * n_cams :].reshape(n_markers, 6)  # [rotvec(3), t(3)]

    # Camera transforms: (n_cams,4,4)
    T_cams = _se3_exp_batch(cam_xis)

    # Marker offset transforms: (n_markers,4,4)
    R_offs = _rodrigues_batch(mk_offs[:, :3])
    T_offs = np.zeros((n_markers, 4, 4))
    T_offs[:, :3, :3] = R_offs
    T_offs[:, :3, 3]  = mk_offs[:, 3:]
    T_offs[:, 3, 3]   = 1.0

    # Refined marker-to-box transforms: (n_markers,4,4)
    T_mks = nom_poses @ T_offs                     # batched matmul

    # Per-detection cam-to-marker transforms: (n_det,4,4)
    T_cam_mk = T_cams[cam_idxs] @ T_mks[mk_idxs]

    # Project all corners at once.
    # corners_hom: (4,4) — columns are homogeneous corner vectors
    # pts_hom[d,:,c] = T_cam_mk[d] @ corners_hom[:,c]
    pts_hom = T_cam_mk @ corners_hom               # (n_det,4,4)
    z = pts_hom[:, 2, :]                           # (n_det,4)
    x_p = fx * pts_hom[:, 0, :] / z + cx          # (n_det,4)
    y_p = fy * pts_hom[:, 1, :] / z + cy          # (n_det,4)
    proj = np.stack([x_p, y_p], axis=2)            # (n_det,4,2)

    reproj_err = (obs_all - proj).reshape(-1)       # (8*n_det,)

    # Priors (vectorized)
    t_prior = (mk_offs[:, 3:] / sigma_t).ravel()  # (3*n_mk,)
    r_prior = (mk_offs[:, :3] / sigma_r).ravel()  # (3*n_mk,)

    return np.concatenate([reproj_err, t_prior, r_prior])


# ── Stats ─────────────────────────────────────────────────────────────────────

def _compute_stats(
    x: np.ndarray,
    det_list: list,
    nom_poses: np.ndarray,
    corners_mkr: np.ndarray,
    K: np.ndarray,
    n_cams: int,
    n_markers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_idxs, mk_idxs, obs_all = _det_arrays(det_list)
    corners_hom = np.hstack([corners_mkr, np.ones((4, 1))]).T  # (4,4)

    cam_xis = x[: 6 * n_cams].reshape(n_cams, 6)
    mk_offs = x[6 * n_cams :].reshape(n_markers, 6)

    T_cams = _se3_exp_batch(cam_xis)
    R_offs = _rodrigues_batch(mk_offs[:, :3])
    T_offs = np.zeros((n_markers, 4, 4))
    T_offs[:, :3, :3] = R_offs
    T_offs[:, :3, 3]  = mk_offs[:, 3:]
    T_offs[:, 3, 3]   = 1.0
    T_mks = nom_poses @ T_offs

    T_cam_mk = T_cams[cam_idxs] @ T_mks[mk_idxs]
    pts_hom = T_cam_mk @ corners_hom
    z = pts_hom[:, 2, :]
    x_p = K[0,0] * pts_hom[:, 0, :] / z + K[0,2]
    y_p = K[1,1] * pts_hom[:, 1, :] / z + K[1,2]
    proj = np.stack([x_p, y_p], axis=2)            # (n_det,4,2)
    sq_per_det = np.sum((obs_all - proj) ** 2, axis=(1, 2))  # (n_det,)

    mk_sq  = np.bincount(mk_idxs,  weights=sq_per_det, minlength=n_markers)
    mk_cnt = np.bincount(mk_idxs,  minlength=n_markers) * 8
    cam_sq = np.bincount(cam_idxs, weights=sq_per_det, minlength=n_cams)
    cam_cnt= np.bincount(cam_idxs, minlength=n_cams) * 8

    per_mk  = np.sqrt(mk_sq  / np.maximum(mk_cnt,  1))
    per_cam = np.sqrt(cam_sq / np.maximum(cam_cnt, 1))
    return per_mk, per_cam, mk_cnt // 8


# ── Bundle adjustment ─────────────────────────────────────────────────────────

def run_bundle_adjustment(
    detections: list,
    box_model: BoxModel,
    init_poses: list,
    K: np.ndarray,
    sigma_t: float,
    sigma_r: np.ndarray,
    huber_scale: float = 1.0,
    outlier_threshold_px: float = 2.0,
    outlier_factor: float = 5.0,
) -> BundleResult:
    n_cams    = len(detections)
    n_markers = len(box_model.ids)

    det_list = _build_detection_list(detections, box_model)
    print(f"  Observations: {len(det_list)} marker detections from {n_cams} images")

    cam_xis_init = np.zeros((n_cams, 6))
    for i, pose in enumerate(init_poses):
        if pose is not None:
            cam_xis_init[i] = _se3_log(pose)
    x0 = np.concatenate([cam_xis_init.ravel(), np.zeros(n_markers * 6)])

    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    def _make_args(dl):
        cam_idxs, mk_idxs, obs_all = _det_arrays(dl)
        corners_hom = np.hstack([box_model.corners_mkr, np.ones((4,1))]).T  # (4,4)
        return (cam_idxs, mk_idxs, obs_all, box_model.nominal_poses, corners_hom,
                fx, fy, cx, cy, sigma_t, sigma_r, n_cams, n_markers)

    args0 = _make_args(det_list)
    rms_init = float(np.sqrt(np.mean(_residuals(x0, *args0)[: 8*len(det_list)] ** 2)))
    print(f"  Initial reprojection RMS: {rms_init:.3f} px")

    def _solve(x_start, dl, label: str):
        args_l = _make_args(dl)
        sp = _build_sparsity(n_cams, n_markers, dl)
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

    # Outlier rejection.
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

    x_final   = result.x
    final_rms = float(np.sqrt(np.mean(result.fun[:8*len(det_list)] ** 2)))
    mk_offs   = x_final[6*n_cams:].reshape(n_markers, 6)
    per_mk, per_cam, n_obs = _compute_stats(
        x_final, det_list, box_model.nominal_poses, box_model.corners_mkr,
        K, n_cams, n_markers,
    )

    return BundleResult(
        x=x_final, n_cams=n_cams, n_markers=n_markers,
        detection_list=det_list,
        per_marker_rms=per_mk, per_image_rms=per_cam,
        offsets_t_m=mk_offs[:, 3:], offsets_r_rad=mk_offs[:, :3],
        n_obs_per_marker=n_obs, final_rms=final_rms,
    )


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(
    detections: list,
    box_model: BoxModel,
    K: np.ndarray,
    sigma_t: float,
    sigma_r: np.ndarray,
    huber_scale: float = 1.0,
) -> float:
    if len(detections) < 3:
        print("  Cross-validation skipped: need >= 3 images")
        return float("nan")

    train        = detections[:-1]
    holdout_path, holdout_det, _ = detections[-1]

    n_cams_train = len(train)
    n_markers    = len(box_model.ids)
    det_list     = _build_detection_list(train, box_model)
    x0 = np.zeros(6 * n_cams_train + 6 * n_markers)

    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    cam_idxs, mk_idxs, obs_all = _det_arrays(det_list)
    corners_hom = np.hstack([box_model.corners_mkr, np.ones((4,1))]).T
    args = (cam_idxs, mk_idxs, obs_all, box_model.nominal_poses, corners_hom,
            fx, fy, cx, cy, sigma_t, sigma_r, n_cams_train, n_markers)
    sp = _build_sparsity(n_cams_train, n_markers, det_list)

    result = least_squares(
        _residuals, x0, args=args,
        method="trf", loss="huber", f_scale=huber_scale,
        jac_sparsity=sp, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=30000,
    )

    mk_offs   = result.x[6*n_cams_train:].reshape(n_markers, 6)
    id_to_idx = {mid: i for i, mid in enumerate(box_model.ids)}
    corners_hom_col = np.hstack([box_model.corners_mkr, np.ones((4,1))])  # (4,4) rows

    obj_pts, img_pts = [], []
    for mid, px in holdout_det.items():
        if mid not in id_to_idx:
            continue
        i = id_to_idx[mid]
        T_box_mk = box_model.nominal_poses[i] @ make_T_offset(mk_offs[i])
        corners_box = (T_box_mk @ corners_hom_col.T).T[:, :3]
        obj_pts.append(corners_box.astype(np.float32))
        img_pts.append(px.astype(np.float32))

    if not obj_pts:
        return float("nan")

    zero_dist = np.zeros(5, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        np.concatenate(obj_pts), np.concatenate(img_pts),
        K.astype(np.float32), zero_dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return float("nan")

    R, _ = cv2.Rodrigues(rvec)
    T_cam_box = np.eye(4)
    T_cam_box[:3, :3] = R
    T_cam_box[:3, 3]  = tvec.ravel()

    sq_sum, n_total = 0.0, 0
    for mid, obs in holdout_det.items():
        if mid not in id_to_idx:
            continue
        i = id_to_idx[mid]
        T_box_mk = box_model.nominal_poses[i] @ make_T_offset(mk_offs[i])
        pts = (T_cam_box @ T_box_mk @ corners_hom_col.T).T[:, :3]
        x_p = fx * pts[:, 0] / pts[:, 2] + cx
        y_p = fy * pts[:, 1] / pts[:, 2] + cy
        sq_sum  += float(np.sum((obs - np.stack([x_p, y_p], 1)) ** 2))
        n_total += 8

    rms = float(np.sqrt(sq_sum / n_total)) if n_total else float("nan")
    print(f"  Cross-validation holdout RMS: {rms:.3f} px ({holdout_path.name})")
    return rms
