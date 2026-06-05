"""Best-fit box-centered frame for BA-output marker poses.

Every marker lies on one of the box's 6 faces. We find the SE(3) transform
T_align that re-expresses BA-output poses in the true box frame
(corner at origin, [0,W] × [0,H] × [0,D] in meters) while minimizing the
sum-of-squared signed distances of every marker corner to its assigned face
plane.

Residuals are over ArUco corners (not just centers): a single corner-to-plane
distance captures both the marker-center offset AND the marker-plane tilt.
Total residual vector length = 4 × M.

The optimization:
  1. Enumerate the 24 axis-aligned proper rotations as starting points
     (escapes axis-permutation/sign local minima).
  2. For each start, alternate: assign each marker to its best face, then
     nonlinear LS refine the 6-DoF T_align with assignments fixed.
  3. Keep the global minimum.

This returns the optimum over all SE(3) and all face assignments.
"""

from __future__ import annotations

from itertools import permutations, product
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from common.se3_utils import _se3_exp, _se3_log

from .bundle import _se3_exp_batch
from .faces import marker_corners_mkr_frame


# (face_name, axis_index, marker_normal_sign_in_box_frame, plane_offset_along_axis)
# Plane equation: x[axis] = off. `sign` = the OUTWARD face normal (physical ArUco
# +Z, always toward the camera) as a direction along `axis` in the box frame.
# OpenGL-style convention: the front face sits at z=0 with outward normal −Z (toward
# a camera at z<0); back sits at z=D with outward normal +Z. The normal term in
# _fit_fixed_faces pulls each marker's +Z onto sign*e_axis, resolving the 180° flip
# ambiguity that a plane-distance-only objective is blind to.
# NOTE: faces.FACE_TABLE seeds front/back the opposite way (into-box) to keep
# BFS/disambiguation init stable; BA converges to this outward truth regardless.
def _face_specs(W: float, H: float, D: float) -> list[tuple[str, int, float, float]]:
    return [
        ("right",  0, +1.0, W),
        ("left",   0, -1.0, 0.0),
        ("top",    1, +1.0, H),
        ("bottom", 1, -1.0, 0.0),
        ("front",  2, -1.0, 0.0),
        ("back",   2, +1.0, D),
    ]


def _assign_faces(
    centers: np.ndarray,
    normals: np.ndarray,
    face_specs: list[tuple[str, int, float, float]],
    scale: float,
    use_normals: bool = True,
) -> list[tuple[str, int, float, float, np.ndarray]]:
    """Assign each marker to a box face.

    use_normals=True : score blends plane distance with +Z↔face-normal agreement.
    use_normals=False: pure geometry — assign to the nearest face plane by absolute
      center-to-plane distance only. Marker orientation is ignored.
    """
    out = []
    for c, n in zip(centers, normals):
        best, best_score = None, np.inf
        for fname, axis, sign, off in face_specs:
            face_n = np.zeros(3); face_n[axis] = sign
            if use_normals:
                cos_a = float(np.dot(n, face_n))
                score = (1.0 - cos_a) + abs(sign * (c[axis] - off)) / scale
            else:
                score = abs(c[axis] - off)
            if score < best_score:
                best_score = score
                best = (fname, axis, sign, off, face_n)
        out.append(best)
    return out


def _kabsch_rotation(N_from: np.ndarray, N_to: np.ndarray) -> np.ndarray:
    """Proper rotation aligning rows of N_from to rows of N_to (least-squares)."""
    Hmat = N_from.T @ N_to
    U, _, Vt = np.linalg.svd(Hmat)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _axis_aligned_rotations() -> list[np.ndarray]:
    """All 24 proper axis-aligned rotations."""
    rots: list[np.ndarray] = []
    for perm in permutations(range(3)):
        P = np.eye(3)[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            R = P @ np.diag(signs)
            if np.linalg.det(R) > 0.5:
                rots.append(R)
    unique: list[np.ndarray] = []
    for R in rots:
        if not any(np.allclose(R, U) for U in unique):
            unique.append(R)
    return unique


def _corners_in_ba_frame(marker_poses: np.ndarray, marker_side_m: float) -> np.ndarray:
    corners_local = marker_corners_mkr_frame(marker_side_m)
    corners_hom = np.hstack([corners_local, np.ones((4, 1))])
    return np.stack([(T @ corners_hom.T).T[:, :3] for T in marker_poses])


def _solve_translation(
    centers0: np.ndarray,
    R: np.ndarray,
    axes: np.ndarray,
    offs: np.ndarray,
) -> np.ndarray:
    A = np.zeros((len(axes), 3), dtype=np.float64)
    A[np.arange(len(axes)), axes] = 1.0
    c_rot = centers0 @ R.T
    b = offs - np.einsum("ij,ij->i", A, c_rot)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


def _combined_residuals(
    xi: np.ndarray,
    pts_ba: np.ndarray,
    normals0: np.ndarray,
    axes: np.ndarray,
    signs: np.ndarray,
    offs: np.ndarray,
    sqrt_w: np.ndarray,
    w_normal: np.ndarray,
) -> np.ndarray:
    """Stacked residual for the global SE(3): per-corner plane distance + per-marker
    normal misalignment. Both blocks carry per-marker sqrt-weights; the normal block
    is scaled (in _fit_fixed_faces) into corner-displacement length so it is
    commensurate with the plane block (meters)."""
    T = _se3_exp(xi)
    R = T[:3, :3]
    t = T[:3, 3]

    pts = pts_ba @ R.T + t
    coord = np.take_along_axis(
        pts, axes[:, None, None].repeat(4, axis=1), axis=2,
    )[:, :, 0]
    plane = ((coord - offs[:, None]) * sqrt_w[:, None]).ravel()       # (M*4,)

    M = len(axes)
    target = np.zeros((M, 3))
    target[np.arange(M), axes] = signs                               # outward face normal
    n_box = normals0 @ R.T                                            # (M,3)
    normal = ((n_box - target) * w_normal[:, None]).ravel()          # (M*3,)

    return np.concatenate([plane, normal])


def _fit_fixed_faces(
    marker_poses: np.ndarray,
    axes: np.ndarray,
    offs: np.ndarray,
    signs: np.ndarray,
    marker_side_m: float,
    weights: np.ndarray | None = None,
    normal_weight: float = 1.0,
    use_normals: bool = True,
) -> tuple[np.ndarray, float]:
    """Solve one global SE(3) snapping marker corners onto their face planes AND
    (optionally) aligning each marker's +Z to the outward face normal (sign*e_axis).

    weights: (M,) per-marker confidence (e.g. n_obs / reproj-rms). Under-observed or
      high-residual markers are downweighted so they cannot drag the global fit.
      Combined with a robust soft_l1 loss that caps the influence of gross outliers
      (e.g. a side marker seen in a single image).
    normal_weight: relative pull of the normal term vs the plane term (1.0 = balanced).
      The normal term resolves the 180° flip ambiguity that the plane term is blind to.
    use_normals=False: drop the normal block entirely — minimize plane distance only
      ("best possible fit", orientation-agnostic). Flips are then resolved downstream
      by reprojection (resolve_marker_flips).
    """
    pts_ba = _corners_in_ba_frame(marker_poses, marker_side_m)
    centers0 = np.array([T[:3, 3] for T in marker_poses])
    normals0 = np.array([T[:3, 2] for T in marker_poses])
    normals0 = normals0 / np.linalg.norm(normals0, axis=1, keepdims=True)

    M = len(axes)
    w = np.ones(M) if weights is None else np.clip(weights, 0.0, None)
    sqrt_w = np.sqrt(w)
    # A tilt δ moves a marker corner by ~(s/2)·δ; scale the (≈δ) normal residual by
    # that length so plane and normal blocks share units (meters). use_normals=False
    # zeros this block, leaving a pure plane-distance objective.
    nw = normal_weight if use_normals else 0.0
    w_normal = sqrt_w * (0.5 * marker_side_m * nw)

    best_T = np.eye(4)
    best_cost = np.inf
    for R0 in _axis_aligned_rotations():
        t0 = _solve_translation(centers0, R0, axes, offs)
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T0[:3, 3] = t0
        xi0 = _se3_log(T0)
        sol = least_squares(
            _combined_residuals,
            xi0,
            args=(pts_ba, normals0, axes, signs, offs, sqrt_w, w_normal),
            method="trf",
            loss="soft_l1",
            f_scale=0.004,
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if sol.cost < best_cost:
            best_cost = float(sol.cost)
            best_T = _se3_exp(sol.x)
    return best_T, best_cost


def fit_box_frame(
    marker_poses: np.ndarray,
    W_mm: float,
    H_mm: float,
    D_mm: float,
    marker_side_m: float,
    max_assign_iter: int = 50,
    progress_cb: Callable[[int, int], None] | None = None,
    weights: np.ndarray | None = None,
    use_normals: bool = True,
) -> tuple[np.ndarray, list[str], dict]:
    """SE(3) box-frame fit with inferred face assignments and rigid markers.

    Only one global transform is optimized. Marker-to-marker relative positions
    and rotations remain unchanged; we choose the box pose/orientation that best
    places all marker corners onto box face planes.

    No face labels are required — assignment is inferred from geometry. With
    use_normals=False, both the assignment and the objective are pure plane-distance
    (orientation-agnostic "best possible fit"); per-marker flips are resolved later
    by reprojection (resolve_marker_flips).

    Args:
      marker_poses: (M,4,4) SE(3), meters, in BA-output frame.
      W_mm, H_mm, D_mm: physical box dimensions (mm).
      marker_side_m: physical marker side (m), kept for API parity.
      weights: (M,) per-marker confidence; sparse/noisy markers downweighted.
      use_normals: include the +Z↔face-normal term (default) or drop it.

    Returns:
      T_align: (4,4) SE(3), BA-output frame → box frame.
      faces:   list of face names per marker.
      info:    diagnostics dict.
    """
    W = W_mm / 1000.0
    H = H_mm / 1000.0
    D = D_mm / 1000.0
    specs = _face_specs(W, H, D)
    scale = max(W, H, D)

    M = len(marker_poses)
    centers0 = np.array([T[:3, 3] for T in marker_poses])          # (M,3)
    normals0 = np.array([T[:3, 2] for T in marker_poses])          # (M,3)
    normals0 = normals0 / np.linalg.norm(normals0, axis=1, keepdims=True)

    best_T = np.eye(4)
    best_assigns: list[tuple[str, int, float, float, np.ndarray]] = []
    best_cost = np.inf

    rotations = _axis_aligned_rotations()
    total_starts = len(rotations)
    for start_idx, R0 in enumerate(rotations, start=1):
        t0 = np.zeros(3, dtype=np.float64)
        prev_labels: tuple[str, ...] | None = None
        cur_T = np.eye(4)
        cur_T[:3, :3] = R0
        cur_T[:3, 3] = t0
        cur_assigns: list[tuple[str, int, float, float, np.ndarray]] = []

        for _ in range(max_assign_iter):
            R = cur_T[:3, :3]
            t = cur_T[:3, 3]
            c_r = centers0 @ R.T + t
            n_r = normals0 @ R.T
            cur_assigns = _assign_faces(c_r, n_r, specs, scale, use_normals=use_normals)
            labels = tuple(a[0] for a in cur_assigns)

            axes = np.array([a[1] for a in cur_assigns], dtype=np.int64)
            signs = np.array([a[2] for a in cur_assigns], dtype=np.float64)
            offs = np.array([a[3] for a in cur_assigns], dtype=np.float64)
            cur_T, cur_cost = _fit_fixed_faces(
                marker_poses, axes, offs, signs, marker_side_m,
                weights=weights, use_normals=use_normals)

            if labels == prev_labels:
                break
            prev_labels = labels

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_T = cur_T
            best_assigns = cur_assigns
        if progress_cb is not None:
            progress_cb(start_idx, total_starts)

    T_align = best_T
    R = T_align[:3, :3]
    t = T_align[:3, 3]
    centers_r = centers0 @ R.T + t
    normals_r = normals0 @ R.T
    target_n = np.array([a[4] for a in best_assigns])

    plane_resid_mm = np.empty(M)
    for i, a in enumerate(best_assigns):
        axis, off, sign = a[1], a[3], a[2]
        plane_resid_mm[i] = sign * (centers_r[i, axis] - off) * 1000.0

    cos_resid = np.clip(np.einsum("ij,ij->i", normals_r, target_n), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_resid))

    info = {
        "rms_plane_mm":   float(np.sqrt(np.mean(plane_resid_mm ** 2))),
        "max_plane_mm":   float(np.max(np.abs(plane_resid_mm))),
        "rms_normal_deg": float(np.sqrt(np.mean(angle_deg ** 2))),
        "max_normal_deg": float(np.max(angle_deg)),
        "final_cost":     float(best_cost),
        "plane_resid_mm":     plane_resid_mm,
        "normal_resid_deg":   angle_deg,
    }
    return T_align, [a[0] for a in best_assigns], info


def fit_box_frame_with_labels(
    marker_poses: np.ndarray,
    face_labels: list[str],
    W_mm: float,
    H_mm: float,
    D_mm: float,
    marker_side_m: float,
    weights: np.ndarray | None = None,
    normal_weight: float = 1.0,
) -> tuple[np.ndarray, list[str], dict]:
    """SE(3) box-frame fit with face assignments FIXED from user labels.

    Skips the 24-rotation multi-start + assignment alternation in fit_box_frame.
    Use this when the box config provides a 'face' key per marker — the geometry is
    pinned, and we only solve for one global SE(3) that maps BA frame → box frame.

    weights: (M,) per-marker confidence (e.g. n_obs / reproj-rms). Under-observed or
      high-residual markers are downweighted so they don't drag the global fit.
    normal_weight: relative pull of the normal-alignment term vs the plane term.
    """
    W = W_mm / 1000.0
    H = H_mm / 1000.0
    D = D_mm / 1000.0
    spec_by_name = {n: (ax, sg, off) for n, ax, sg, off in _face_specs(W, H, D)}
    for f in face_labels:
        if f not in spec_by_name:
            raise ValueError(f"Unknown face label '{f}'. "
                             f"Expected one of {list(spec_by_name)}.")

    M = len(marker_poses)
    axes  = np.array([spec_by_name[f][0] for f in face_labels], dtype=np.int64)
    signs = np.array([spec_by_name[f][1] for f in face_labels], dtype=np.float64)
    offs  = np.array([spec_by_name[f][2] for f in face_labels])

    centers0 = np.array([T[:3, 3] for T in marker_poses])
    normals0 = np.array([T[:3, 2] for T in marker_poses])
    normals0 = normals0 / np.linalg.norm(normals0, axis=1, keepdims=True)

    normals_box = np.zeros((M, 3))
    for i in range(M):
        normals_box[i, axes[i]] = spec_by_name[face_labels[i]][1]

    T_align, best_cost = _fit_fixed_faces(
        marker_poses, axes, offs, signs, marker_side_m,
        weights=weights, normal_weight=normal_weight)

    R = T_align[:3, :3]; t = T_align[:3, 3]
    pts_ba = _corners_in_ba_frame(marker_poses, marker_side_m)
    pts = pts_ba @ R.T + t
    centers_r = centers0 @ R.T + t
    normals_r = normals0 @ R.T

    plane_resid_per_corner_mm = np.empty((M, 4))
    plane_resid_mm = np.empty(M)
    for i in range(M):
        plane_resid_per_corner_mm[i] = (pts[i, :, axes[i]] - offs[i]) * 1000.0
        plane_resid_mm[i] = signs[i] * (centers_r[i, axes[i]] - offs[i]) * 1000.0

    cos_resid = np.clip(np.einsum("ij,ij->i", normals_r, normals_box), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_resid))

    info = {
        "rms_plane_mm":   float(np.sqrt(np.mean(plane_resid_per_corner_mm ** 2))),
        "max_plane_mm":   float(np.max(np.abs(plane_resid_per_corner_mm))),
        "rms_normal_deg": float(np.sqrt(np.mean(angle_deg ** 2))),
        "max_normal_deg": float(np.max(angle_deg)),
        "final_cost":     float(best_cost),
        "plane_resid_mm":     plane_resid_mm,
        "normal_resid_deg":   angle_deg,
    }
    return T_align, list(face_labels), info


def apply_box_fit(result, T_align: np.ndarray) -> None:
    """In-place: transform marker_poses + camera xis to express in box frame."""
    for i in range(result.n_markers):
        result.marker_poses[i] = T_align @ result.marker_poses[i]

    R = T_align[:3, :3]
    t = T_align[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    n_cams = result.n_cams
    cam_xis = result.x[: 6 * n_cams].reshape(n_cams, 6).copy()
    T_cams = _se3_exp_batch(cam_xis)
    for i in range(n_cams):
        T_new = T_cams[i] @ T_inv
        result.x[6 * i : 6 * i + 6] = _se3_log(T_new)


def _marker_reproj_rms(
    R: np.ndarray,
    c: np.ndarray,
    dets: list[tuple[int, np.ndarray]],
    T_cam_box: np.ndarray,
    corners_mkr: np.ndarray,
    K: np.ndarray,
) -> float:
    """RMS reprojection (px) of one marker's corners over all images that saw it."""
    if not dets:
        return 0.0
    T_mk = np.eye(4)
    T_mk[:3, :3] = R
    T_mk[:3, 3] = c
    se, n = 0.0, 0
    for cam_idx, obs in dets:
        P = T_cam_box[cam_idx] @ T_mk
        pc = (P[:3, :3] @ corners_mkr.T + P[:3, 3:4]).T          # (4,3)
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        proj = np.stack([u, v], axis=1)
        se += float(np.sum((proj - obs) ** 2))
        n += 4
    return float(np.sqrt(se / max(n, 1)))


def resolve_marker_flips(
    result,
    outward_normals: np.ndarray,
    K: np.ndarray,
    marker_side_m: float,
) -> list[int]:
    """Fix per-marker 180° flips left in the BA cloud (in-place, box frame).

    A flat marker's +Z must equal its outward face normal. Where BA settled on the
    flipped pose (+Z·N < 0) — a 180° rotation about an in-plane axis — the two valid
    un-flips differ by a 180° twist about the normal. We pick whichever reprojects
    best across ALL images that saw the marker, which is the multi-view information
    BA's near-frontal-dominated, Huber-capped objective failed to commit to.

    Markers that are merely tilted/garbage (+Z·N ≥ 0, e.g. seen once) are left as-is.
    Call AFTER apply_box_fit (needs camera poses + marker poses in box frame).
    Returns the list of marker IDs that were flipped back.
    """
    n_cams = result.n_cams
    T_cam_box = _se3_exp_batch(result.x[: 6 * n_cams].reshape(n_cams, 6))
    corners_mkr = marker_corners_mkr_frame(marker_side_m)

    dets_by_mk: dict[int, list[tuple[int, np.ndarray]]] = {}
    for cam_idx, mk_idx, obs in result.detection_list:
        dets_by_mk.setdefault(mk_idx, []).append((cam_idx, obs))

    Rx = np.diag([1.0, -1.0, -1.0])   # 180° about marker x: +Z→−Z, keeps twist A
    Ry = np.diag([-1.0, 1.0, -1.0])   # 180° about marker y: +Z→−Z, twist B (=A+180°)

    flipped: list[int] = []
    for i in range(result.n_markers):
        R = result.marker_poses[i][:3, :3]
        c = result.marker_poses[i][:3, 3]
        N = outward_normals[i]
        if float(R[:, 2] @ N) >= 0.0:
            continue                                            # not flipped
        dets = dets_by_mk.get(i, [])
        cand_a = R @ Rx
        cand_b = R @ Ry
        ra = _marker_reproj_rms(cand_a, c, dets, T_cam_box, corners_mkr, K)
        rb = _marker_reproj_rms(cand_b, c, dets, T_cam_box, corners_mkr, K)
        result.marker_poses[i][:3, :3] = cand_a if ra <= rb else cand_b
        flipped.append(result.marker_ids[i])

    return flipped
