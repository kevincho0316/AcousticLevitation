"""Co-visibility graph initialization for full self-calibration.

No nominal layout. Each marker's pose in the box frame is unknown a priori
and seeded purely from the images:

  1. Per (image, marker) detection: cv2.solvePnPGeneric with
     SOLVEPNP_IPPE_SQUARE → up to 2 candidate T_cam_marker poses
     (square-planar ambiguity).

  2. Build co-visibility graph: nodes = observed marker IDs,
     edges = pairs seen together.

  3. Pick anchor (most observations; ties → smallest id). Anchor's
     T_box_marker is fixed to identity → box frame coincides with
     the anchor marker's local frame.

  4. BFS from anchor. For each visited→unvisited edge, choose the
     pair of IPPE candidates (across shared images) that minimizes
     reprojection of both markers' corners in those images. Use the
     resulting T_box_marker estimates, averaged over shared images
     via Lie-mean.

  5. Disconnected components → drop with a warning.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from common.se3_utils import _average_se3

from .faces import marker_corners_mkr_frame


# ── Per-detection IPPE candidates ─────────────────────────────────────────────

def per_marker_ippe(
    detections: list[tuple[Path, dict[int, np.ndarray], np.ndarray]],
    K: np.ndarray,
    marker_side_m: float,
) -> list[dict[int, list[tuple[np.ndarray, float]]]]:
    """Per-image, per-marker IPPE-square candidates.

    Returns list (one per image) of {marker_id: [(T_cam_marker, reproj_rms_px), ...]}.
    Each marker yields 1 or 2 candidates ordered by reprojection error
    (best first, per OpenCV convention).
    """
    K_f32 = K.astype(np.float32)
    zero_dist = np.zeros(5, dtype=np.float32)
    obj_pts = marker_corners_mkr_frame(marker_side_m).astype(np.float32)

    out: list[dict[int, list[tuple[np.ndarray, float]]]] = []
    for _, det, _ in detections:
        per_img: dict[int, list[tuple[np.ndarray, float]]] = {}
        for mid, px in det.items():
            img_pts = px.astype(np.float32)
            ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj_pts, img_pts, K_f32, zero_dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or len(rvecs) == 0:
                continue
            cands: list[tuple[np.ndarray, float]] = []
            for rv, tv, e in zip(rvecs, tvecs, errs.ravel() if errs is not None else [0.0] * len(rvecs)):
                R, _ = cv2.Rodrigues(rv)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = tv.ravel()
                # IPPE-reported reprojection (already rms-ish); recompute for safety.
                proj = (K @ ((R @ obj_pts.T) + tv.reshape(3, 1)))
                z = proj[2]
                u = proj[0] / z
                v = proj[1] / z
                err_px = float(np.sqrt(np.mean((u - img_pts[:, 0]) ** 2 + (v - img_pts[:, 1]) ** 2)))
                cands.append((T, err_px))
            cands.sort(key=lambda c: c[1])
            per_img[mid] = cands
        out.append(per_img)
    return out


# ── Co-visibility graph ───────────────────────────────────────────────────────

def build_covis_graph(
    per_img_cands: list[dict[int, list[tuple[np.ndarray, float]]]],
) -> tuple[list[int], dict[int, dict[int, list[int]]], dict[int, int]]:
    """Return (sorted_marker_ids, adjacency, n_obs).

    adjacency[a][b] = list of image indices where both a and b were detected.
    n_obs[mid] = number of images containing marker mid.
    """
    n_obs: dict[int, int] = {}
    adj: dict[int, dict[int, list[int]]] = {}
    for img_idx, per_img in enumerate(per_img_cands):
        ids = list(per_img.keys())
        for mid in ids:
            n_obs[mid] = n_obs.get(mid, 0) + 1
            adj.setdefault(mid, {})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                adj[a].setdefault(b, []).append(img_idx)
                adj[b].setdefault(a, []).append(img_idx)
    return sorted(n_obs.keys()), adj, n_obs


def pick_anchor(
    n_obs: dict[int, int],
    override: int | None = None,
) -> int:
    """Auto-pick marker with most observations (ties → smallest id) or use override."""
    if override is not None:
        if override not in n_obs:
            raise ValueError(f"--anchor-marker-id={override} was never observed")
        return override
    # Sort by (-n_obs, id) → max obs, tie-break smallest id.
    return sorted(n_obs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ── BFS seeding ───────────────────────────────────────────────────────────────

def _project_rms(
    T_cam_box: np.ndarray,
    T_box_mk: np.ndarray,
    corners_mkr_h: np.ndarray,   # (4,4) homogeneous columns
    obs_px: np.ndarray,           # (4,2)
    K: np.ndarray,
) -> float:
    pts_cam = (T_cam_box @ T_box_mk @ corners_mkr_h)        # (4,4) homogeneous columns
    x = pts_cam[0, :]
    y = pts_cam[1, :]
    z = pts_cam[2, :]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    proj = np.stack([u, v], axis=1)
    return float(np.sqrt(np.mean((proj - obs_px) ** 2)))


def init_marker_poses(
    per_img_cands: list[dict[int, list[tuple[np.ndarray, float]]]],
    detections: list,
    adj: dict[int, dict[int, list[int]]],
    anchor: int,
    K: np.ndarray,
    marker_side_m: float,
) -> dict[int, np.ndarray]:
    """BFS from anchor; for each frontier edge, disambiguate IPPE candidates
    by joint reprojection across shared images. Returns {mid: T_box_marker(4,4)}.
    Anchor maps to identity. Disconnected markers are simply absent.
    """
    corners_mkr = marker_corners_mkr_frame(marker_side_m)          # (4,3)
    corners_mkr_h = np.vstack([corners_mkr.T, np.ones((1, 4))])    # (4,4) hom columns

    poses: dict[int, np.ndarray] = {anchor: np.eye(4)}
    visited: set[int] = {anchor}
    queue: deque[int] = deque([anchor])

    while queue:
        a = queue.popleft()
        T_box_a = poses[a]
        for b, img_idxs in adj.get(a, {}).items():
            if b in visited:
                continue
            best_score = float("inf")
            best_relposes: list[np.ndarray] = []
            best_kl: tuple[int, int] | None = None

            cands_a = [per_img_cands[i][a] for i in img_idxs]
            cands_b = [per_img_cands[i][b] for i in img_idxs]

            obs_a = [detections[i][1][a] for i in img_idxs]
            obs_b = [detections[i][1][b] for i in img_idxs]

            n_cand_a = max(len(c) for c in cands_a)
            n_cand_b = max(len(c) for c in cands_b)

            for k in range(n_cand_a):
                for l in range(n_cand_b):
                    score = 0.0
                    rel_per_img: list[np.ndarray] = []
                    ok = True
                    for ci in range(len(img_idxs)):
                        if k >= len(cands_a[ci]) or l >= len(cands_b[ci]):
                            ok = False
                            break
                        T_cam_a, _ = cands_a[ci][k]
                        T_cam_b, _ = cands_b[ci][l]
                        # Camera pose implied by anchor (or already-placed) marker a:
                        T_cam_box = T_cam_a @ np.linalg.inv(T_box_a)
                        # Marker b in box frame:
                        T_box_b_candidate = np.linalg.inv(T_cam_box) @ T_cam_b
                        rel_per_img.append(T_box_b_candidate)
                        # Score: reprojection of both markers in this image
                        score += _project_rms(T_cam_box, T_box_a, corners_mkr_h, obs_a[ci], K)
                        score += _project_rms(T_cam_box, T_box_b_candidate, corners_mkr_h, obs_b[ci], K)
                    if not ok:
                        continue
                    if score < best_score:
                        best_score = score
                        best_relposes = rel_per_img
                        best_kl = (k, l)

            if best_kl is None or not best_relposes:
                continue

            T_box_b = _average_se3(best_relposes)
            poses[b] = T_box_b
            visited.add(b)
            queue.append(b)

    return poses
