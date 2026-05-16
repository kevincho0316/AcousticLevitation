"""Seed camera poses via solvePnP using current best-known marker corners."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .faces import BoxModel


def init_camera_poses(
    detections: list[tuple[Path, dict[int, np.ndarray], np.ndarray]],
    box_model: BoxModel,
    corners_box_m: dict[int, np.ndarray],
    K: np.ndarray,
) -> list[np.ndarray | None]:
    """
    Per-image solvePnP using all detected markers' best-known corner positions.

    corners_box_m: {marker_id: (4,3) corners in box frame (meters)} — the
    current best estimate, e.g. from corners_box_frame_m in the loaded config.

    Returns list of T_cam_box (4×4 float64) or None per image.
    """
    K_f32 = K.astype(np.float32)
    zero_dist = np.zeros(5, dtype=np.float32)

    poses: list[np.ndarray | None] = []
    n_failed = 0

    for path, det, _ in detections:
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []

        for mid, px in det.items():
            if mid not in corners_box_m:
                continue
            obj_pts.append(corners_box_m[mid].astype(np.float32))
            img_pts.append(px.astype(np.float32))

        if not obj_pts:
            poses.append(None)
            n_failed += 1
            continue

        obj_all = np.concatenate(obj_pts)  # (4k,3)
        img_all = np.concatenate(img_pts)  # (4k,2)

        ok, rvec, tvec = cv2.solvePnP(
            obj_all, img_all, K_f32, zero_dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            poses.append(None)
            n_failed += 1
            continue

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        poses.append(T)

    n_ok = len(poses) - n_failed
    print(f"  Init poses: {n_ok}/{len(detections)} succeeded")
    if n_ok == 0:
        raise RuntimeError(
            "All solvePnP calls failed. Check that corners_box_frame in box.yaml "
            "is a reasonable initial estimate and marker IDs match."
        )
    return poses
