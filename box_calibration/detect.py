"""Per-image ArUco detection on undistorted images.

Undistorts up front so the bundle adjustment can use a pure pinhole model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _detect_markers_compat(gray, aruco_dict, params):
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def detect_images(
    image_paths: list[Path],
    K: np.ndarray,
    dist: np.ndarray,
    aruco_dict_name: str,
    valid_ids: set[int],
    min_markers: int,
) -> list[tuple[Path, dict[int, np.ndarray], np.ndarray]]:
    """
    Detect ArUco markers in each image after undistortion.

    Returns list of (path, {marker_id: corners_px(4,2)}, undist_img) for
    accepted images (those with >= min_markers valid markers).
    """
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(getattr(cv2.aruco, aruco_dict_name))

    try:
        det_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        det_params = cv2.aruco.DetectorParameters_create()
    try:
        det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det_params.cornerRefinementWinSize = 5
        det_params.cornerRefinementMinAccuracy = 0.01
    except AttributeError:
        pass

    K_f32 = K.astype(np.float32)
    dist_f32 = dist.astype(np.float32)

    results: list[tuple[Path, dict[int, np.ndarray], np.ndarray]] = []
    n_rejected = 0

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"    WARN: cannot read {path}")
            n_rejected += 1
            continue

        img_ud = cv2.undistort(img, K_f32, dist_f32)
        gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)
        corners_all, ids_all = _detect_markers_compat(gray, aruco_dict, det_params)

        if ids_all is None:
            n_rejected += 1
            continue

        det: dict[int, np.ndarray] = {}
        for c_arr, mid in zip(corners_all, ids_all.ravel()):
            mid = int(mid)
            if mid in valid_ids:
                det[mid] = c_arr.reshape(4, 2).astype(np.float64)

        if len(det) < min_markers:
            n_rejected += 1
            continue

        results.append((path, det, img_ud))

    n_accepted = len(results)
    print(f"  Detection: {n_accepted} accepted, {n_rejected} skipped")
    if n_accepted == 0:
        raise RuntimeError(
            "No images accepted. Check image paths, marker IDs, min_markers, "
            "and that intrinsics match the camera."
        )
    return results
