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


def _save_detect_debug(
    path: Path,
    img_ud: np.ndarray,
    det: dict[int, np.ndarray],
    all_corners: list,
    all_ids,
    valid_ids: set[int],
    min_markers: int,
    debug_dir: Path,
) -> None:
    vis = img_ud.copy()
    accepted = len(det) >= min_markers

    # Draw all detected markers (dim for invalid IDs)
    if all_ids is not None:
        for c_arr, mid in zip(all_corners, all_ids.ravel()):
            mid = int(mid)
            pts = c_arr.reshape(4, 2).astype(np.int32)
            color = (0, 220, 0) if mid in valid_ids else (80, 80, 80)
            cv2.polylines(vis, [pts], True, color, 2)
            cv2.putText(vis, str(mid), tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for pt in pts:
                cv2.circle(vis, tuple(pt), 5, color, -1)

    status = "OK" if accepted else f"SKIP(<{min_markers} markers)"
    color_status = (0, 220, 0) if accepted else (0, 60, 220)
    cv2.putText(vis, f"{path.name} | {len(det)} markers | {status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

    out_name = f"detect_{path.stem}.jpg"
    cv2.imwrite(str(debug_dir / out_name), vis)


def detect_images(
    image_paths: list[Path],
    K: np.ndarray,
    dist: np.ndarray,
    aruco_dict_name: str,
    valid_ids: set[int],
    min_markers: int,
    debug_dir: Path | None = None,
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

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

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

        det: dict[int, np.ndarray] = {}
        if ids_all is not None:
            for c_arr, mid in zip(corners_all, ids_all.ravel()):
                mid = int(mid)
                if mid in valid_ids:
                    det[mid] = c_arr.reshape(4, 2).astype(np.float64)

        if debug_dir is not None:
            _save_detect_debug(
                path, img_ud, det,
                corners_all if ids_all is not None else [],
                ids_all, valid_ids, min_markers, debug_dir,
            )

        if ids_all is None or len(det) < min_markers:
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
