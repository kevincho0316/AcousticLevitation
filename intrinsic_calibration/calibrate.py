"""
Per-camera intrinsic calibration using a ChArUco board.

Usage:
    python -m intrinsic_calibration.calibrate \\
        --camera-id cam_front \\
        --images-dir images/cam_front_charuco/ \\
        --output calibration/cam_front_intrinsics.yaml \\
        [--squares-x 9] [--squares-y 6] \\
        [--square-length 0.04] [--marker-length 0.02] \\
        [--dict DICT_5X5_100] [--max-reproj-px 1.0]

Capture 30–50 images of the ChArUco board at varied poses (angles, distances,
positions across the full frame) before running this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CameraIntrinsics
from common.io_utils import save_intrinsics


# ── ArUco API compatibility ───────────────────────────────────────────────────

def _get_aruco_dict(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_charuco_board(squares_x: int, squares_y: int,
                         square_len: float, marker_len: float,
                         aruco_dict) -> cv2.aruco.CharucoBoard:
    try:
        # OpenCV 4.7+
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_len, marker_len, aruco_dict
        )
    except TypeError:
        # OpenCV < 4.7
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_len, marker_len, aruco_dict
        )
    return board


def _detect_charuco(image: np.ndarray, board: cv2.aruco.CharucoBoard,
                    aruco_dict, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    try:
        # OpenCV 4.7+ path
        charuco_detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, m_corners, m_ids = charuco_detector.detectBoard(gray)
    except AttributeError:
        # Legacy path
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if m_ids is None or len(m_ids) == 0:
            return None, None
        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, board
        )

    if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
        return None, None
    return ch_corners, ch_ids


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_camera(
    image_paths: list[Path],
    camera_id: str,
    squares_x: int = 9,
    squares_y: int = 6,
    square_length: float = 0.04,
    marker_length: float = 0.02,
    dict_name: str = "DICT_5X5_100",
    max_reproj_px: float = 1.0,
) -> CameraIntrinsics:
    aruco_dict = _get_aruco_dict(dict_name)
    board = _make_charuco_board(squares_x, squares_y, square_length, marker_length, aruco_dict)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print(f"Processing {len(image_paths)} images …")
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  SKIP {path.name}: cannot read")
            continue
        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print(f"  SKIP {path.name}: resolution mismatch ({w}×{h} vs {image_size})")
            continue

        corners, ids = _detect_charuco(img, board, aruco_dict, params)
        if corners is None:
            print(f"  SKIP {path.name}: insufficient ChArUco corners detected")
            continue
        all_corners.append(corners)
        all_ids.append(ids)
        print(f"  OK   {path.name}: {len(ids)} corners")

    if len(all_corners) < 5:
        raise RuntimeError(
            f"Only {len(all_corners)} usable images (need ≥ 5). "
            "Capture more views at varied angles and distances."
        )

    assert image_size is not None
    try:
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None
        )
    except cv2.error as exc:
        raise RuntimeError(f"Calibration failed: {exc}") from exc

    print(f"\nInitial calibration: mean reprojection error = {ret:.4f} px")

    # Reject images with high per-image reprojection error and re-calibrate.
    good_corners: list[np.ndarray] = []
    good_ids: list[np.ndarray] = []
    rejected = 0
    for corners, ids_, rvec, tvec in zip(all_corners, all_ids, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            board.getChessboardCorners()[ids_.ravel()], rvec, tvec, K, dist
        )
        observed = corners.reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(observed - projected, axis=1)))
        if err <= max_reproj_px:
            good_corners.append(corners)
            good_ids.append(ids_)
        else:
            rejected += 1

    if rejected > 0:
        print(f"Rejected {rejected} images with reprojection error > {max_reproj_px} px. Re-calibrating …")
        if len(good_corners) < 5:
            raise RuntimeError("Too few good images after outlier rejection.")
        ret, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            good_corners, good_ids, board, image_size, None, None
        )
        print(f"Final calibration: mean reprojection error = {ret:.4f} px  ({len(good_corners)} images)")

    return CameraIntrinsics(
        camera_id=camera_id,
        K=K,
        dist=dist.ravel(),
        resolution=image_size,
        reprojection_error=float(ret),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChArUco intrinsic calibration")
    p.add_argument("--camera-id", required=True)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--squares-x", type=int, default=9)
    p.add_argument("--squares-y", type=int, default=6)
    p.add_argument("--square-length", type=float, default=0.04,
                   help="ChArUco square side length in meters")
    p.add_argument("--marker-length", type=float, default=0.02,
                   help="Embedded marker side length in meters")
    p.add_argument("--dict", default="DICT_5X5_100",
                   help="ArUco dictionary name (e.g. DICT_4X4_50)")
    p.add_argument("--max-reproj-px", type=float, default=1.0,
                   help="Per-image reprojection error threshold for outlier rejection")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    image_paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    )
    if not image_paths:
        sys.exit(f"No images found in {args.images_dir}")

    intr = calibrate_camera(
        image_paths=image_paths,
        camera_id=args.camera_id,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_name=args.dict,
        max_reproj_px=args.max_reproj_px,
    )
    save_intrinsics(intr, args.output)
    print(f"\nSaved intrinsics to {args.output}")
    print(f"  K =\n{intr.K}")
    print(f"  dist = {intr.dist}")
    print(f"  resolution = {intr.resolution}")
    print(f"  reprojection_error = {intr.reprojection_error:.4f} px")


if __name__ == "__main__":
    main()
