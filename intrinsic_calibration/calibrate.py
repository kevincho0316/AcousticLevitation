"""
Per-camera intrinsic calibration using a ChArUco board.

Usage:
    python -m intrinsic_calibration.calibrate \\
        --camera-id cam_front \\
        --images-dir images/cam_front_charuco/ \\
        --output calibration/cam_front_intrinsics.yaml \\
        [--squares-x 8] [--squares-y 11] \\
        [--square-length 0.015] [--marker-length 0.011] \\
        [--dict DICT_4X4_50] [--max-reproj-px 1.0]

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
from common.io_utils import load_yaml, save_intrinsics


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

    ch_corners, ch_ids = None, None

    m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if m_ids is not None and len(m_ids) > 0:
        try:
            _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                m_corners, m_ids, gray, board
            )
        except cv2.error:
            pass

    if ch_corners is None or ch_ids is None or len(ch_ids) < 6:
        try:
            charuco_detector = cv2.aruco.CharucoDetector(board)
            ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
        except AttributeError:
            pass

    if ch_corners is None or ch_ids is None or len(ch_ids) < 6:
        return None, None
    return ch_corners, ch_ids


def _normalize_chessboard_corners(corners: np.ndarray, inner_x: int, inner_y: int) -> np.ndarray:
    """Normalize corner ordering so corner[0] is always top-left of board in image.
    OpenCV findChessboardCorners can return corners in different directions depending
    on image/board orientation, causing inconsistent 2D-3D correspondences."""
    pts = corners.reshape(inner_y, inner_x, 2)
    # Flip rows if row-0 mean y is greater than last-row mean y (bottom-to-top ordering)
    if pts[0].mean(axis=0)[1] > pts[-1].mean(axis=0)[1]:
        pts = pts[::-1]
    # Flip cols if col-0 mean x is greater than last-col mean x (right-to-left ordering)
    if pts[:, 0].mean(axis=0)[0] > pts[:, -1].mean(axis=0)[0]:
        pts = pts[:, ::-1]
    return pts.reshape(-1, 1, 2).astype(np.float32)


def _detect_chessboard(image: np.ndarray, inner_x: int, inner_y: int) -> np.ndarray | None:
    """Fallback: find chessboard corners when ChArUco interpolation fails (OpenCV 4.10+ bug)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (inner_x, inner_y), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return _normalize_chessboard_corners(corners, inner_x, inner_y)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_camera(
    image_paths: list[Path],
    camera_id: str,
    squares_x: int = 8,
    squares_y: int = 11,
    square_length: float = 0.015,
    marker_length: float = 0.011,
    dict_name: str = "DICT_4X4_50",
    max_reproj_px: float = 1.0,
) -> CameraIntrinsics:
    aruco_dict = _get_aruco_dict(dict_name)
    board = _make_charuco_board(squares_x, squares_y, square_length, marker_length, aruco_dict)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    inner_x, inner_y = squares_x - 1, squares_y - 1
    charuco_corners: list[np.ndarray] = []
    charuco_ids: list[np.ndarray] = []
    chess_corners: list[np.ndarray] = []
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
            if sorted([w, h]) == sorted(image_size):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                h, w = img.shape[:2]
            else:
                print(f"  SKIP {path.name}: resolution mismatch ({w}×{h} vs {image_size})")
                continue

        corners, ids = _detect_charuco(img, board, aruco_dict, params)
        if corners is not None:
            charuco_corners.append(corners)
            charuco_ids.append(ids)
            print(f"  OK   {path.name}: {len(ids)} ChArUco corners")
            continue

        # ChArUco failed — try plain chessboard (OpenCV 4.10+ interpolation bug workaround)
        cb_corners = _detect_chessboard(img, inner_x, inner_y)
        if cb_corners is not None:
            chess_corners.append(cb_corners)
            print(f"  OK   {path.name}: {inner_x*inner_y} chessboard corners (fallback)")
        else:
            print(f"  SKIP {path.name}: no corners detected")

    assert image_size is not None

    # ── ChArUco path ──────────────────────────────────────────────────────────
    if len(charuco_corners) >= 5:
        try:
            ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charuco_corners, charuco_ids, board, image_size, None, None
            )
        except cv2.error as exc:
            raise RuntimeError(f"Calibration failed: {exc}") from exc

        print(f"\nInitial calibration (ChArUco): mean reprojection error = {ret:.4f} px")

        good_corners: list[np.ndarray] = []
        good_ids: list[np.ndarray] = []
        rejected = 0
        for corners, ids_, rvec, tvec in zip(charuco_corners, charuco_ids, rvecs, tvecs):
            projected, _ = cv2.projectPoints(
                board.getChessboardCorners()[ids_.ravel()], rvec, tvec, K, dist
            )
            err = float(np.mean(np.linalg.norm(
                corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
            )))
            if err <= max_reproj_px:
                good_corners.append(corners)
                good_ids.append(ids_)
            else:
                rejected += 1

        if rejected > 0:
            print(f"Rejected {rejected} images with error > {max_reproj_px} px. Re-calibrating …")
            if len(good_corners) < 5:
                raise RuntimeError("Too few good images after outlier rejection.")
            ret, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
                good_corners, good_ids, board, image_size, None, None
            )
            print(f"Final calibration: {ret:.4f} px  ({len(good_corners)} images)")

    # ── Chessboard fallback path ──────────────────────────────────────────────
    elif len(chess_corners) >= 5:
        print(f"\nUsing chessboard fallback ({len(chess_corners)} images) - ChArUco interpolation unavailable")
        objp = np.zeros((inner_x * inner_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:inner_x, 0:inner_y].T.reshape(-1, 2) * square_length

        obj_points = [objp] * len(chess_corners)
        try:
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, chess_corners, image_size, None, None
            )
        except cv2.error as exc:
            raise RuntimeError(f"Calibration failed: {exc}") from exc

        print(f"\nInitial calibration (chessboard): mean reprojection error = {ret:.4f} px")

        good_pts: list[np.ndarray] = []
        good_obj: list[np.ndarray] = []
        rejected = 0
        for corners_2d, rvec, tvec in zip(chess_corners, rvecs, tvecs):
            projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            err = float(np.mean(np.linalg.norm(
                corners_2d.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
            )))
            if err <= max_reproj_px:
                good_pts.append(corners_2d)
                good_obj.append(objp)
            else:
                rejected += 1

        if rejected > 0:
            print(f"Rejected {rejected} images with error > {max_reproj_px} px. Re-calibrating …")
            if len(good_pts) < 5:
                raise RuntimeError("Too few good images after outlier rejection.")
            ret, K, dist, _, _ = cv2.calibrateCamera(
                good_obj, good_pts, image_size, None, None
            )
            print(f"Final calibration: {ret:.4f} px  ({len(good_pts)} images)")

    else:
        total = len(charuco_corners) + len(chess_corners)
        raise RuntimeError(
            f"Only {total} usable images (need ≥ 5). "
            "Capture more views at varied angles and distances."
        )

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
    p.add_argument("--camera-id", required=True,
                   help="Camera ID (must match 'id' field in cameras.yaml)")
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--cameras-config", type=Path, default=None,
                   help="cameras.yaml — derives --output from the camera's intrinsics_file")
    p.add_argument("--output", type=Path, default=None,
                   help="Output YAML path (required if --cameras-config is not provided)")
    p.add_argument("--squares-x", type=int, default=8)
    p.add_argument("--squares-y", type=int, default=11)
    p.add_argument("--square-length", type=float, default=0.015,
                   help="ChArUco square side length in meters")
    p.add_argument("--marker-length", type=float, default=0.011,
                   help="Embedded marker side length in meters")
    p.add_argument("--dict", default="DICT_4X4_50",
                   help="ArUco dictionary name (e.g. DICT_4X4_50)")
    p.add_argument("--max-reproj-px", type=float, default=1.0,
                   help="Per-image reprojection error threshold for outlier rejection")
    return p.parse_args()


def _resolve_output(args: argparse.Namespace) -> Path:
    """Return output path: explicit --output, or derived from cameras.yaml intrinsics_file."""
    if args.output is not None:
        return args.output
    if args.cameras_config is None:
        sys.exit("Provide --output or --cameras-config to derive the output path.")
    cam_cfg = load_yaml(args.cameras_config)
    matches = [c for c in cam_cfg["cameras"] if c["id"] == args.camera_id]
    if not matches:
        sys.exit(
            f"Camera '{args.camera_id}' not found in {args.cameras_config}. "
            f"Available: {[c['id'] for c in cam_cfg['cameras']]}"
        )
    return Path(matches[0]["intrinsics_file"])


def main() -> None:
    args = _parse_args()
    output = _resolve_output(args)

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
    save_intrinsics(intr, output)
    print(f"\nSaved intrinsics to {output}")
    print(f"  K =\n{intr.K}")
    print(f"  dist = {intr.dist}")
    print(f"  resolution = {intr.resolution}")
    print(f"  reprojection_error = {intr.reprojection_error:.4f} px")


if __name__ == "__main__":
    main()
