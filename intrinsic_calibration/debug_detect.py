"""
Quick diagnostic: try detecting ChArUco corners with various board configs.
Usage: python -m intrinsic_calibration.debug_detect --image path/to/image.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
    "DICT_6X6_50", "DICT_6X6_100",
    "DICT_ARUCO_ORIGINAL",
]
BOARD_SIZES = [
    (5, 7), (7, 5), (9, 6), (6, 9), (5, 5), (8, 6), (6, 8),
    (8, 11), (11, 8), (10, 9), (9, 10), (7, 13), (13, 7),
    (6, 15), (10, 7), (7, 10), (12, 8), (8, 12), (11, 9), (9, 11),
]


def try_detect(gray, squares_x, squares_y, dict_name):
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    except AttributeError:
        return 0

    try:
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), 0.04, 0.02, aruco_dict)
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, 0.04, 0.02, aruco_dict)

    try:
        det = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, m_corners, m_ids = det.detectBoard(gray)
        n_markers = len(m_ids) if m_ids is not None else 0
        n_corners = len(ch_ids) if ch_ids is not None else 0
    except AttributeError:
        try:
            params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            params = cv2.aruco.DetectorParameters()
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        n_markers = len(m_ids) if m_ids is not None else 0
        if n_markers == 0:
            return 0
        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, gray, board)
        n_corners = len(ch_ids) if ch_ids is not None else 0

    return n_markers, n_corners


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Image: {args.image}  size: {img.shape[1]}x{img.shape[0]}")
    print()

    best = []
    for dict_name in DICTS:
        for sx, sy in BOARD_SIZES:
            result = try_detect(gray, sx, sy, dict_name)
            if result and result[0] > 0:
                best.append((result[0], result[1], dict_name, sx, sy))

    # Raw marker scan with best dict
    for dict_name in DICTS:
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        except AttributeError:
            continue
        try:
            params = cv2.aruco.DetectorParameters()
        except AttributeError:
            params = cv2.aruco.DetectorParameters_create()
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if m_ids is not None and len(m_ids) > 0:
            ids_flat = sorted(m_ids.ravel().tolist())
            print(f"Raw markers [{dict_name}]: count={len(ids_flat)}  max_id={max(ids_flat)}")
            print(f"  IDs: {ids_flat[:60]}{'...' if len(ids_flat)>60 else ''}")
            print()
            break

    if not best:
        print("No ChArUco corners matched any board size.")
        print("Check: correct board dimensions, image not blurry, lighting OK.")
    else:
        best.sort(reverse=True)
        print(f"{'markers':>8}  {'corners':>8}  {'dict':<25}  board")
        print("-" * 60)
        for n_m, n_c, d, sx, sy in best[:15]:
            print(f"{n_m:>8}  {n_c:>8}  {d:<25}  {sx}x{sy}")


if __name__ == "__main__":
    main()
