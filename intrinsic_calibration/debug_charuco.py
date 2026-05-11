"""
Verbose diagnostic: step-by-step ChArUco detection on a single image.
Usage: python -m intrinsic_calibration.debug_charuco --image path/to/image.jpg
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--squares-x", type=int, default=8)
    p.add_argument("--squares-y", type=int, default=11)
    p.add_argument("--square-length", type=float, default=0.015)
    p.add_argument("--marker-length", type=float, default=0.011)
    p.add_argument("--dict", default="DICT_4X4_50")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read {args.image}")

    print(f"OpenCV version : {cv2.__version__}")
    print(f"Image loaded   : {img.shape[1]}x{img.shape[0]}  ({args.image})")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    try:
        board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            args.square_length, args.marker_length, aruco_dict
        )
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(
            args.squares_x, args.squares_y,
            args.square_length, args.marker_length, aruco_dict
        )

    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    # Step 1: detectMarkers
    m_corners, m_ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    n_markers = len(m_ids) if m_ids is not None else 0
    print(f"\n[Step 1] detectMarkers : {n_markers} markers  (rejected={len(rejected) if rejected else 0})")
    if m_ids is not None:
        print(f"         IDs           : {sorted(m_ids.ravel().tolist())}")

    if n_markers == 0:
        print("No markers — wrong dict or board not in frame.")
        return

    # Step 2: interpolateCornersCharuco (legacy)
    try:
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, board
        )
        n_corners = len(ch_ids) if ch_ids is not None else 0
        print(f"\n[Step 2] interpolateCornersCharuco : retval={retval}  corners={n_corners}")
        if ch_ids is not None and len(ch_ids) > 0:
            print(f"         corner IDs : {sorted(ch_ids.ravel().tolist()[:20])}")
    except Exception as e:
        print(f"\n[Step 2] interpolateCornersCharuco FAILED: {e}")
        retval, ch_corners, ch_ids = 0, None, None

    # Step 3: new API CharucoDetector
    try:
        det = cv2.aruco.CharucoDetector(board)
        ch2_corners, ch2_ids, m2_corners, m2_ids = det.detectBoard(gray)
        n2 = len(ch2_ids) if ch2_ids is not None else 0
        n2m = len(m2_ids) if m2_ids is not None else 0
        print(f"\n[Step 3] CharucoDetector.detectBoard : corners={n2}  markers={n2m}")
    except AttributeError:
        print("\n[Step 3] CharucoDetector not available (OpenCV < 4.7)")
    except Exception as e:
        print(f"\n[Step 3] CharucoDetector FAILED: {e}")

    # Draw and save debug image
    vis = img.copy()
    cv2.aruco.drawDetectedMarkers(vis, m_corners, m_ids)
    if ch_corners is not None and ch_ids is not None and len(ch_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
    out = Path(args.image).stem + "_debug.jpg"
    cv2.imwrite(out, vis)
    print(f"\nDebug image saved: {out}")

def test_synthetic(squares_x, squares_y, square_length, marker_length, dict_name):
    """Generate a board image and test detection on it."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    try:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, aruco_dict
        )
    except TypeError:
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length, marker_length, aruco_dict
        )

    img = board.generateImage((squares_x * 80, squares_y * 80), marginSize=20, borderBits=1)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("synthetic_board.jpg", img)

    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    n_m = len(m_ids) if m_ids is not None else 0

    try:
        retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, board
        )
        n_c = len(ch_ids) if ch_ids is not None else 0
    except Exception as e:
        retval, n_c = 0, 0
        print(f"  interpolateCornersCharuco error: {e}")

    try:
        det = cv2.aruco.CharucoDetector(board)
        ch2_c, ch2_ids, _, _ = det.detectBoard(gray)
        n_c2 = len(ch2_ids) if ch2_ids is not None else 0
    except Exception:
        n_c2 = -1

    print(f"\n[Synthetic board test]  markers={n_m}  legacy_corners={n_c}  new_api_corners={n_c2}")
    print(f"  Saved synthetic_board.jpg")
    expected_markers = (squares_x * squares_y) // 2
    expected_corners = (squares_x - 1) * (squares_y - 1)
    print(f"  Expected: markers={expected_markers}  corners={expected_corners}")


if __name__ == "__main__":
    import sys as _sys
    if "--synthetic" in _sys.argv:
        _sys.argv.remove("--synthetic")
        p = argparse.ArgumentParser()
        p.add_argument("--squares-x", type=int, default=8)
        p.add_argument("--squares-y", type=int, default=11)
        p.add_argument("--square-length", type=float, default=0.015)
        p.add_argument("--marker-length", type=float, default=0.011)
        p.add_argument("--dict", default="DICT_4X4_50")
        args = p.parse_args()
        test_synthetic(args.squares_x, args.squares_y, args.square_length, args.marker_length, args.dict)
    else:
        main()
