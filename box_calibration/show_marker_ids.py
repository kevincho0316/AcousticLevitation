"""
Quick diagnostic: detect ArUco markers in all images and annotate with IDs.
Run from project root:
    python -m box_calibration.show_marker_ids --images-dir captures/boxConfig/img
Output images saved alongside originals as *_ids.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--dict", default="DICT_4X4_50")
    args = p.parse_args()

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        use_new_api = True
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        use_new_api = False

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
    images = sorted(p for ext in exts for p in args.images_dir.glob(ext))
    if not images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if use_new_api:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        vis = img.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            for corner_set, mid in zip(corners, ids.ravel()):
                c = corner_set.reshape(4, 2).mean(axis=0).astype(int)
                cv2.putText(vis, str(mid), tuple(c),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            print(f"{img_path.name}: IDs = {sorted(ids.ravel().tolist())}")
        else:
            print(f"{img_path.name}: no markers detected")

        out_dir = img_path.parent / "id_labels"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / (img_path.stem + "_ids.jpg")
        cv2.imwrite(str(out), vis)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
