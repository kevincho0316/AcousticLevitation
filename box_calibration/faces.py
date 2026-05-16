"""Face convention table, nominal SE(3) marker poses, and BoxModel.

Box frame: X=right(width), Y=up(height), Z=front→back(depth).
Origin: front-bottom-left corner.

Marker local frame:
  +X = r_vec (rightward when viewed from outside the box)
  +Y = u_vec (upward when viewed from outside the box)
  +Z = outward face normal (toward the camera)

ArUco corner order in marker frame (z=0 plane):
  0=TL(-s/2,+s/2,0)  1=TR(+s/2,+s/2,0)
  2=BR(+s/2,-s/2,0)  3=BL(-s/2,-s/2,0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FACE_TABLE: dict[str, dict] = {
    "front": {
        "normal": np.array([ 0,  0,  1], dtype=np.float64),  # cross(r,u) = into box
        "r":      np.array([ 1,  0,  0], dtype=np.float64),
        "u":      np.array([ 0,  1,  0], dtype=np.float64),
        "c":      lambda W, D, H: np.array([W / 2, H / 2, 0.0]),
    },
    "back": {
        "normal": np.array([ 0,  0, -1], dtype=np.float64),  # cross(r,u) = into box
        "r":      np.array([-1,  0,  0], dtype=np.float64),
        "u":      np.array([ 0,  1,  0], dtype=np.float64),
        "c":      lambda W, D, H: np.array([W / 2, H / 2, D]),
    },
    "right": {
        "normal": np.array([ 1,  0,  0], dtype=np.float64),
        "r":      np.array([ 0,  0, -1], dtype=np.float64),
        "u":      np.array([ 0,  1,  0], dtype=np.float64),
        "c":      lambda W, D, H: np.array([W, H / 2, D / 2]),
    },
    "left": {
        "normal": np.array([-1,  0,  0], dtype=np.float64),
        "r":      np.array([ 0,  0,  1], dtype=np.float64),
        "u":      np.array([ 0,  1,  0], dtype=np.float64),
        "c":      lambda W, D, H: np.array([0.0, H / 2, D / 2]),
    },
    "top": {
        "normal": np.array([ 0,  1,  0], dtype=np.float64),
        "r":      np.array([ 1,  0,  0], dtype=np.float64),
        "u":      np.array([ 0,  0, -1], dtype=np.float64),
        "c":      lambda W, D, H: np.array([W / 2, H, D / 2]),
    },
    "bottom": {
        "normal": np.array([ 0, -1,  0], dtype=np.float64),
        "r":      np.array([ 1,  0,  0], dtype=np.float64),
        "u":      np.array([ 0,  0,  1], dtype=np.float64),
        "c":      lambda W, D, H: np.array([W / 2, 0.0, D / 2]),
    },
}


def nominal_pose(face: str, center_m: np.ndarray) -> np.ndarray:
    """4×4 SE(3) T_box_marker: marker local frame → box frame."""
    ft = FACE_TABLE[face]
    R = np.column_stack([ft["r"], ft["u"], ft["normal"]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center_m
    return T


def marker_corners_mkr_frame(s: float) -> np.ndarray:
    """(4,3) ArUco corner positions in marker local frame (all at z=0)."""
    h = s / 2.0
    return np.array([
        [-h,  h, 0],  # 0: TL
        [ h,  h, 0],  # 1: TR
        [ h, -h, 0],  # 2: BR
        [-h, -h, 0],  # 3: BL
    ], dtype=np.float64)


def nominal_center_m(marker: dict, W: float, D: float, H: float) -> np.ndarray:
    """Best available nominal center in box frame (meters).

    Priority: center_box_mm > centroid of corners_box_frame > face center.
    Using corners_box_frame centroid avoids huge offsets when markers were
    already placed off-center from the face center.
    """
    if "center_box_mm" in marker:
        return np.array(marker["center_box_mm"], dtype=np.float64) / 1000.0
    if "corners_box_frame" in marker:
        c = np.array(marker["corners_box_frame"], dtype=np.float64) / 1000.0
        return (c[0] + c[2]) / 2.0  # diagonal midpoint (TL + BR)
    return FACE_TABLE[marker["face"]]["c"](W, D, H)


@dataclass
class BoxModel:
    ids: list[int]
    faces: list[str]
    nominal_poses: np.ndarray   # (N,4,4) T_box_marker_nominal
    corners_mkr: np.ndarray     # (4,3) in marker local frame — same for all markers
    centers_m: np.ndarray       # (N,3) nominal centers in box frame (m)


def build_box_model(box_cfg: dict) -> BoxModel:
    dims = box_cfg["box_dimensions"]
    W = float(dims["width_mm"]) / 1000.0
    D = float(dims["depth_mm"]) / 1000.0
    H = float(dims["height_mm"]) / 1000.0
    s = float(box_cfg["marker_side_mm"]) / 1000.0

    ids, faces, poses, centers = [], [], [], []
    for m in box_cfg["markers"]:
        if "face" not in m:
            raise ValueError(f"Marker {m.get('id','?')} has no 'face' key in box config.")
        face = m["face"]
        ctr = nominal_center_m(m, W, D, H)
        ids.append(int(m["id"]))
        faces.append(face)
        poses.append(nominal_pose(face, ctr))
        centers.append(ctr)

    return BoxModel(
        ids=ids,
        faces=faces,
        nominal_poses=np.stack(poses),
        corners_mkr=marker_corners_mkr_frame(s),
        centers_m=np.stack(centers),
    )
