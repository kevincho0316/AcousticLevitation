"""Generate synthetic camera poses around a box."""
from __future__ import annotations

import numpy as np

from common import CameraIntrinsics
from synthetic_tests.synth.scene import SyntheticScene

_FACE_NORMALS: dict[str, np.ndarray] = {
    "front":  np.array([ 0.0,  0.0, -1.0]),
    "back":   np.array([ 0.0,  0.0,  1.0]),
    "right":  np.array([ 1.0,  0.0,  0.0]),
    "left":   np.array([-1.0,  0.0,  0.0]),
    "top":    np.array([ 0.0,  1.0,  0.0]),
    "bottom": np.array([ 0.0, -1.0,  0.0]),
}


def _lookat_T_cam_box(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return 4x4 T_cam_box (OpenCV: X right, Y down, Z forward)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, world_up)) > 0.98:
        world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(world_up, fwd)
    right = right / np.linalg.norm(right)
    down = np.cross(right, fwd)
    down = down / np.linalg.norm(down)

    R = np.stack([right, down, fwd], axis=0)
    t = -R @ eye
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_cameras_ring(
    n: int = 4,
    box_dims_m: tuple[float, float, float] = (0.12, 0.12, 0.06),
    radius_m: float = 0.45,
    height_above_center_m: float = 0.10,
    image_size: tuple[int, int] = (1280, 720),
    fx: float = 1000.0,
    fy: float | None = None,
    angle_offset_deg: float = 45.0,
) -> list[dict]:
    """Place n cameras on a ring around the box. angle_offset=45 puts them at corners."""
    W, D, H = box_dims_m
    box_center = np.array([W / 2.0, H / 2.0, D / 2.0])

    if fy is None:
        fy = fx
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0

    cameras = []
    for i in range(n):
        angle = np.deg2rad(angle_offset_deg + i * 360.0 / n)
        eye = box_center + np.array([
            radius_m * np.cos(angle),
            height_above_center_m,
            radius_m * np.sin(angle),
        ])
        T_cam_box = _lookat_T_cam_box(eye, box_center)
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        intr = CameraIntrinsics(
            camera_id=f"cam_{i}",
            K=K,
            dist=np.zeros(5),
            resolution=(w, h),
            reprojection_error=0.0,
        )
        cameras.append({
            "id": f"cam_{i}",
            "intrinsics": intr,
            "T_cam_box": T_cam_box,
        })
    return cameras


def make_default_scene(
    ball_position_box: np.ndarray | None = None,
    n_cameras: int = 4,
    box_dims_m: tuple[float, float, float] = (0.12, 0.12, 0.06),
    marker_side_m: float = 0.030,
    image_size: tuple[int, int] = (1280, 720),
    fx: float = 1000.0,
    radius_m: float = 0.45,
    height_above_center_m: float = 0.10,
    angle_offset_deg: float = 45.0,
) -> SyntheticScene:
    if ball_position_box is None:
        W, D, H = box_dims_m
        ball_position_box = np.array([W / 2.0, H / 2.0, D / 2.0])

    markers = _build_markers(box_dims_m, marker_side_m)
    cameras = make_cameras_ring(
        n=n_cameras,
        box_dims_m=box_dims_m,
        radius_m=radius_m,
        height_above_center_m=height_above_center_m,
        image_size=image_size,
        fx=fx,
        angle_offset_deg=angle_offset_deg,
    )
    return SyntheticScene(
        box_dims_m=box_dims_m,
        marker_side_m=marker_side_m,
        markers=markers,
        ball_position_box=ball_position_box,
        cameras=cameras,
        image_size=image_size,
    )


def _build_markers(
    box_dims_m: tuple[float, float, float],
    marker_side_m: float,
) -> list[dict]:
    """Build 8 box markers matching config/box.yaml layout, with face normals."""
    W, D, H = box_dims_m
    s = marker_side_m
    h = s / 2.0

    def corners(center, r_vec, u_vec):
        return np.array([
            center - h * r_vec + h * u_vec,
            center + h * r_vec + h * u_vec,
            center + h * r_vec - h * u_vec,
            center - h * r_vec - h * u_vec,
        ], dtype=np.float64)

    markers = [
        {"id": 0, "face": "front",
         "face_normal": _FACE_NORMALS["front"].copy(),
         "corners_box_frame_m": corners(
             np.array([W/2, H/2, 0.0]),
             np.array([1,0,0], float), np.array([0,1,0], float))},
        {"id": 1, "face": "right",
         "face_normal": _FACE_NORMALS["right"].copy(),
         "corners_box_frame_m": corners(
             np.array([W, H/2, D/2]),
             np.array([0,0,-1], float), np.array([0,1,0], float))},
        {"id": 2, "face": "back",
         "face_normal": _FACE_NORMALS["back"].copy(),
         "corners_box_frame_m": corners(
             np.array([W/2, H/2, D]),
             np.array([-1,0,0], float), np.array([0,1,0], float))},
        {"id": 3, "face": "left",
         "face_normal": _FACE_NORMALS["left"].copy(),
         "corners_box_frame_m": corners(
             np.array([0.0, H/2, D/2]),
             np.array([0,0,1], float), np.array([0,1,0], float))},
        {"id": 4, "face": "top",
         "face_normal": _FACE_NORMALS["top"].copy(),
         "corners_box_frame_m": corners(
             np.array([W*0.25, H, D*0.25]),
             np.array([1,0,0], float), np.array([0,0,-1], float))},
        {"id": 5, "face": "top",
         "face_normal": _FACE_NORMALS["top"].copy(),
         "corners_box_frame_m": corners(
             np.array([W*0.75, H, D*0.25]),
             np.array([1,0,0], float), np.array([0,0,-1], float))},
        {"id": 6, "face": "top",
         "face_normal": _FACE_NORMALS["top"].copy(),
         "corners_box_frame_m": corners(
             np.array([W*0.25, H, D*0.75]),
             np.array([1,0,0], float), np.array([0,0,-1], float))},
        {"id": 7, "face": "top",
         "face_normal": _FACE_NORMALS["top"].copy(),
         "corners_box_frame_m": corners(
             np.array([W*0.75, H, D*0.75]),
             np.array([1,0,0], float), np.array([0,0,-1], float))},
    ]
    return markers
