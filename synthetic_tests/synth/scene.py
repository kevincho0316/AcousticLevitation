"""Ground-truth scene description for synthetic tests."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common import CameraIntrinsics, CameraPose


@dataclass
class SyntheticScene:
    box_dims_m: tuple[float, float, float]   # (W, D, H)
    marker_side_m: float
    markers: list[dict]                      # each: {id, face, corners_box_frame_m (4,3)}
    ball_position_box: np.ndarray            # (3,) meters, box frame
    cameras: list[dict]                      # each: {id, intrinsics, T_cam_box (4,4)}
    image_size: tuple[int, int] = (1280, 720)
    ball_radius_m: float = 0.0065
    aruco_dict_name: str = "DICT_4X4_50"

    def get_intrinsics(self, cam_id: str) -> CameraIntrinsics:
        for c in self.cameras:
            if c["id"] == cam_id:
                return c["intrinsics"]
        raise KeyError(cam_id)

    def get_T_cam_box(self, cam_id: str) -> np.ndarray:
        for c in self.cameras:
            if c["id"] == cam_id:
                return c["T_cam_box"]
        raise KeyError(cam_id)

    def to_camera_pose(self, cam_id: str) -> CameraPose:
        return CameraPose(
            camera_id=cam_id,
            T_cam_box=self.get_T_cam_box(cam_id),
            reprojection_error=0.0,
            n_markers_used=len(self.markers),
            n_frames_used=1,
        )

    def to_box_cfg(self) -> dict:
        """Return a box_cfg dict compatible with the pipeline's format."""
        import cv2
        W, D, H = self.box_dims_m
        cfg: dict[str, Any] = {
            "box_dimensions": {
                "width_mm": W * 1000,
                "depth_mm": D * 1000,
                "height_mm": H * 1000,
            },
            "marker_side_mm": self.marker_side_m * 1000,
            "marker_side_m": self.marker_side_m,
            "marker_position_uncertainty_m": 5e-4,
            "aruco_dictionary": self.aruco_dict_name,
            "markers": [
                {
                    "id": m["id"],
                    "face": m["face"],
                    "corners_box_frame_m": m["corners_box_frame_m"],
                }
                for m in self.markers
            ],
        }
        return cfg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _cvt(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, CameraIntrinsics):
                return {
                    "camera_id": obj.camera_id,
                    "K": obj.K.tolist(),
                    "dist": obj.dist.tolist(),
                    "resolution": list(obj.resolution),
                    "reprojection_error": obj.reprojection_error,
                }
            if isinstance(obj, dict):
                return {k: _cvt(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cvt(v) for v in obj]
            return obj

        data = {
            "box_dims_m": list(self.box_dims_m),
            "marker_side_m": self.marker_side_m,
            "markers": _cvt(self.markers),
            "ball_position_box": self.ball_position_box.tolist(),
            "cameras": _cvt(self.cameras),
            "image_size": list(self.image_size),
            "ball_radius_m": self.ball_radius_m,
            "aruco_dict_name": self.aruco_dict_name,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
