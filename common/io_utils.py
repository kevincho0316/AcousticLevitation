"""YAML / JSON I/O helpers for calibration data and pipeline results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common import CameraIntrinsics


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _np_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
    return dumper.represent_data(data.tolist())

yaml.add_representer(np.ndarray, _np_representer)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ── Camera intrinsics ─────────────────────────────────────────────────────────

def save_intrinsics(intr: CameraIntrinsics, path: str | Path) -> None:
    data = {
        "camera_id": intr.camera_id,
        "resolution": list(intr.resolution),
        "camera_matrix": intr.K.tolist(),
        "distortion_coefficients": intr.dist.tolist(),
        "reprojection_error_px": float(intr.reprojection_error),
    }
    save_yaml(data, path)


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    data = load_yaml(path)
    return CameraIntrinsics(
        camera_id=data["camera_id"],
        K=np.array(data["camera_matrix"], dtype=np.float64),
        dist=np.array(data["distortion_coefficients"], dtype=np.float64),
        resolution=tuple(data["resolution"]),
        reprojection_error=float(data["reprojection_error_px"]),
    )


# ── Box configuration ─────────────────────────────────────────────────────────

def load_box_config(path: str | Path) -> dict:
    """Load box.yaml and convert marker corners from mm to meters."""
    cfg = load_yaml(path)
    for marker in cfg["markers"]:
        corners_mm = np.array(marker["corners_box_frame"], dtype=np.float64)
        marker["corners_box_frame_m"] = corners_mm / 1000.0
    cfg["marker_side_m"] = cfg["marker_side_mm"] / 1000.0
    cfg["marker_position_uncertainty_m"] = cfg.get("marker_position_uncertainty_mm", 0.5) / 1000.0
    return cfg


def load_cameras_config(path: str | Path) -> dict:
    return load_yaml(path)


# ── Box→sim transform ─────────────────────────────────────────────────────────

def load_box_to_sim_transform(box_cfg: dict) -> np.ndarray:
    """Return 4×4 SE(3) matrix transforming box-frame meters → sim-frame meters."""
    b2s = box_cfg.get("box_to_sim", {})
    t = np.array(b2s.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    R = np.array(b2s.get("rotation_matrix", np.eye(3).tolist()), dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ── Numpy ↔ JSON ──────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)
