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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
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


# ── Marker corner auto-computation ───────────────────────────────────────────
#
# Box coordinate frame:
#   Origin: front-bottom-left corner
#   X: left → right (width),  Y: bottom → top (height),  Z: front → back (depth)
#
# For each face the local "right" and "up" vectors define how corners are laid out
# when the face is viewed from outside (camera side).  ArUco corner order:
#   0=top-left  1=top-right  2=bottom-right  3=bottom-left  (clockwise from top-left)
#
# "right" and "up" chosen so that walking clockwise around the box, the marker
# always reads naturally from the outside.

_FACE_AXES: dict[str, dict] = {
    # face     right-vec       up-vec         face-center-fn(W,D,H)
    "front":  {"r": np.array([ 1, 0,  0]), "u": np.array([0, 1, 0]),
               "c": lambda W, D, H: np.array([W/2,  H/2, 0.0])},
    "back":   {"r": np.array([-1, 0,  0]), "u": np.array([0, 1, 0]),
               "c": lambda W, D, H: np.array([W/2,  H/2,   D])},
    "right":  {"r": np.array([ 0, 0, -1]), "u": np.array([0, 1, 0]),
               "c": lambda W, D, H: np.array([  W,  H/2, D/2])},
    "left":   {"r": np.array([ 0, 0,  1]), "u": np.array([0, 1, 0]),
               "c": lambda W, D, H: np.array([0.0,  H/2, D/2])},
    # Top face: right=+X, up=-Z (toward front) when viewed from above looking down.
    "top":    {"r": np.array([ 1, 0,  0]), "u": np.array([0, 0, -1]),
               "c": lambda W, D, H: np.array([W/2,    H, D/2])},
    "bottom": {"r": np.array([ 1, 0,  0]), "u": np.array([0, 0,  1]),
               "c": lambda W, D, H: np.array([W/2,  0.0, D/2])},
}


def _marker_corners_m(marker: dict, W: float, D: float, H: float, s: float) -> np.ndarray:
    """Return (4,3) corner positions in box frame (meters).

    Two ways to place a marker:
      a) Explicit: ``corners_box_frame`` list of 4 × [x,y,z] in mm → just convert.
      b) Auto: ``face`` + optional ``center_box_mm: [x,y,z]``
               If center_box_mm is omitted the marker is centred on its face.
    """
    # Explicit corners take priority — backward-compatible with old configs.
    if "corners_box_frame" in marker:
        return np.array(marker["corners_box_frame"], dtype=np.float64) / 1000.0

    face = marker["face"]
    axes = _FACE_AXES[face]
    r_vec = axes["r"].astype(np.float64)
    u_vec = axes["u"].astype(np.float64)

    if "center_box_mm" in marker:
        center = np.array(marker["center_box_mm"], dtype=np.float64) / 1000.0
    else:
        center = axes["c"](W, D, H)  # already in meters (W,D,H are meters here)

    h = s / 2.0
    return np.array([
        center - h * r_vec + h * u_vec,   # corner 0: top-left
        center + h * r_vec + h * u_vec,   # corner 1: top-right
        center + h * r_vec - h * u_vec,   # corner 2: bottom-right
        center - h * r_vec - h * u_vec,   # corner 3: bottom-left
    ], dtype=np.float64)


# ── Box configuration ─────────────────────────────────────────────────────────

def load_box_config(path: str | Path) -> dict:
    """Load box.yaml and resolve marker corners (auto or explicit) → meters."""
    cfg = load_yaml(path)

    dims = cfg["box_dimensions"]
    W = float(dims["width_mm"])  / 1000.0
    D = float(dims["depth_mm"])  / 1000.0
    H = float(dims["height_mm"]) / 1000.0

    s = float(cfg["marker_side_mm"]) / 1000.0

    for marker in cfg["markers"]:
        marker["corners_box_frame_m"] = _marker_corners_m(marker, W, D, H, s)

    cfg["marker_side_m"] = s
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
