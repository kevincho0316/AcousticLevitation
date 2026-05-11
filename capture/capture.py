"""
Multi-camera frame capture.

Usage:
    python -m capture.capture \\
        --config config/cameras.yaml \\
        --output sessions/session_001 \\
        [--n-frames 200]

    python -m capture.capture --list-cameras   # discover device indices

Each camera's frames are saved to:
    sessions/session_001/<camera_id>/frame_NNNN.png

A metadata.json is written alongside with timestamps, camera settings,
and the cameras config snapshot.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import load_cameras_config


# ── Camera context manager ────────────────────────────────────────────────────

class _Camera:
    def __init__(self, device_index: int, cam_cfg: dict):
        self._index = device_index
        self._cfg = cam_cfg
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "_Camera":
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cfg['id']} (index {self._index})")

        w, h = self._cfg["capture_resolution"]
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Give sensor time to power on before reading (first open is cold).
        time.sleep(1.5)

        # Warm-up with auto-exposure so the sensor adapts before we lock settings.
        for _ in range(30):
            self._cap.read()

        # Disable autofocus.
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if self._cfg.get("focus") is not None:
            self._cap.set(cv2.CAP_PROP_FOCUS, self._cfg["focus"])

        # Lock manual exposure only if explicitly configured.
        # CAP_PROP_AUTO_EXPOSURE: 1 = manual on most UVC/DSHOW; 3 = auto.
        if self._cfg.get("exposure") is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self._cfg["exposure"])
            # Let sensor settle after exposure change.
            for _ in range(20):
                self._cap.read()

        return self

    def read(self) -> np.ndarray:
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Camera {self._cfg['id']}: frame read failed")
        return frame

    def __exit__(self, *_) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ── Session capture ───────────────────────────────────────────────────────────

def capture_session(
    cameras_config_path: str | Path,
    output_dir: str | Path,
    n_frames: int | None = None,
) -> Path:
    cfg = load_cameras_config(cameras_config_path)
    cameras = cfg["cameras"]
    serial_to_index: dict[str, int] = cfg.get("serial_to_index", {})
    default_n = cfg.get("frames_per_camera", 200)
    n_frames = n_frames if n_frames is not None else default_n

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_meta: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_frames_requested": n_frames,
        "cameras": [],
    }

    cam_metas: list[dict | None] = [None] * len(cameras)
    lock = threading.Lock()

    def _capture_one(cam_cfg: dict, slot: int) -> None:
        cam_id = cam_cfg["id"]
        serial = cam_cfg.get("serial", "")
        device_index = serial_to_index.get(serial, None)
        if device_index is None:
            print(f"WARN: no device index for camera {cam_id} (serial {serial}). Skipping.")
            return

        cam_dir = output_dir / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCapturing {n_frames} frames from {cam_id} (device {device_index}) …")
        t_start = time.monotonic()
        frame_paths: list[str] = []

        try:
            with _Camera(device_index, cam_cfg) as cam:
                for i in range(n_frames):
                    frame = cam.read()
                    frame_path = cam_dir / f"frame_{i:04d}.png"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path.name)
                    if (i + 1) % 50 == 0:
                        print(f"  {cam_id}: {i+1}/{n_frames} frames")
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return

        elapsed = time.monotonic() - t_start
        print(f"  Done [{cam_id}]: {len(frame_paths)} frames in {elapsed:.1f} s ({len(frame_paths)/elapsed:.1f} fps)")

        actual_res = _read_resolution(cam_dir / frame_paths[0]) if frame_paths else None
        meta = {
            "id": cam_id,
            "serial": serial,
            "device_index": device_index,
            "n_frames_captured": len(frame_paths),
            "resolution_actual": actual_res,
            "resolution_requested": cam_cfg["capture_resolution"],
            "exposure": cam_cfg["exposure"],
            "focus": cam_cfg.get("focus"),
            "elapsed_s": round(elapsed, 2),
        }
        with lock:
            cam_metas[slot] = meta

    threads = [
        threading.Thread(target=_capture_one, args=(cam_cfg, i), daemon=True)
        for i, cam_cfg in enumerate(cameras)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    session_meta["cameras"] = [m for m in cam_metas if m is not None]

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(session_meta, f, indent=2)
    print(f"\nSession metadata saved to {meta_path}")
    return output_dir


def _read_resolution(path: Path) -> list[int] | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return [w, h]


# ── Camera listing ────────────────────────────────────────────────────────────

def list_cameras(max_index: int = 10) -> None:
    print("Probing camera device indices …")
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index {idx}: {w}×{h}")
            cap.release()
        else:
            cap.release()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-camera capture")
    p.add_argument("--config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--output", type=Path, default=Path("sessions/session_001"))
    p.add_argument("--n-frames", type=int, default=None,
                   help="Override frames_per_camera from config")
    p.add_argument("--list-cameras", action="store_true",
                   help="Print available camera device indices and exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_cameras:
        list_cameras()
        return
    capture_session(args.config, args.output, args.n_frames)


if __name__ == "__main__":
    main()
