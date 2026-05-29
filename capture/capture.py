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
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import load_cameras_config


# ── Camera context manager ────────────────────────────────────────────────────

def _camera_backend() -> int:
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2


def _camera_source(device_index: int) -> int:
    # Use the OpenCV/V4L2 device index directly on Linux. The mapping is
    # resolved from /sys and validated before capture starts.
    return device_index


def _open_capture(device_index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(_camera_source(device_index), _camera_backend())


class _Camera:
    def __init__(self, device_index: int, cam_cfg: dict):
        self._index = device_index
        self._cfg = cam_cfg
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "_Camera":
        self._cap = _open_capture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cfg['id']} (index {self._index})")

        # Must be set before the first read so V4L2 allocates only 1 kernel buffer.
        # Prevents old frames piling up while other cameras are being opened.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Request MJPEG so the camera sends compressed frames over USB instead of
        # raw YUYV. Three 1080p YUYV streams simultaneously = ~375 MB/s, which
        # saturates a shared USB controller. MJPEG cuts bandwidth ~10×.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        w, h = self._cfg["capture_resolution"]
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Give sensor time to power on before reading (first open is cold).
        time.sleep(1.5)

        # Warm-up with auto-exposure so the sensor adapts before we lock settings.
        for _ in range(30):
            self._read_frame(timeout=5.0)

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
                self._read_frame(timeout=5.0)

        return self

    def drain(self, n: int = 10) -> None:
        """Discard n frames to flush the V4L2 buffer."""
        assert self._cap is not None
        for _ in range(n):
            try:
                self._read_frame(timeout=0.25)
            except RuntimeError:
                break

    def _read_frame(self, timeout: float = 2.0) -> np.ndarray:
        assert self._cap is not None
        deadline = time.monotonic() + timeout
        last_error = ""
        while True:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
            last_error = f"Camera {self._cfg['id']} (device {self._index})"
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        raise RuntimeError(f"{last_error}: frame read failed")

    def read(self) -> np.ndarray:
        return self._read_frame(timeout=2.0)

    def __exit__(self, *_) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ── Session capture ───────────────────────────────────────────────────────────

def _probe_camera_indices(max_index: int = 16) -> list[int]:
    """Return camera indices that OpenCV can actually open on this machine."""
    found: list[int] = []
    for idx in range(max_index):
        cap = _open_capture(idx)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(idx)
        finally:
            cap.release()
    return found


def _resolve_device_indices(
    cameras: list[dict],
    serial_to_index: dict[str, int],
    max_index: int = 16,
) -> list[int]:
    """Resolve each configured camera to a unique OpenCV device index.

    Preference order:
      1. Explicit serial_to_index mapping in cameras.yaml
      2. Optional per-camera device_index field
      3. Remaining detected cameras in config order

    The fallback makes fresh installs usable even before serials are filled in,
    but it still prints the resolved mapping so users can pin it down later.
    """
    available = _probe_camera_indices(max_index=max_index)
    remaining = list(available)
    assigned: list[int] = []
    lines: list[str] = []
    used_auto = False

    for cam_cfg in cameras:
        cam_id = cam_cfg["id"]
        serial = cam_cfg.get("serial", "")
        idx: int | None = None
        source = ""

        if serial and serial in serial_to_index:
            idx = int(serial_to_index[serial])
            source = "serial"
        elif cam_cfg.get("device_index") is not None:
            idx = int(cam_cfg["device_index"])
            source = "device_index"
        elif remaining:
            idx = remaining.pop(0)
            source = "auto"
            used_auto = True
        else:
            raise RuntimeError(
                f"No available camera index left for {cam_id}. "
                f"Detected cameras: {available}"
            )

        if idx in assigned:
            raise RuntimeError(
                f"Duplicate device index {idx} assigned to {cam_id}. "
                f"Resolved indices must be unique."
            )

        assigned.append(idx)
        if idx in remaining:
            remaining.remove(idx)

        if source == "auto":
            lines.append(f"  {cam_id}: device {idx} (auto-assigned)")
        elif source == "device_index":
            lines.append(f"  {cam_id}: device {idx} (from device_index)")
        else:
            lines.append(f"  {cam_id}: device {idx} (from serial {serial})")

    print("Resolved camera mapping:")
    for line in lines:
        print(line)
    if used_auto:
        print("WARN: one or more cameras used automatic device assignment.")
        print("      Fill in real serial_to_index values to make mappings stable.")
    return assigned


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
    for cam_cfg in cameras:
        (output_dir / cam_cfg["id"]).mkdir(parents=True, exist_ok=True)

    device_indices = _resolve_device_indices(cameras, serial_to_index)

    # Open cameras sequentially to avoid simultaneous USB warmup contention.
    cam_objects: list[_Camera] = []
    try:
        for cam_cfg, idx in zip(cameras, device_indices):
            print(f"Opening {cam_cfg['id']} (device {idx}) …")
            cam = _Camera(idx, cam_cfg)
            cam.__enter__()
            cam_objects.append(cam)

        # Flush stale frames that accumulated in each camera's buffer while the
        # other cameras were warming up.
        print("Flushing buffers …")
        for cam in cam_objects:
            cam.drain(15)

        # Round-robin: each iteration captures one frame from each camera in order.
        frame_counts = [0] * len(cameras)
        t_start = time.monotonic()
        for i in range(n_frames):
            for j, (cam, cam_cfg) in enumerate(zip(cam_objects, cameras)):
                frame = cam.read()
                frame_path = output_dir / cam_cfg["id"] / f"frame_{i:04d}.png"
                cv2.imwrite(str(frame_path), frame)
                frame_counts[j] += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_frames} frames")

    finally:
        for cam in cam_objects:
            cam.__exit__(None, None, None)

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {n_frames} frames × {len(cameras)} cameras in {elapsed:.1f} s")

    session_meta: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_frames_requested": n_frames,
        "cameras": [
            {
                "id": cam_cfg["id"],
                "serial": cam_cfg.get("serial", ""),
                "device_index": device_indices[j],
                "n_frames_captured": frame_counts[j],
                "resolution_requested": cam_cfg["capture_resolution"],
                "exposure": cam_cfg.get("exposure"),
                "focus": cam_cfg.get("focus"),
            }
            for j, cam_cfg in enumerate(cameras)
        ],
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="UTF-8") as f:
        json.dump(session_meta, f, indent=2)
    print(f"Session metadata saved to {meta_path}")
    return output_dir


def _read_resolution(path: Path) -> list[int] | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return [w, h]


# ── Camera listing ────────────────────────────────────────────────────────────

def _v4l2_device_name(idx: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{idx}/name")
    try:
        return name_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def list_cameras(max_index: int = 10) -> None:
    print("Probing camera device indices …")
    found = False
    for idx in range(max_index):
        cap = _open_capture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = _v4l2_device_name(idx)
            name_str = f"  [{name}]" if name else ""
            print(f"  index {idx}: {w}×{h}{name_str}")
            cap.release()
            found = True
        else:
            cap.release()
    if not found:
        print("  No cameras found.")


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
