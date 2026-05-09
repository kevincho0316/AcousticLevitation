"""
White ball center detection with sub-pixel precision.

Pipeline per frame:
  1. Undistort.
  2. Convert to grayscale, apply Gaussian blur.
  3. Otsu threshold (high-contrast setup: white ball, black background).
  4. Find connected components; keep the largest blob whose area is within
     the plausible range for a ball at camera distance.
  5. Fit a circle to Canny edge pixels of the blob region by linear least squares.
  6. The circle center is the sub-pixel 2D ball position.
  7. Reject frames where: blob is at the frame edge, fit residual is too large,
     or blob area is implausible.

Per-camera output:
  - Mean center over accepted frames: (u̅, v̅)
  - Covariance of the mean: Σ / N  (sample covariance divided by N)
  - Per-frame centers for diagnostics

Usage:
    python -m ball_detector.detect \\
        --session sessions/session_001 \\
        --cameras-config config/cameras.yaml \\
        --calibration-dir calibration/ \\
        --output sessions/session_001/ball_detections.json \\
        [--min-area 50] [--max-area 50000] [--max-fit-residual 2.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BallDetection2D
from common.io_utils import load_cameras_config, load_intrinsics, save_json


# ── Circle fit ────────────────────────────────────────────────────────────────

def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a circle to (x, y) edge pixels by linear least squares.

    Linearization of (x-cx)^2 + (y-cy)^2 = r^2:
        2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2) = x^2 + y^2
    """
    if len(x) < 6:
        return None
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = result
    r_sq = c + cx ** 2 + cy ** 2
    if r_sq <= 0.0:
        return None
    return float(cx), float(cy), float(np.sqrt(r_sq))


def _circle_residual(x: np.ndarray, y: np.ndarray, cx: float, cy: float, r: float) -> float:
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(np.mean(np.abs(distances - r)))


# ── Interactive blob selection UI ─────────────────────────────────────────────

_BLOB_COLORS = [
    (0, 255, 255), (255, 128, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 128), (128, 255, 0), (255, 255, 0), (0, 200, 255), (255, 80, 80),
]


def select_blob_ui(
    frame: np.ndarray,
    min_area: int = 50,
    max_area: int = 50_000,
    title: str = "Select ball blob",
) -> tuple[int, int] | None:
    """Detect all blobs, label them 1…N, let user pick one.

    Controls:
        Left-click    — select nearest blob
        1–9           — select blob by number (sorted largest→smallest)
        Enter/Space   — confirm
        ESC           — cancel (returns None)
        +/-           — zoom in/out

    Returns integer (x, y) centroid of selected blob in original frame, or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    blobs: list[dict] = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            blobs.append({
                "label": lbl,
                "area": area,
                "cx": int(centroids[lbl, 0]),
                "cy": int(centroids[lbl, 1]),
            })
    blobs.sort(key=lambda b: b["area"], reverse=True)   # #1 = largest

    state: dict = {"selected": None, "zoom": 1.0}

    def _make_overlay() -> np.ndarray:
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for i, blob in enumerate(blobs):
            color = _BLOB_COLORS[i % len(_BLOB_COLORS)]
            is_sel = (state["selected"] == i)
            outline_color = (0, 255, 0) if is_sel else color
            thickness = 3 if is_sel else 2

            mask = (labels == blob["label"]).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, outline_color, thickness)
            cv2.circle(vis, (blob["cx"], blob["cy"]), 18, outline_color, thickness)
            cv2.putText(vis, str(i + 1),
                        (blob["cx"] + 10, blob["cy"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, outline_color, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{blob['area']}px",
                        (blob["cx"] + 10, blob["cy"] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, outline_color, 1, cv2.LINE_AA)

        if not blobs:
            cv2.putText(vis, "No blobs found — check threshold/area limits",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if state["selected"] is not None:
            hint = f"Blob {state['selected'] + 1} selected  |  Enter=confirm  ESC=cancel  +/- zoom"
        else:
            hint = f"{len(blobs)} blobs found  |  Click or press 1-9 to select  |  ESC=cancel"
        cv2.putText(vis, hint, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return vis

    def _redraw() -> None:
        base = _make_overlay()
        h, w = base.shape[:2]
        disp = cv2.resize(base, (int(w * state["zoom"]), int(h * state["zoom"])),
                          interpolation=cv2.INTER_LINEAR)
        cv2.imshow(title, disp)

    def _mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and blobs:
            ox = int(x / state["zoom"])
            oy = int(y / state["zoom"])
            dists = [(ox - b["cx"]) ** 2 + (oy - b["cy"]) ** 2 for b in blobs]
            state["selected"] = int(np.argmin(dists))
            _redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, _mouse)
    _redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):                        # Enter / Space — confirm
            if state["selected"] is not None:
                break
        elif key == 27:                            # ESC — cancel
            state["selected"] = None
            break
        elif ord("1") <= key <= ord("9"):          # number key
            idx = key - ord("1")
            if idx < len(blobs):
                state["selected"] = idx
                _redraw()
        elif key in (ord("+"), ord("=")):
            state["zoom"] = min(state["zoom"] * 1.25, 8.0)
            _redraw()
        elif key == ord("-"):
            state["zoom"] = max(state["zoom"] / 1.25, 0.25)
            _redraw()

    cv2.destroyWindow(title)
    if state["selected"] is None:
        return None
    b = blobs[state["selected"]]
    return b["cx"], b["cy"]


# ── Single-frame detection ────────────────────────────────────────────────────

def detect_ball_frame(
    frame: np.ndarray,
    intrinsics,
    min_area: int = 50,
    max_area: int = 50_000,
    max_fit_residual: float = 3.0,
    edge_margin: int = 10,
    roi_center: tuple[int, int] | None = None,
    roi_radius: int = 60,
) -> tuple[float, float] | None:
    """Detect ball center in a single already-undistorted frame.

    When roi_center is provided the automatic blob search is skipped; detection
    runs only inside a (2*roi_radius) x (2*roi_radius) window around that point.

    Returns (u, v) sub-pixel center or None if detection fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    h, w = gray.shape

    if roi_center is not None:
        # ── ROI-guided path (user selected) ──────────────────────────────────
        cx_seed, cy_seed = roi_center
        rx0 = max(cx_seed - roi_radius, 0)
        ry0 = max(cy_seed - roi_radius, 0)
        rx1 = min(cx_seed + roi_radius, w)
        ry1 = min(cy_seed + roi_radius, h)

        roi_gray = cv2.GaussianBlur(gray[ry0:ry1, rx0:rx1], (5, 5), 0)
        edges = cv2.Canny(roi_gray, 30, 80)
        ey, ex = np.where(edges > 0)

        if len(ex) >= 6:
            ex_g = ex.astype(np.float64) + rx0
            ey_g = ey.astype(np.float64) + ry0
            fit = _fit_circle(ex_g, ey_g)
            if fit is not None:
                cx, cy, r = fit
                if r >= 1.0 and _circle_residual(ex_g, ey_g, cx, cy, r) <= max_fit_residual:
                    return float(cx), float(cy)

        # Fall back to intensity-weighted centroid inside ROI.
        _, thresh_roi = cv2.threshold(roi_gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ys, xs = np.where(thresh_roi > 0)
        if len(xs) == 0:
            return None
        weights = roi_gray[ys, xs].astype(np.float64)
        total = weights.sum()
        if total == 0:
            return None
        return float((xs * weights).sum() / total) + rx0, \
               float((ys * weights).sum() / total) + ry0

    # ── Automatic blob-search path ────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if n_labels <= 1:
        return None

    best_label = -1
    best_area = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area and area > best_area:
            best_area = area
            best_label = label

    if best_label < 0:
        return None

    x0 = int(stats[best_label, cv2.CC_STAT_LEFT])
    y0 = int(stats[best_label, cv2.CC_STAT_TOP])
    bw = int(stats[best_label, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    if (x0 <= edge_margin or y0 <= edge_margin or
            x0 + bw >= w - edge_margin or y0 + bh >= h - edge_margin):
        return None

    pad = 5
    rx0 = max(x0 - pad, 0)
    ry0 = max(y0 - pad, 0)
    rx1 = min(x0 + bw + pad, w)
    ry1 = min(y0 + bh + pad, h)
    roi_gray = gray[ry0:ry1, rx0:rx1]
    edges = cv2.Canny(roi_gray, 30, 80)
    ey, ex = np.where(edges > 0)
    if len(ex) < 6:
        blob_mask = (labels == best_label).astype(np.uint8)
        ys, xs = np.where(blob_mask > 0)
        weights = gray[ys, xs].astype(np.float64)
        total = weights.sum()
        if total == 0:
            return None
        u = float((xs * weights).sum() / total)
        v = float((ys * weights).sum() / total)
        return u, v

    ex_global = ex.astype(np.float64) + rx0
    ey_global = ey.astype(np.float64) + ry0

    fit = _fit_circle(ex_global, ey_global)
    if fit is None:
        return None
    cx, cy, r = fit

    if r < 1.0:
        return None

    residual = _circle_residual(ex_global, ey_global, cx, cy, r)
    if residual > max_fit_residual:
        return None

    return float(cx), float(cy)


# ── Per-camera temporal averaging ─────────────────────────────────────────────

def detect_ball_camera(
    frame_paths: list[Path],
    intrinsics,
    min_area: int = 50,
    max_area: int = 50_000,
    max_fit_residual: float = 3.0,
    edge_margin: int = 10,
    interactive: bool = False,
    roi_radius: int = 60,
) -> BallDetection2D:
    """Detect and temporally average the ball center across all frames.

    When interactive=True the first valid frame is shown and the user clicks the
    ball; every subsequent frame is then searched only within roi_radius pixels
    of that seed point instead of running the automatic blob search.
    """
    K = intrinsics.K.astype(np.float32)
    dist = intrinsics.dist.astype(np.float32)

    roi_center: tuple[int, int] | None = None

    if interactive:
        for path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img_ud = cv2.undistort(img, K, dist)
            roi_center = select_blob_ui(
                img_ud,
                min_area=min_area,
                max_area=max_area,
                title=f"Select ball blob — {intrinsics.camera_id}",
            )
            if roi_center is None:
                raise RuntimeError(
                    f"Camera {intrinsics.camera_id}: selection cancelled by user."
                )
            print(f"  {intrinsics.camera_id}: user selected blob at ({roi_center[0]}, {roi_center[1]})")
            break

    centers: list[tuple[float, float]] = []
    n_rejected = 0

    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            n_rejected += 1
            continue
        img_ud = cv2.undistort(img, K, dist)
        result = detect_ball_frame(
            img_ud, intrinsics,
            min_area=min_area, max_area=max_area,
            max_fit_residual=max_fit_residual, edge_margin=edge_margin,
            roi_center=roi_center, roi_radius=roi_radius,
        )
        if result is None:
            n_rejected += 1
        else:
            centers.append(result)

    if len(centers) < 2:
        raise RuntimeError(
            f"Camera {intrinsics.camera_id}: only {len(centers)} valid detections "
            f"({n_rejected} rejected). Check lighting and ball visibility."
        )

    pts = np.array(centers, dtype=np.float64)  # (N, 2)
    mean_center = pts.mean(axis=0)
    n = len(pts)
    # Sample covariance of the mean: Σ_sample / N
    if n >= 2:
        cov_sample = np.cov(pts.T, ddof=1)        # (2,2) sample covariance
        cov_mean = cov_sample / n                  # covariance of the mean
    else:
        cov_mean = np.eye(2) * 1e-4                # fallback: assume 0.01 px std

    print(
        f"  {intrinsics.camera_id}: {n} accepted, {n_rejected} rejected, "
        f"mean=({mean_center[0]:.3f}, {mean_center[1]:.3f}), "
        f"std=({np.sqrt(cov_mean[0,0])*np.sqrt(n):.3f}, {np.sqrt(cov_mean[1,1])*np.sqrt(n):.3f}) px"
    )

    return BallDetection2D(
        camera_id=intrinsics.camera_id,
        center=mean_center,
        covariance=cov_mean,
        n_frames_accepted=n,
        n_frames_rejected=n_rejected,
        per_frame_centers=pts,
    )


# ── Session-level detection ───────────────────────────────────────────────────

def detect_session(
    session_dir: Path,
    cameras_config_path: Path,
    calibration_dir: Path,
    output_path: Path,
    min_area: int = 50,
    max_area: int = 50_000,
    max_fit_residual: float = 3.0,
    interactive: bool = False,
    roi_radius: int = 60,
) -> dict[str, BallDetection2D]:
    cam_cfg = load_cameras_config(cameras_config_path)
    detections: dict[str, BallDetection2D] = {}
    results_json: dict = {"detections": {}}

    for cam in cam_cfg["cameras"]:
        cam_id = cam["id"]
        intr_path = calibration_dir / Path(cam["intrinsics_file"]).name
        if not intr_path.exists():
            print(f"WARN: intrinsics not found for {cam_id}. Skipping.")
            continue
        intrinsics = load_intrinsics(intr_path)

        frame_dir = session_dir / cam_id
        if not frame_dir.exists():
            print(f"WARN: frame directory not found for {cam_id}. Skipping.")
            continue
        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        if not frame_paths:
            print(f"WARN: no frames for {cam_id}. Skipping.")
            continue

        print(f"\nDetecting ball in {cam_id} ({len(frame_paths)} frames) …")
        try:
            det = detect_ball_camera(
                frame_paths, intrinsics,
                min_area=min_area, max_area=max_area,
                max_fit_residual=max_fit_residual,
                interactive=interactive, roi_radius=roi_radius,
            )
            detections[cam_id] = det
            results_json["detections"][cam_id] = {
                "center_uv": det.center.tolist(),
                "covariance_uv": det.covariance.tolist(),
                "n_frames_accepted": det.n_frames_accepted,
                "n_frames_rejected": det.n_frames_rejected,
            }
        except RuntimeError as exc:
            print(f"ERROR: {exc}")

    save_json(results_json, output_path)
    print(f"\nDetections saved to {output_path}")
    return detections


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ball center detection")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--max-area", type=int, default=50_000)
    p.add_argument("--max-fit-residual", type=float, default=3.0)
    p.add_argument("--interactive", action="store_true",
                   help="Show each camera's first frame and let the user click the ball.")
    p.add_argument("--roi-radius", type=int, default=60,
                   help="Search window half-size (px) around the user-selected seed point.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (args.session / "ball_detections.json")
    detect_session(
        session_dir=args.session,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=output,
        min_area=args.min_area,
        max_area=args.max_area,
        max_fit_residual=args.max_fit_residual,
        interactive=args.interactive,
        roi_radius=args.roi_radius,
    )


if __name__ == "__main__":
    main()
