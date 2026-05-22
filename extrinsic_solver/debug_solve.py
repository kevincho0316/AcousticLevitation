"""
Visual debug for the extrinsic solver.

Renders, per frame, what estimatePoseBoard sees and produces:
  - detected ArUco markers (cyan outlines + IDs)
  - estimated box-frame axes drawn at the box origin
  - every board marker reprojected with the estimated pose (red)
    overlaid against its detected corners (green), connected by lines
  - a text panel: markers found, faces covered, per-marker reproj error,
    and whether the frame would pass the solver's accept/reject gates

Use this when a camera reports "no valid poses estimated" or a suspicious
reprojection error: it shows whether markers are missing, the box.yaml
layout is wrong, or the pose itself is bad.

Usage:
    # one image
    python -m extrinsic_solver.debug_solve \\
        --image sessions/session_001/cam_left/frame_0000.png \\
        --intrinsics calibration/cam_left_intrinsics.yaml \\
        --box-config config/box.yaml \\
        --output debug_extrinsic/

    # one representative frame per camera in a session
    python -m extrinsic_solver.debug_solve \\
        --session sessions/session_001 \\
        --cameras-config config/cameras.yaml \\
        --calibration-dir calibration/ \\
        --box-config config/box.yaml \\
        --output debug_extrinsic/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CameraIntrinsics
from common.io_utils import load_box_config, load_cameras_config, load_intrinsics
from extrinsic_solver.solve import _build_board, _detect_markers

# ── colours (BGR) ─────────────────────────────────────────────────────────────
GREEN = (0, 220, 0)      # detected corners
RED = (0, 0, 255)        # reprojected corners
CYAN = (255, 220, 0)
YELLOW = (0, 220, 220)
GREY = (160, 160, 160)


def _board_ids(board) -> np.ndarray:
    return board.getIds() if hasattr(board, "getIds") else board.ids


def _project_marker(obj_p: np.ndarray, rvec, tvec, K, dist) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj_p.astype(np.float32), rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def _put_panel(img: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """Draw a semi-transparent text panel in the top-left corner."""
    pad, lh = 10, 24
    w = 460
    h = pad * 2 + lh * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (text, colour) in enumerate(lines):
        y = pad + lh * (i + 1) - 6
        cv2.putText(img, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 1, cv2.LINE_AA)


def render_debug(
    image_path: Path,
    intrinsics: CameraIntrinsics,
    box_cfg: dict,
    min_markers: int = 3,
    max_reproj_px: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Return (debug_image, stats) for a single frame."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    K = intrinsics.K.astype(np.float32)
    dist = intrinsics.dist.astype(np.float32)
    zero_dist = np.zeros_like(dist)

    # Same pipeline as the solver: undistort, then detect on the gray image.
    img_ud = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)

    board, aruco_dict = _build_board(box_cfg)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()

    corners, ids = _detect_markers(gray, aruco_dict, params)
    vis = img_ud.copy()

    stats: dict = {"n_markers": 0, "faces": [], "pose_ok": False,
                   "mean_reproj": float("inf"), "accepted": False}

    if ids is None or len(ids) == 0:
        _put_panel(vis, [("NO MARKERS DETECTED", RED),
                         ("check dictionary / lighting / box visibility", GREY)])
        return vis, stats

    detected_ids = set(int(i) for i in ids.ravel())
    id_to_face = {m["id"]: m.get("face", "?") for m in box_cfg["markers"]}
    faces = sorted({id_to_face.get(i, "?") for i in detected_ids})
    stats["n_markers"] = len(detected_ids)
    stats["faces"] = faces

    # Detected markers (cyan outline + id).
    cv2.aruco.drawDetectedMarkers(vis, corners, ids, CYAN)

    # Estimate the board pose.
    n_valid, rvec, tvec = cv2.aruco.estimatePoseBoard(
        corners, ids, board, K, zero_dist, None, None
    )
    stats["pose_ok"] = n_valid > 0

    per_marker_err: list[tuple[int, float]] = []
    if n_valid > 0:
        # Box-frame axes at the origin — length = half the smallest box side.
        dims = box_cfg.get("box_dimensions")
        if isinstance(dims, dict):
            sides = [dims["width_mm"], dims["depth_mm"], dims["height_mm"]]
            axis_len = min(float(s) for s in sides) / 1000.0 / 2.0
        elif dims:
            axis_len = min(float(s) for s in dims) / 1000.0 / 2.0
        else:
            axis_len = box_cfg.get("marker_side_m", 0.025) * 2.0
        cv2.drawFrameAxes(vis, K, zero_dist, rvec, tvec, axis_len, 2)

        obj_pts = board.getObjPoints()
        bids = _board_ids(board).ravel()
        det_map = {int(i): c.reshape(4, 2)
                   for i, c in zip(ids.ravel(), corners)}

        for slot, marker_id in enumerate(bids):
            obj_p = obj_pts[slot].astype(np.float32)
            proj = _project_marker(obj_p, rvec, tvec, K, zero_dist)
            # Reprojected marker outline (red).
            cv2.polylines(vis, [proj.astype(np.int32)], True, RED, 1, cv2.LINE_AA)
            obs = det_map.get(int(marker_id))
            if obs is None:
                continue  # board marker not visible in this frame
            err = float(np.mean(np.linalg.norm(obs - proj, axis=1)))
            per_marker_err.append((int(marker_id), err))
            for (ox, oy), (px, py) in zip(obs, proj):
                cv2.circle(vis, (int(ox), int(oy)), 4, GREEN, -1, cv2.LINE_AA)
                cv2.circle(vis, (int(px), int(py)), 3, RED, -1, cv2.LINE_AA)
                cv2.line(vis, (int(ox), int(oy)), (int(px), int(py)),
                         YELLOW, 1, cv2.LINE_AA)

    mean_reproj = (float(np.mean([e for _, e in per_marker_err]))
                   if per_marker_err else float("inf"))
    stats["mean_reproj"] = mean_reproj

    # Replicate the solver's accept/reject gates.
    gate_markers = len(detected_ids) >= min_markers
    gate_faces = len(faces) >= 2
    gate_pose = n_valid > 0
    gate_reproj = mean_reproj <= max_reproj_px
    accepted = gate_markers and gate_faces and gate_pose and gate_reproj
    stats["accepted"] = accepted

    def ok(b: bool) -> tuple:
        return GREEN if b else RED

    lines = [
        (f"{intrinsics.camera_id}   {image_path.name}", YELLOW),
        (f"markers: {len(detected_ids)} (need >={min_markers})", ok(gate_markers)),
        (f"faces:   {','.join(faces)} (need >=2)", ok(gate_faces)),
        (f"pose:    {'solved' if gate_pose else 'FAILED'}", ok(gate_pose)),
        (f"reproj:  {mean_reproj:.3f} px (max {max_reproj_px})", ok(gate_reproj)),
        ("FRAME ACCEPTED" if accepted else "FRAME REJECTED", ok(accepted)),
    ]
    for mid, err in sorted(per_marker_err):
        lines.append((f"  id {mid:>3}: {err:.3f} px",
                      GREEN if err <= max_reproj_px else RED))
    _put_panel(vis, lines)

    # Legend.
    h = vis.shape[0]
    cv2.circle(vis, (20, h - 40), 5, GREEN, -1)
    cv2.putText(vis, "detected", (32, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, GREEN, 1, cv2.LINE_AA)
    cv2.circle(vis, (140, h - 40), 5, RED, -1)
    cv2.putText(vis, "reprojected", (152, h - 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, RED, 1, cv2.LINE_AA)
    return vis, stats


def _first_frame(cam_dir: Path) -> Path | None:
    # Prefer capture frames; fall back to any png. Skip background.png.
    frames = sorted(cam_dir.glob("frame_*.png"))
    if not frames:
        frames = [f for f in sorted(cam_dir.glob("*.png"))
                  if f.stem != "background"]
    return frames[0] if frames else None


def main() -> None:
    p = argparse.ArgumentParser(description="Visual debug for the extrinsic solver")
    p.add_argument("--image", type=Path, help="single frame to debug")
    p.add_argument("--intrinsics", type=Path, help="intrinsics YAML (with --image)")
    p.add_argument("--session", type=Path, help="session dir (per-camera mode)")
    p.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    p.add_argument("--box-config", type=Path, default=Path("config/box.yaml"))
    p.add_argument("--output", type=Path, default=Path("debug_extrinsic"))
    p.add_argument("--min-markers", type=int, default=3)
    p.add_argument("--max-reproj-px", type=float, default=2.0)
    args = p.parse_args()

    box_cfg = load_box_config(args.box_config)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, CameraIntrinsics]] = []

    if args.image:
        if not args.intrinsics:
            sys.exit("--intrinsics is required with --image")
        jobs.append((args.image, load_intrinsics(args.intrinsics)))
    elif args.session:
        cam_cfg = load_cameras_config(args.cameras_config)
        for cam in cam_cfg["cameras"]:
            cam_id = cam["id"]
            intr_path = args.calibration_dir / Path(cam["intrinsics_file"]).name
            cam_dir = args.session / cam_id
            if not intr_path.exists():
                print(f"WARN: intrinsics missing for {cam_id}. Skipping.")
                continue
            frame = _first_frame(cam_dir)
            if frame is None:
                print(f"WARN: no frames for {cam_id}. Skipping.")
                continue
            jobs.append((frame, load_intrinsics(intr_path)))
    else:
        sys.exit("provide either --image or --session")

    for frame_path, intrinsics in jobs:
        try:
            vis, stats = render_debug(
                frame_path, intrinsics, box_cfg,
                min_markers=args.min_markers, max_reproj_px=args.max_reproj_px,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            continue
        out = args.output / f"debug_{intrinsics.camera_id}_{frame_path.stem}.png"
        cv2.imwrite(str(out), vis)
        flag = "ACCEPT" if stats["accepted"] else "REJECT"
        print(f"  {intrinsics.camera_id}: {stats['n_markers']} markers, "
              f"reproj {stats['mean_reproj']:.3f} px  [{flag}]  -> {out}")


if __name__ == "__main__":
    main()
