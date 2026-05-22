"""
Per-capture camera pose estimation via the ArUco board on the box.

For each camera:
  1. Load all frames for that camera from the session directory.
  2. Undistort each frame using the stored intrinsics.
  3. Detect ArUco markers and run estimatePoseBoard against the box board.
  4. Reject frames with too few markers or high reprojection error.
  5. Average accepted poses using SE(3) Lie algebra averaging.

Returns a CameraPose (4×4 T_cam_box) per camera.

Usage:
    python -m extrinsic_solver.solve \\
        --session sessions/session_001 \\
        --box-config config/box.yaml \\
        --cameras-config config/cameras.yaml \\
        --calibration-dir calibration/ \\
        --output sessions/session_001/extrinsics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CameraIntrinsics, CameraPose
from common.io_utils import (
    load_box_config,
    load_cameras_config,
    load_intrinsics,
    save_json,
)


from common.se3_utils import _hat, _se3_log, _se3_exp, _average_se3


# ── ArUco board ───────────────────────────────────────────────────────────────

def _build_board(box_cfg: dict) -> tuple[cv2.aruco.Board, cv2.aruco.Dictionary]:
    dict_name = box_cfg.get("aruco_dictionary", "DICT_4X4_50")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    obj_points_list: list[np.ndarray] = []
    ids_list: list[int] = []
    for marker in box_cfg["markers"]:
        corners = np.array(marker["corners_box_frame_m"], dtype=np.float32)  # (4,3)
        obj_points_list.append(corners)
        ids_list.append(int(marker["id"]))

    ids_arr = np.array(ids_list, dtype=np.int32)
    try:
        board = cv2.aruco.Board(obj_points_list, aruco_dict, ids_arr)
    except AttributeError:
        board = cv2.aruco.Board_create(obj_points_list, aruco_dict, ids_arr)

    return board, aruco_dict


def _detect_markers(gray: np.ndarray, aruco_dict, params):
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def _reprojection_error(corners: list, ids: np.ndarray, rvec: np.ndarray,
                         tvec: np.ndarray, board: cv2.aruco.Board,
                         K: np.ndarray, dist: np.ndarray) -> float:
    errors: list[float] = []
    for corner_set, marker_id in zip(corners, ids.ravel()):
        obj_pts = board.getObjPoints()
        # Find this marker's object points
        board_ids = board.getIds() if hasattr(board, "getIds") else board.ids
        idx = np.where(board_ids == marker_id)[0]
        if len(idx) == 0:
            continue
        obj_p = obj_pts[int(idx[0])].astype(np.float32)
        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, K, dist)
        obs = corner_set.reshape(4, 2)
        proj = proj.reshape(4, 2)
        errors.append(float(np.mean(np.linalg.norm(obs - proj, axis=1))))
    return float(np.mean(errors)) if errors else float("inf")


# ── Per-camera pose estimation ────────────────────────────────────────────────

def estimate_camera_pose(
    frame_paths: list[Path],
    intrinsics: CameraIntrinsics,
    box_cfg: dict,
    min_markers: int = 3,
    max_reproj_px: float = 2.0,
) -> CameraPose:
    board, aruco_dict = _build_board(box_cfg)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    # Sub-pixel corner refinement: reduces corner noise from ~1 px to ~0.1-0.3 px.
    try:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMinAccuracy = 0.01
    except AttributeError:
        pass

    K = intrinsics.K.astype(np.float32)
    dist = intrinsics.dist.astype(np.float32)

    # Build a marker-id → object-points map from the board config.
    board_obj_pts = {m["id"]: np.array(m["corners_box_frame_m"], dtype=np.float64)
                     for m in box_cfg["markers"]}

    pooled_obj: list[np.ndarray] = []
    pooled_img: list[np.ndarray] = []
    n_accepted = 0
    n_rejected = 0

    # First pass: collect all corner correspondences from accepted frames.
    # An initial per-frame pose is used only for reprojection-error gating.
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img_ud = cv2.undistort(img, K, dist)
        gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)

        corners, ids = _detect_markers(gray, aruco_dict, params)
        if ids is None or len(ids) < min_markers:
            n_rejected += 1
            continue

        detected_ids = set(ids.ravel())
        face_set = {m["face"] for m in box_cfg["markers"] if m["id"] in detected_ids}
        if len(face_set) < 2:
            n_rejected += 1
            continue

        n_valid, rvec_f, tvec_f = cv2.aruco.estimatePoseBoard(
            corners, ids, board, K, np.zeros_like(dist), None, None
        )
        if n_valid == 0:
            n_rejected += 1
            continue

        err = _reprojection_error(corners, ids, rvec_f, tvec_f, board, K, np.zeros_like(dist))
        if err > max_reproj_px:
            n_rejected += 1
            continue

        # Frame accepted — pool its corners.
        for c_arr, mid in zip(corners, ids.ravel()):
            mid = int(mid)
            if mid in board_obj_pts:
                pooled_obj.append(board_obj_pts[mid])          # (4, 3)
                pooled_img.append(c_arr.reshape(4, 2))         # (4, 2)
        n_accepted += 1

    if n_accepted == 0:
        raise RuntimeError(
            f"Camera {intrinsics.camera_id}: no valid poses estimated. "
            f"Rejected all {len(frame_paths)} frames (min_markers={min_markers}, max_reproj={max_reproj_px} px). "
            "Check that the box is visible and the box.yaml marker layout is correct."
        )

    all_obj = np.vstack(pooled_obj).astype(np.float64)   # (4·M·F, 3)
    all_img = np.vstack(pooled_img).astype(np.float64)   # (4·M·F, 2)
    K_f64 = K.astype(np.float64)
    zeros_dist = np.zeros(5, dtype=np.float64)

    # Pooled solve: one optimal pose from all corners across all accepted frames.
    _, rvec, tvec = cv2.solvePnP(
        all_obj, all_img, K_f64, zeros_dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    cv2.solvePnPRefineLM(all_obj, all_img, K_f64, zeros_dist, rvec, tvec)

    # Reprojection error of the pooled solution.
    proj, jac = cv2.projectPoints(all_obj, rvec, tvec, K_f64, zeros_dist)
    residuals = all_img - proj.reshape(-1, 2)
    mean_reproj = float(np.mean(np.linalg.norm(residuals, axis=1)))

    # 6×6 pose covariance: Σ = σ² · (JᵀJ)⁻¹  (columns 0-5 are rvec/tvec Jacobian).
    J_pose = jac[:, :6].astype(np.float64)         # (2N, 6)
    sigma2 = float(np.mean(residuals ** 2))
    JtJ = J_pose.T @ J_pose
    try:
        pose_cov = sigma2 * np.linalg.inv(JtJ)     # (6, 6)
    except np.linalg.LinAlgError:
        pose_cov = sigma2 * np.linalg.pinv(JtJ)

    R, _ = cv2.Rodrigues(rvec)
    T_cam_box = np.eye(4)
    T_cam_box[:3, :3] = R
    T_cam_box[:3, 3] = tvec.ravel()

    print(
        f"  {intrinsics.camera_id}: {n_accepted} frames used, "
        f"{n_rejected} rejected, mean reproj = {mean_reproj:.3f} px"
    )
    return CameraPose(
        camera_id=intrinsics.camera_id,
        T_cam_box=T_cam_box,
        reprojection_error=mean_reproj,
        n_markers_used=min_markers,
        n_frames_used=n_accepted,
        pose_covariance=pose_cov,
    )


# ── Session-level solver ──────────────────────────────────────────────────────

def solve_session(
    session_dir: Path,
    box_config_path: Path,
    cameras_config_path: Path,
    calibration_dir: Path,
    output_path: Path,
    min_markers: int = 3,
    max_reproj_px: float = 2.0,
) -> dict[str, CameraPose]:
    box_cfg = load_box_config(box_config_path)
    cam_cfg = load_cameras_config(cameras_config_path)

    poses: dict[str, CameraPose] = {}
    results_json: dict = {"poses": {}}

    for cam in cam_cfg["cameras"]:
        cam_id = cam["id"]
        intr_path = calibration_dir / Path(cam["intrinsics_file"]).name
        if not intr_path.exists():
            print(f"WARN: intrinsics not found for {cam_id} at {intr_path}. Skipping.")
            continue
        intrinsics = load_intrinsics(intr_path)

        cam_frame_dir = session_dir / cam_id
        if not cam_frame_dir.exists():
            print(f"WARN: frame directory not found for {cam_id}. Skipping.")
            continue
        frame_paths = sorted(cam_frame_dir.glob("*.png"))
        if not frame_paths:
            print(f"WARN: no frames found for {cam_id}. Skipping.")
            continue

        print(f"\nEstimating pose for {cam_id} ({len(frame_paths)} frames) …")
        try:
            pose = estimate_camera_pose(
                frame_paths, intrinsics, box_cfg,
                min_markers=min_markers, max_reproj_px=max_reproj_px,
            )
            poses[cam_id] = pose
            entry: dict = {
                "T_cam_box": pose.T_cam_box.tolist(),
                "reprojection_error_px": pose.reprojection_error,
                "n_frames_used": pose.n_frames_used,
            }
            if pose.pose_covariance is not None:
                entry["pose_covariance"] = pose.pose_covariance.tolist()
            results_json["poses"][cam_id] = entry
        except RuntimeError as exc:
            print(f"ERROR: {exc}")

    save_json(results_json, output_path)
    print(f"\nExtrinsics saved to {output_path}")
    return poses


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extrinsic pose estimation via ArUco board")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--box-config", type=Path, default=Path("config/box.yaml"))
    p.add_argument("--cameras-config", type=Path, default=Path("config/cameras.yaml"))
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-markers", type=int, default=3)
    p.add_argument("--max-reproj-px", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (args.session / "extrinsics.json")
    solve_session(
        session_dir=args.session,
        box_config_path=args.box_config,
        cameras_config_path=args.cameras_config,
        calibration_dir=args.calibration_dir,
        output_path=output,
        min_markers=args.min_markers,
        max_reproj_px=args.max_reproj_px,
    )


if __name__ == "__main__":
    main()
