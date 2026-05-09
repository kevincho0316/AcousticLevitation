"""Render synthetic frames: ArUco markers on box faces + white ball."""
from __future__ import annotations

import numpy as np
import cv2

from synthetic_tests.synth.scene import SyntheticScene


def _project(K: np.ndarray, T_cam_box: np.ndarray, X_box: np.ndarray) -> tuple[np.ndarray, float]:
    """Project box-frame point. Returns (uv, Z_cam). Z_cam ≤ 0 means behind camera."""
    Xc = T_cam_box[:3, :3] @ X_box + T_cam_box[:3, 3]
    if Xc[2] <= 0:
        return np.zeros(2), Xc[2]
    u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
    v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
    return np.array([u, v]), Xc[2]


def _render_marker(
    canvas: np.ndarray,
    K: np.ndarray,
    T_cam_box: np.ndarray,
    corners_box: np.ndarray,   # (4,3)
    marker_id: int,
    aruco_dict,
    face_normal: np.ndarray | None = None,
    marker_px: int = 200,
) -> None:
    """Warp one ArUco marker onto canvas using perspective projection."""
    h_img, w_img = canvas.shape[:2]

    # Skip if face is pointing away from or edge-on to camera.
    if face_normal is not None:
        normal_cam = T_cam_box[:3, :3] @ face_normal
        if normal_cam[2] >= -0.05:   # edge-on or facing away
            return

    # Project all 4 corners; skip if any behind camera.
    dst_pts = []
    for corner in corners_box:
        uv, Z = _project(K, T_cam_box, corner)
        if Z <= 1e-4:
            return
        dst_pts.append(uv)
    dst = np.array(dst_pts, dtype=np.float32)  # (4,2)

    # Skip if all projected corners are far outside the image.
    margin = max(w_img, h_img)
    if (dst[:, 0].max() < -margin or dst[:, 0].min() > w_img + margin or
            dst[:, 1].max() < -margin or dst[:, 1].min() > h_img + margin):
        return

    # Generate marker image with a white quiet zone (1 cell wide).
    cell = marker_px // 6   # DICT_4X4 = 4 data + 1 border = 6 cells total
    pad = cell              # 1-cell quiet zone
    total = marker_px + 2 * pad
    marker_img = np.ones((total, total), dtype=np.uint8) * 255
    m = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
    marker_img[pad:pad + marker_px, pad:pad + marker_px] = m

    # Check winding order of projected quad via shoelace.
    # shoelace > 0 → CW in screen (Y-down) → correct ArUco orientation.
    # shoelace < 0 → CCW → marker projects mirrored; swap L/R src corners to fix.
    shoelace = sum(
        dst[i, 0] * dst[(i + 1) % 4, 1] - dst[(i + 1) % 4, 0] * dst[i, 1]
        for i in range(4)
    )
    if shoelace >= 0:
        src_pts = np.float32([
            [pad, pad],
            [pad + marker_px, pad],
            [pad + marker_px, pad + marker_px],
            [pad, pad + marker_px],
        ])
    else:
        # Swap left/right source columns so the projected marker is not mirrored.
        src_pts = np.float32([
            [pad + marker_px, pad],
            [pad, pad],
            [pad, pad + marker_px],
            [pad + marker_px, pad + marker_px],
        ])

    H_mat, _ = cv2.findHomography(src_pts, dst)
    if H_mat is None:
        return

    warped = cv2.warpPerspective(marker_img, H_mat, (w_img, h_img),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

    # Composite: paint warped marker over canvas (max blend keeps white ball on top later).
    np.maximum(canvas, warped, out=canvas)


def _render_ball(
    canvas: np.ndarray,
    K: np.ndarray,
    T_cam_box: np.ndarray,
    X_ball: np.ndarray,
    radius_m: float,
    upsample: int = 4,
) -> None:
    """Draw white filled circle for ball with sub-pixel accuracy via upsampling."""
    uv_center, Z = _project(K, T_cam_box, X_ball)
    if Z <= 0:
        return

    # Projected radius: use a point on the sphere silhouette.
    # Approximate: r_px = fx * radius_m / Z
    fx = K[0, 0]
    r_px = fx * radius_m / Z

    h_img, w_img = canvas.shape[:2]
    hu, hw = h_img * upsample, w_img * upsample

    # Draw on upsampled canvas then downsample for anti-aliased sub-pixel placement.
    up_canvas = np.zeros((hu, hw), dtype=np.uint8)
    uc = int(round(uv_center[0] * upsample))
    vc = int(round(uv_center[1] * upsample))
    rc = int(round(r_px * upsample))
    cv2.circle(up_canvas, (uc, vc), rc, 255, -1, cv2.LINE_AA)

    ball_layer = cv2.resize(up_canvas, (w_img, h_img), interpolation=cv2.INTER_AREA)
    np.maximum(canvas, ball_layer, out=canvas)


def render_frame(
    scene: SyntheticScene,
    cam_id: str,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render one synthetic frame for a given camera. Returns BGR image (H×W×3)."""
    cam = next(c for c in scene.cameras if c["id"] == cam_id)
    intr = cam["intrinsics"]
    T = cam["T_cam_box"]
    K = intr.K
    w, h = scene.image_size

    canvas = np.zeros((h, w), dtype=np.uint8)

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, scene.aruco_dict_name)
    )

    for marker in scene.markers:
        _render_marker(canvas, K, T, marker["corners_box_frame_m"],
                       marker["id"], aruco_dict,
                       face_normal=marker.get("face_normal"))

    _render_ball(canvas, K, T, scene.ball_position_box, scene.ball_radius_m)

    if noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0, noise_sigma, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Convert to BGR (3-channel) so cv2.imread-compatible code works.
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def render_scene(
    scene: SyntheticScene,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Render all cameras. Returns {cam_id: BGR image}."""
    return {
        c["id"]: render_frame(scene, c["id"], noise_sigma=noise_sigma, rng=rng)
        for c in scene.cameras
    }


def write_session_frames(
    scene: SyntheticScene,
    session_dir,
    n_frames: int = 10,
    noise_sigma: float = 3.0,
    seed: int = 0,
) -> None:
    """Write synthetic session frames to disk in the pipeline's expected layout."""
    from pathlib import Path
    session_dir = Path(session_dir)
    rng = np.random.default_rng(seed)

    for cam in scene.cameras:
        cam_dir = session_dir / cam["id"]
        cam_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            img = render_frame(scene, cam["id"], noise_sigma=noise_sigma, rng=rng)
            cv2.imwrite(str(cam_dir / f"frame_{i:04d}.png"), img)
