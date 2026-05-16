"""Output: enriched YAML, debug overlays, 3D plot for self-calibration result."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common.io_utils import load_yaml, save_yaml

from .bundle import BundleResult
from .faces import marker_corners_mkr_frame


def get_refined_corners_m(result: BundleResult, marker_side_m: float) -> list[np.ndarray]:
    """(4,3) refined corner positions in box frame (meters) per marker."""
    corners_mkr = marker_corners_mkr_frame(marker_side_m)
    corners_hom = np.hstack([corners_mkr, np.ones((4, 1))])
    out = []
    for i in range(result.n_markers):
        T = result.marker_poses[i]
        out.append((T @ corners_hom.T).T[:, :3])
    return out


def write_output_yaml(
    raw_box_cfg_path: Path,
    box_cfg: dict,
    result: BundleResult,
    marker_side_m: float,
    output_path: Path,
) -> None:
    """Write enriched box YAML with refined corners + per-marker diagnostics.

    Drops face/center_box_mm/rotation_deg/offset_* fields (no longer
    meaningful). Adds is_anchor on the anchor marker.
    """
    raw_cfg = load_yaml(raw_box_cfg_path)
    ref_corners = get_refined_corners_m(result, marker_side_m)

    patch: dict[int, dict] = {}
    for i, mid in enumerate(result.marker_ids):
        T = result.marker_poses[i]
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        patch[mid] = {
            "corners_box_frame": (ref_corners[i] * 1000.0).tolist(),
            "pose_translation_mm": (T[:3, 3] * 1000.0).tolist(),
            "pose_rotvec_deg": np.degrees(rvec.ravel()).tolist(),
            "reprojection_rms_px": round(float(result.per_marker_rms[i]), 4),
            "n_observations": int(result.n_obs_per_marker[i]),
            "is_anchor": (i == result.anchor_idx),
        }

    # Stale keys to remove from each marker entry.
    stale = (
        "face", "center_box_mm", "rotation_deg",
        "offset_translation_mm", "offset_rotation_deg",
    )

    for m in raw_cfg["markers"]:
        mid = int(m["id"])
        if mid not in patch:
            continue
        for k in stale:
            m.pop(k, None)
        for k, v in patch[mid].items():
            m[k] = v

    save_yaml(raw_cfg, output_path)
    print(f"  Output → {output_path}")


def save_debug_overlays(
    detections: list,
    result: BundleResult,
    K: np.ndarray,
    marker_side_m: float,
    debug_dir: Path,
) -> None:
    """Detected corners (green) vs reprojected corners (red) per image."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_cams = result.n_cams
    n_markers = result.n_markers
    corners_mkr = marker_corners_mkr_frame(marker_side_m)
    corners_hom = np.hstack([corners_mkr, np.ones((4, 1))])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    from .bundle import _se3_exp_batch
    cam_xis = result.x[: 6 * n_cams].reshape(n_cams, 6)
    T_cams = _se3_exp_batch(cam_xis)

    cam_dets: dict[int, list] = {i: [] for i in range(n_cams)}
    for cam_idx, mk_idx, obs in result.detection_list:
        cam_dets[cam_idx].append((mk_idx, obs))

    for cam_idx, (path, _, img_ud) in enumerate(detections):
        vis = img_ud.copy()
        T_cam_box = T_cams[cam_idx]

        for mk_idx, obs in cam_dets[cam_idx]:
            T_box_mk = result.marker_poses[mk_idx]
            T_cam_mk = T_cam_box @ T_box_mk
            pts = (T_cam_mk @ corners_hom.T).T[:, :3]
            x_p = fx * pts[:, 0] / pts[:, 2] + cx
            y_p = fy * pts[:, 1] / pts[:, 2] + cy
            proj = np.stack([x_p, y_p], axis=1)

            for pt in obs.astype(int):
                cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
            for pt in proj.astype(int):
                cv2.circle(vis, tuple(pt), 6, (0, 0, 255), 2)
            mid = result.marker_ids[mk_idx]
            cv2.putText(vis, str(mid), tuple(obs[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imwrite(str(debug_dir / f"debug_{Path(path).stem}.jpg"), vis)

    print(f"  Debug overlays → {debug_dir}/ (green=detected, red=reprojected)")


def save_3d_plot(
    result: BundleResult,
    box_cfg: dict,
    marker_side_m: float,
    debug_dir: Path,
) -> None:
    """3D plot of refined marker quads + box-dimension wireframe."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  3D plot skipped: matplotlib not available")
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    ref_corners = get_refined_corners_m(result, marker_side_m)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, mid in enumerate(result.marker_ids):
        q = ref_corners[i] * 1000.0  # mm
        quad = np.vstack([q, q[0]])
        is_anchor = (i == result.anchor_idx)
        label = f"id={mid}" + ("  (anchor)" if is_anchor else "")
        ax.plot(quad[:, 0], quad[:, 2], quad[:, 1],
                color=f"C{i % 10}",
                linewidth=3 if is_anchor else 2,
                label=label)

    # Box wireframe at origin, advisory.
    dims = box_cfg.get("box_dimensions", {})
    W = float(dims.get("width_mm", 0.0))
    H = float(dims.get("height_mm", 0.0))
    D = float(dims.get("depth_mm", 0.0))
    if W > 0 and H > 0 and D > 0:
        verts = np.array([
            [0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0],
            [0, 0, D], [W, 0, D], [W, H, D], [0, H, D],
        ])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            ax.plot([verts[a,0], verts[b,0]],
                    [verts[a,2], verts[b,2]],
                    [verts[a,1], verts[b,1]],
                    color="lightgray", linewidth=0.7, linestyle="--")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Y (mm)")
    ax.set_title("Refined marker positions (anchor = origin)")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    out = debug_dir / "markers_3d.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  3D plot → {out}")
