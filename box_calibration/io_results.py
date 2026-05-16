"""Output: enriched YAML, debug overlays, 3D plot."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common.io_utils import load_yaml, save_yaml
from common.se3_utils import _se3_exp
from .faces import BoxModel
from .bundle import BundleResult, make_T_offset


def get_refined_corners_m(box_model: BoxModel, result: BundleResult) -> list[np.ndarray]:
    """(4,3) refined corner positions in box frame (meters) for each marker."""
    n_markers = len(box_model.ids)
    mk_offs = result.x[6 * result.n_cams :].reshape(n_markers, 6)
    corners_hom = np.hstack([box_model.corners_mkr, np.ones((4, 1))])
    out = []
    for i in range(n_markers):
        T = box_model.nominal_poses[i] @ make_T_offset(mk_offs[i])
        out.append((T @ corners_hom.T).T[:, :3])
    return out


def write_output_yaml(
    raw_box_cfg_path: Path,
    box_model: BoxModel,
    result: BundleResult,
    output_path: Path,
) -> None:
    """Write enriched box YAML: refined corners + per-marker diagnostics."""
    raw_cfg = load_yaml(raw_box_cfg_path)
    n_markers = len(box_model.ids)
    mk_offs = result.x[6 * result.n_cams :].reshape(n_markers, 6)
    ref_corners = get_refined_corners_m(box_model, result)

    patch: dict[int, dict] = {}
    for i, mid in enumerate(box_model.ids):
        patch[mid] = {
            "corners_box_frame": (ref_corners[i] * 1000.0).tolist(),
            "offset_translation_mm": (mk_offs[i, 3:] * 1000.0).tolist(),
            "offset_rotation_deg": np.degrees(mk_offs[i, :3]).tolist(),
            "reprojection_rms_px": round(float(result.per_marker_rms[i]), 4),
            "n_observations": int(result.n_obs_per_marker[i]),
        }

    for m in raw_cfg["markers"]:
        mid = int(m["id"])
        if mid not in patch:
            continue
        d = patch[mid]
        m["corners_box_frame"] = d["corners_box_frame"]
        m["offset_translation_mm"] = d["offset_translation_mm"]
        m["offset_rotation_deg"] = d["offset_rotation_deg"]
        m["reprojection_rms_px"] = d["reprojection_rms_px"]
        m["n_observations"] = d["n_observations"]

    save_yaml(raw_cfg, output_path)
    print(f"  Output → {output_path}")


def save_debug_overlays(
    detections: list,
    result: BundleResult,
    box_model: BoxModel,
    K: np.ndarray,
    debug_dir: Path,
) -> None:
    """Detected corners (green) vs reprojected corners (red) per image."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_cams = result.n_cams
    n_markers = len(box_model.ids)
    cam_xis = result.x[: 6 * n_cams].reshape(n_cams, 6)
    mk_offs = result.x[6 * n_cams :].reshape(n_markers, 6)
    corners_hom = np.hstack([box_model.corners_mkr, np.ones((4, 1))])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    cam_dets: dict[int, list] = {i: [] for i in range(n_cams)}
    for cam_idx, mk_idx, obs in result.detection_list:
        cam_dets[cam_idx].append((mk_idx, obs))

    for cam_idx, (path, _, img_ud) in enumerate(detections):
        vis = img_ud.copy()
        T_cam_box = _se3_exp(cam_xis[cam_idx])

        for mk_idx, obs in cam_dets[cam_idx]:
            T_box_mk = box_model.nominal_poses[mk_idx] @ make_T_offset(mk_offs[mk_idx])
            T_cam_mk = T_cam_box @ T_box_mk
            pts = (T_cam_mk @ corners_hom.T).T[:, :3]
            x_p = fx * pts[:, 0] / pts[:, 2] + cx
            y_p = fy * pts[:, 1] / pts[:, 2] + cy
            proj = np.stack([x_p, y_p], axis=1)

            for pt in obs.astype(int):
                cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
            for pt in proj.astype(int):
                cv2.circle(vis, tuple(pt), 6, (0, 0, 255), 2)
            # Label marker index near first detected corner
            mid = box_model.ids[mk_idx]
            cv2.putText(vis, str(mid), tuple(obs[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imwrite(str(debug_dir / f"debug_{Path(path).stem}.jpg"), vis)

    print(f"  Debug overlays → {debug_dir}/ (green=detected, red=reprojected)")


def save_3d_plot(
    box_model: BoxModel,
    result: BundleResult,
    debug_dir: Path,
) -> None:
    """3D plot: nominal (gray) vs refined (colored) marker corners."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  3D plot skipped: matplotlib not available")
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    n_markers = len(box_model.ids)
    mk_offs = result.x[6 * result.n_cams :].reshape(n_markers, 6)
    corners_hom = np.hstack([box_model.corners_mkr, np.ones((4, 1))])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(n_markers):
        mid = box_model.ids[i]
        face = box_model.faces[i]

        # Nominal (gray)
        nom = (box_model.nominal_poses[i] @ corners_hom.T).T[:, :3] * 1000  # mm
        quad = np.vstack([nom, nom[0]])
        ax.plot(quad[:, 0], quad[:, 2], quad[:, 1], color="lightgray", linewidth=1)

        # Refined (colored)
        T_ref = box_model.nominal_poses[i] @ make_T_offset(mk_offs[i])
        ref = (T_ref @ corners_hom.T).T[:, :3] * 1000  # mm
        quad_r = np.vstack([ref, ref[0]])
        ax.plot(quad_r[:, 0], quad_r[:, 2], quad_r[:, 1],
                color=f"C{i % 10}", linewidth=2, label=f"id={mid} ({face})")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Y (mm)")
    ax.set_title("Nominal (gray) vs Refined (colored) marker positions")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    out = debug_dir / "markers_3d.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  3D plot → {out}")
