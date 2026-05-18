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

    Marker poses expressed in box-centered frame (corner-at-origin) after
    best-fit alignment to box faces. Each marker carries its inferred face
    label + on-plane residual.
    """
    raw_cfg = load_yaml(raw_box_cfg_path)
    ref_corners = get_refined_corners_m(result, marker_side_m)
    face_assigns = getattr(result, "face_assigns", [None] * result.n_markers)

    patch: dict[int, dict] = {}
    for i, mid in enumerate(result.marker_ids):
        T = result.marker_poses[i]
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        entry = {
            "corners_box_frame": (ref_corners[i] * 1000.0).tolist(),
            "pose_translation_mm": (T[:3, 3] * 1000.0).tolist(),
            "pose_rotvec_deg": np.degrees(rvec.ravel()).tolist(),
            "reprojection_rms_px": round(float(result.per_marker_rms[i]), 4),
            "n_observations": int(result.n_obs_per_marker[i]),
        }
        if face_assigns[i] is not None:
            entry["face"] = face_assigns[i]
        patch[mid] = entry

    # Stale keys to remove from each marker entry.
    stale = (
        "center_box_mm", "rotation_deg",
        "offset_translation_mm", "offset_rotation_deg",
        "is_anchor",
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
    box_cfg: dict,
    debug_dir: Path,
) -> None:
    """Detected corners, marker reprojections, and transparent fitted box."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    n_cams = result.n_cams
    corners_mkr = marker_corners_mkr_frame(marker_side_m)
    corners_hom = np.hstack([corners_mkr, np.ones((4, 1))])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dims = box_cfg.get("box_dimensions", {})
    W = float(dims.get("width_mm", 0.0)) / 1000.0
    H = float(dims.get("height_mm", 0.0)) / 1000.0
    D = float(dims.get("depth_mm", 0.0)) / 1000.0
    has_box = W > 0.0 and H > 0.0 and D > 0.0

    from .bundle import _se3_exp_batch
    cam_xis = result.x[: 6 * n_cams].reshape(n_cams, 6)
    T_cams = _se3_exp_batch(cam_xis)

    cam_dets: dict[int, list] = {i: [] for i in range(n_cams)}
    for cam_idx, mk_idx, obs in result.detection_list:
        cam_dets[cam_idx].append((mk_idx, obs))

    for cam_idx, (path, _, img_ud) in enumerate(detections):
        vis = img_ud.copy()
        T_cam_box = T_cams[cam_idx]

        if has_box:
            _draw_transparent_box(vis, T_cam_box, fx, fy, cx, cy, W, H, D)

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

    print(f"  Debug overlays → {debug_dir}/ (green=detected, red=reprojected, cyan=fit box)")


def _draw_transparent_box(
    image: np.ndarray,
    T_cam_box: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    W: float,
    H: float,
    D: float,
) -> None:
    """Project the fitted box and composite visible faces with transparency."""
    corners_box = np.array([
        [0.0, 0.0, 0.0],
        [W,   0.0, 0.0],
        [W,   H,   0.0],
        [0.0, H,   0.0],
        [0.0, 0.0, D],
        [W,   0.0, D],
        [W,   H,   D],
        [0.0, H,   D],
    ], dtype=np.float64)
    faces = [
        ([0, 1, 2, 3], np.array([0.0, 0.0, -1.0])),  # front
        ([4, 5, 6, 7], np.array([0.0, 0.0,  1.0])),  # back
        ([0, 1, 5, 4], np.array([0.0, -1.0, 0.0])),  # bottom
        ([3, 2, 6, 7], np.array([0.0,  1.0, 0.0])),  # top
        ([0, 3, 7, 4], np.array([-1.0, 0.0, 0.0])),  # left
        ([1, 2, 6, 5], np.array([ 1.0, 0.0, 0.0])),  # right
    ]

    pts_cam = (T_cam_box[:3, :3] @ corners_box.T).T + T_cam_box[:3, 3]
    if np.any(pts_cam[:, 2] <= 1e-6):
        return

    proj = np.empty((8, 2), dtype=np.float64)
    proj[:, 0] = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    proj[:, 1] = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy

    overlay = image.copy()
    fill_color = (255, 255, 0)
    edge_color = (255, 255, 0)

    face_depths: list[tuple[float, list[int]]] = []
    for idxs, normal_box in faces:
        normal_cam = T_cam_box[:3, :3] @ normal_box
        if normal_cam[2] >= 0.0:
            continue
        depth = float(np.mean(pts_cam[idxs, 2]))
        face_depths.append((depth, idxs))

    for _, idxs in sorted(face_depths, reverse=True):
        poly = np.round(proj[idxs]).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, fill_color, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.18, image, 0.82, 0.0, dst=image)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i0, i1 in edges:
        p0 = tuple(np.round(proj[i0]).astype(int))
        p1 = tuple(np.round(proj[i1]).astype(int))
        cv2.line(image, p0, p1, edge_color, 2, lineType=cv2.LINE_AA)


_MARKER_PALETTE = [
    '#e74c3c', '#2ecc71', '#3498db', '#9b59b6',
    '#f39c12', '#1abc9c', '#e67e22', '#e91e8c',
    '#16a085', '#d35400', '#8e44ad', '#27ae60',
]


def save_3d_plot(
    result: BundleResult,
    box_cfg: dict,
    marker_side_m: float,
    debug_dir: Path,
) -> None:
    """Styled multi-view plot: 3D iso + top/side/front ortho + legend."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  3D plot skipped: matplotlib not available")
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    ref_corners_mm = [c * 1000.0 for c in get_refined_corners_m(result, marker_side_m)]
    ids = result.marker_ids
    n_obs = result.n_obs_per_marker
    face_assigns = getattr(result, "face_assigns", [None] * len(ids))
    colors = {mid: _MARKER_PALETTE[i % len(_MARKER_PALETTE)]
              for i, mid in enumerate(ids)}

    # Box dims, corner-at-origin frame: x∈[0,W], y∈[0,H], z∈[0,D].
    dims = box_cfg.get("box_dimensions", {})
    W = float(dims.get("width_mm", 0.0))
    H = float(dims.get("height_mm", 0.0))
    D = float(dims.get("depth_mm", 0.0))
    has_box = W > 0 and H > 0 and D > 0

    all_pts = np.vstack(ref_corners_mm)
    pad = 10.0
    xlim = (min(all_pts[:, 0].min(), 0.0 if has_box else 0) - pad,
            max(all_pts[:, 0].max(), W   if has_box else 0) + pad)
    ylim = (min(all_pts[:, 1].min(), 0.0 if has_box else 0) - pad,
            max(all_pts[:, 1].max(), H   if has_box else 0) + pad)
    zlim = (min(all_pts[:, 2].min(), 0.0 if has_box else 0) - pad,
            max(all_pts[:, 2].max(), D   if has_box else 0) + pad)

    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle(
        f'Refined ArUco Markers — {W:.0f} × {D:.0f} × {H:.0f} mm' if has_box
        else 'Refined ArUco Markers',
        color='white', fontsize=16, fontweight='bold', y=0.97,
    )

    # ── 3D iso view ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#16213e')

    if has_box:
        x0, x1 = 0.0, W
        y0, y1 = 0.0, H
        z0, z1 = 0.0, D
        box_faces = [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
        ]
        ax3.add_collection3d(Poly3DCollection(
            box_faces, alpha=0.08, linewidth=0.8,
            edgecolor='#7f8c8d', facecolor='#2c3e50'))

    for i, mid in enumerate(ids):
        c = ref_corners_mm[i]
        color = colors[mid]
        ax3.add_collection3d(Poly3DCollection(
            [c.tolist()], alpha=0.7, linewidth=1.5,
            edgecolor=color, facecolor=color))
        cx, cy, cz = c.mean(axis=0)
        face_tag = face_assigns[i] if face_assigns[i] else ''
        lbl = f'#{mid}' + (f'·{face_tag[:1].upper()}' if face_tag else '')
        ax3.text(cx, cy, cz, lbl, color='white', fontsize=7.5,
                 fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                           alpha=0.75, edgecolor='none'))

    ax3.set_xlabel('X (mm)', color='#bdc3c7', fontsize=8, labelpad=6)
    ax3.set_ylabel('Y (mm)', color='#bdc3c7', fontsize=8, labelpad=6)
    ax3.set_zlabel('Z (mm)', color='#bdc3c7', fontsize=8, labelpad=6)
    ax3.tick_params(colors='#7f8c8d', labelsize=7)
    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#2c3e50')
    ax3.grid(True, color='#2c3e50', linewidth=0.5)
    ax3.set_xlim(*xlim); ax3.set_ylim(*ylim); ax3.set_zlim(*zlim)
    ax3.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
    ax3.set_title('3D View (isometric)', color='#ecf0f1', fontsize=11, pad=8)
    ax3.view_init(elev=22, azim=-55)

    # ── Ortho projections ────────────────────────────────────────────────
    ax_top   = fig.add_axes([0.54, 0.55, 0.21, 0.38], facecolor='#16213e')
    ax_side  = fig.add_axes([0.78, 0.55, 0.21, 0.38], facecolor='#16213e')
    ax_front = fig.add_axes([0.54, 0.08, 0.21, 0.38], facecolor='#16213e')
    ax_leg   = fig.add_axes([0.78, 0.08, 0.21, 0.38], facecolor='#16213e')

    def draw_box_rect(ax, rx, ry, rw, rh):
        rect = plt.Rectangle((rx, ry), rw, rh,
                             linewidth=1.2, edgecolor='#7f8c8d',
                             facecolor='#2c3e50', alpha=0.4)
        ax.add_patch(rect)

    def style_ortho(ax, title, xlabel, ylabel, xl, yl):
        ax.set_title(title, color='#ecf0f1', fontsize=9, pad=4)
        ax.set_xlabel(xlabel, color='#bdc3c7', fontsize=7)
        ax.set_ylabel(ylabel, color='#bdc3c7', fontsize=7)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.tick_params(colors='#7f8c8d', labelsize=6)
        ax.grid(True, color='#2c3e50', linewidth=0.4, linestyle='--')
        ax.set_aspect('equal')
        for s in ax.spines.values():
            s.set_edgecolor('#34495e')

    def plot_ortho(ax, ai, bi):
        for i, mid in enumerate(ids):
            c = ref_corners_mm[i]
            xs = c[:, ai].tolist() + [c[0, ai]]
            ys = c[:, bi].tolist() + [c[0, bi]]
            ax.fill(xs, ys, color=colors[mid], alpha=0.55)
            ax.plot(xs, ys, color=colors[mid], lw=1)
            ax.text(c[:, ai].mean(), c[:, bi].mean(), str(mid),
                    color='white', fontsize=6, ha='center', va='center',
                    fontweight='bold')

    if has_box:
        draw_box_rect(ax_top,   0.0, 0.0, W, H)
        draw_box_rect(ax_side,  0.0, 0.0, H, D)
        draw_box_rect(ax_front, 0.0, 0.0, W, D)
    plot_ortho(ax_top,   0, 1)
    plot_ortho(ax_side,  1, 2)
    plot_ortho(ax_front, 0, 2)
    style_ortho(ax_top,   'Top (X–Y)',   'X (mm)', 'Y (mm)', xlim, ylim)
    style_ortho(ax_side,  'Side (Y–Z)',  'Y (mm)', 'Z (mm)', ylim, zlim)
    style_ortho(ax_front, 'Front (X–Z)', 'X (mm)', 'Z (mm)', xlim, zlim)

    # ── Fit residuals per marker (right bottom) ─────────────────────────
    ax_leg.set_facecolor('#16213e')
    ax_leg.set_title('Box-fit residuals per marker',
                     color='#ecf0f1', fontsize=9, pad=4)
    fit_info = getattr(result, "fit_info", None)
    if fit_info is not None:
        plane_mm = np.asarray(fit_info["plane_resid_mm"])
        norm_deg = np.asarray(fit_info["normal_resid_deg"])
        x = np.arange(len(ids))
        bar_w = 0.38
        bar_colors = [colors[mid] for mid in ids]

        ax_leg.bar(x - bar_w / 2, np.abs(plane_mm), bar_w,
                   color=bar_colors, edgecolor='white', linewidth=0.4,
                   label='|plane| mm')
        ax_leg.set_ylabel('|plane resid| mm', color='#bdc3c7', fontsize=7)
        ax_leg.tick_params(axis='y', colors='#bdc3c7', labelsize=6)

        ax2 = ax_leg.twinx()
        ax2.bar(x + bar_w / 2, norm_deg, bar_w,
                color=bar_colors, edgecolor='white', linewidth=0.4,
                hatch='///', alpha=0.85, label='normal °')
        ax2.set_ylabel('normal resid °', color='#bdc3c7', fontsize=7)
        ax2.tick_params(axis='y', colors='#bdc3c7', labelsize=6)
        for s in ax2.spines.values():
            s.set_edgecolor('#34495e')

        labels_x = [f"{mid}\n[{(face_assigns[i] or '—')[:3]}]"
                    for i, mid in enumerate(ids)]
        ax_leg.set_xticks(x)
        ax_leg.set_xticklabels(labels_x, color='#bdc3c7', fontsize=6)
        ax_leg.set_xlabel('marker id / face', color='#bdc3c7', fontsize=7)
        ax_leg.grid(True, axis='y', color='#2c3e50', linewidth=0.4, linestyle='--')
        for s in ax_leg.spines.values():
            s.set_edgecolor('#34495e')

        # Inline legend for the two bar types.
        from matplotlib.patches import Patch as _Patch
        ax_leg.legend(handles=[
            _Patch(facecolor='#7f8c8d', edgecolor='white', label='|plane| mm'),
            _Patch(facecolor='#7f8c8d', edgecolor='white', hatch='///',
                   label='normal °'),
        ], loc='upper right', fontsize=6, facecolor='#16213e',
            edgecolor='#34495e', labelcolor='white', framealpha=0.85)
    else:
        ax_leg.axis('off')
        ax_leg.text(0.5, 0.5, 'no fit_info attached',
                    color='#7f8c8d', ha='center', va='center', fontsize=8)

    info = (
        f"Markers: {len(ids)} on faces\n"
        f"Frame: box-corner-at-origin\n"
        f"Marker side: {marker_side_m*1000:.1f} mm\n"
        f"Global RMS: {float(np.mean(result.per_marker_rms)):.3f} px"
    )
    fig.text(0.54, 0.03, info, color='#95a5a6', fontsize=7.5,
             va='bottom', family='monospace')

    out = debug_dir / "markers_3d.png"
    plt.savefig(str(out), dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  3D plot → {out}")
