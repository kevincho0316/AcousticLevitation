"""
3D scene plot: box, ArUco markers, cameras, and the triangulated ball,
all drawn in the box coordinate frame (millimetres).

Data sources (all optional — whatever is present gets drawn):
  - box config              -> box wireframe + marker quads
  - <session>/extrinsics.json     -> camera centers + view frustums
  - <session>/triangulation.json  -> ball position

Usage (standalone window):
    python -m visualization.scene_3d \\
        --session session/session_1 \\
        --box-config config/box.yaml

The GUI imports `render_scene` to draw into an embedded matplotlib canvas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import load_box_config, load_json

M_TO_MM = 1000.0

# face name -> colour
_FACE_COLOURS = {
    "front": "#e57373", "back": "#64b5f6",
    "left": "#81c784", "right": "#ffb74d",
    "top": "#ba68c8", "bottom": "#4db6ac",
}
_CAM_COLOURS = ["#ff5252", "#448aff", "#00c853", "#ffab00",
                "#d500f9", "#00bcd4", "#ff6d00", "#c6ff00"]

# matplotlib always draws its Z axis vertical. Box frame has height on Y, so
# remap box (X,Y,Z) -> plot (X, Z, Y): the vertical plot axis becomes height,
# OpenGL-style. Every point is passed through _to_plot before being drawn.
_PLOT_PERM = [0, 2, 1]


def _to_plot(pts) -> np.ndarray:
    """Box-frame (X,Y,Z) -> plot-frame (X, Z, Y) so height points up."""
    return np.asarray(pts, dtype=float)[..., _PLOT_PERM]


# ── Scene loading ─────────────────────────────────────────────────────────────

def load_scene(session_dir: Path | None, box_config_path: Path) -> dict:
    """Collect everything drawable into a plain dict (all lengths in mm)."""
    box_cfg = load_box_config(box_config_path)
    scene: dict = {"markers": [], "cameras": [], "ball": None, "box_dims": None}

    # Box frame axes: X = width, Y = height, Z = depth (matches box.yaml
    # marker coords — left/right faces on X, top/bottom on Y, front/back on Z).
    dims = box_cfg.get("box_dimensions")
    if isinstance(dims, dict):
        scene["box_dims"] = (float(dims["width_mm"]),
                             float(dims["height_mm"]),
                             float(dims["depth_mm"]))

    for marker in box_cfg["markers"]:
        corners_m = marker.get("corners_box_frame_m")
        if corners_m is None:
            continue
        scene["markers"].append({
            "id": int(marker["id"]),
            "face": marker.get("face", "?"),
            "corners_mm": np.asarray(corners_m, dtype=float) * M_TO_MM,  # (4,3)
        })

    if session_dir is not None:
        extr_path = session_dir / "extrinsics.json"
        if extr_path.exists():
            poses = load_json(extr_path).get("poses", {})
            for cam_id, p in poses.items():
                T = np.asarray(p["T_cam_box"], dtype=float)  # box -> cam
                R, t = T[:3, :3], T[:3, 3]
                center_mm = (-R.T @ t) * M_TO_MM            # camera center, box frame
                scene["cameras"].append({
                    "id": cam_id,
                    "R": R,                                 # box -> cam rotation
                    "center_mm": center_mm,
                    "reproj_px": p.get("reprojection_error_px"),
                })

        tri_path = session_dir / "triangulation.json"
        if tri_path.exists():
            tri = load_json(tri_path)
            pos = tri.get("position_box_m")
            if pos is not None:
                scene["ball"] = {
                    "pos_mm": np.asarray(pos, dtype=float) * M_TO_MM,
                    "n_cameras": tri.get("n_cameras"),
                }

    # Recenter so the box center sits at the origin (0,0,0).
    if scene["box_dims"]:
        W, H, D = scene["box_dims"]
        center = np.array([W / 2.0, H / 2.0, D / 2.0])
    elif scene["markers"]:
        allc = np.vstack([m["corners_mm"] for m in scene["markers"]])
        center = (allc.min(axis=0) + allc.max(axis=0)) / 2.0
    else:
        center = np.zeros(3)
    scene["box_center_mm"] = center
    for m in scene["markers"]:
        m["corners_mm"] = m["corners_mm"] - center
    for cam in scene["cameras"]:
        cam["center_mm"] = cam["center_mm"] - center
    if scene["ball"] is not None:
        scene["ball"]["pos_mm"] = scene["ball"]["pos_mm"] - center
    return scene


# ── Drawing primitives ────────────────────────────────────────────────────────

def _draw_box(ax, dims) -> None:
    """Wireframe cuboid centered on the origin (X=width, Y=height, Z=depth)."""
    W, H, D = dims
    x, y, z = W / 2.0, H / 2.0, D / 2.0
    pts = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                    [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=float)
    pts = _to_plot(pts)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(pts[a], pts[b]), color="#888888", lw=1.0,
                ls="--", alpha=0.6)


def _draw_markers(ax) -> None:
    pass  # placeholder kept for symmetry; markers drawn inline below


def _draw_camera(ax, cam: dict, colour: str) -> None:
    """Camera center + a small view frustum pointing along its optical axis."""
    C = cam["center_mm"]
    R = cam["R"]                       # box -> cam
    Rt = R.T                          # cam -> box (columns = cam axes in box frame)

    # Frustum: 4 image-corner rays at depth d (cam-frame +Z is the optical axis).
    d = 25.0
    a = 14.0
    corners_cam = np.array([[-a, -a, d], [a, -a, d],
                            [a, a, d], [-a, a, d]], dtype=float)
    corners_box = (Rt @ corners_cam.T).T + C

    Cp = _to_plot(C)
    cp = _to_plot(corners_box)
    ax.scatter(*Cp, color=colour, s=70, marker="^",
               edgecolors="black", linewidths=0.6, depthshade=False)
    ax.text(Cp[0], Cp[1], Cp[2] + 6, cam["id"], color=colour,
            fontsize=8, weight="bold")
    for cb in cp:
        ax.plot(*zip(Cp, cb), color=colour, lw=0.8, alpha=0.7)
    loop = np.vstack([cp, cp[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=colour, lw=0.9, alpha=0.7)


def _draw_ball(ax, ball: dict) -> None:
    p_box = ball["pos_mm"]
    p = _to_plot(p_box)
    ax.scatter(*p, color="#ffffff", s=160, marker="o",
               edgecolors="#1565c0", linewidths=1.6, depthshade=False)
    # Label shows box-frame coords (X, Y, Z), not the remapped plot order.
    label = f"ball ({p_box[0]:.1f}, {p_box[1]:.1f}, {p_box[2]:.1f}) mm"
    ax.text(p[0], p[1], p[2] + 8, label, color="#1565c0",
            fontsize=8, weight="bold")


def _set_equal_aspect(ax, all_pts: np.ndarray) -> None:
    """Force a cubic bounding box so geometry is not distorted."""
    if all_pts.size == 0:
        return
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max()) * 0.6 + 1e-6
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


# ── Top-level render ──────────────────────────────────────────────────────────

def render_scene(session_dir: Path | None, box_config_path: Path, ax) -> dict:
    """Draw the full scene onto a 3D axes. Returns the loaded scene dict."""
    scene = load_scene(session_dir, box_config_path)
    # Preserve the viewing angle across refreshes (ax.clear resets it).
    elev, azim = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev=elev, azim=azim)
    collected: list[np.ndarray] = []

    if scene["box_dims"]:
        _draw_box(ax, scene["box_dims"])
        W, H, D = scene["box_dims"]
        collected.append(_to_plot(np.array(
            [[-W / 2, -H / 2, -D / 2], [W / 2, H / 2, D / 2]], dtype=float)))

    # Markers: filled quads coloured by face, id label at the centroid.
    drawn_faces: set[str] = set()
    for m in scene["markers"]:
        c = _to_plot(m["corners_mm"])
        colour = _FACE_COLOURS.get(m["face"], "#9e9e9e")
        face_label = m["face"] if m["face"] not in drawn_faces else None
        drawn_faces.add(m["face"])
        poly = np.vstack([c, c[0]])
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color=colour, lw=1.4,
                label=face_label)
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax.add_collection3d(Poly3DCollection(
                [c], facecolor=colour, alpha=0.35, edgecolor=colour))
        except Exception:
            pass
        cen = c.mean(axis=0)
        ax.text(cen[0], cen[1], cen[2], str(m["id"]),
                color="black", fontsize=7, ha="center", va="center")
        collected.append(c)

    for i, cam in enumerate(scene["cameras"]):
        _draw_camera(ax, cam, _CAM_COLOURS[i % len(_CAM_COLOURS)])
        collected.append(_to_plot(cam["center_mm"])[None, :])

    if scene["ball"] is not None:
        _draw_ball(ax, scene["ball"])
        collected.append(_to_plot(scene["ball"]["pos_mm"])[None, :])

    all_pts = np.vstack(collected) if collected else np.empty((0, 3))
    _set_equal_aspect(ax, all_pts)

    # Axes are in plot order (X, Z, Y); vertical axis is box-frame height.
    ax.set_xlabel("X · width (mm)")
    ax.set_ylabel("Z · depth (mm)")
    ax.set_zlabel("Y · height (mm)")
    n_cam = len(scene["cameras"])
    n_mk = len(scene["markers"])
    has_ball = "ball" if scene["ball"] is not None else "no ball"
    ax.set_title(f"Box frame — {n_mk} markers, {n_cam} cameras, {has_ball}")
    if drawn_faces:
        ax.legend(loc="upper left", fontsize=7)
    return scene


def enable_scroll_zoom(ax, step: float = 0.85) -> int:
    """Wheel up = zoom in, wheel down = zoom out. Returns the mpl callback id."""
    def _on_scroll(event):
        if event.inaxes is not ax:
            return
        factor = step if event.button == "up" else 1.0 / step
        for get_lim, set_lim in (
            (ax.get_xlim3d, ax.set_xlim3d),
            (ax.get_ylim3d, ax.set_ylim3d),
            (ax.get_zlim3d, ax.set_zlim3d),
        ):
            lo, hi = get_lim()
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * factor
            set_lim(mid - half, mid + half)
        ax.figure.canvas.draw_idle()

    return ax.figure.canvas.mpl_connect("scroll_event", _on_scroll)


def build_figure(session_dir: Path | None, box_config_path: Path):
    """Create a standalone Figure with the scene rendered."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    render_scene(session_dir, box_config_path, ax)
    enable_scroll_zoom(ax)
    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="3D scene plot (box, markers, cameras, ball)")
    p.add_argument("--session", type=Path, default=None,
                   help="session dir holding extrinsics.json / triangulation.json")
    p.add_argument("--box-config", type=Path, default=Path("config/box.yaml"))
    p.add_argument("--save", type=Path, default=None,
                   help="save PNG instead of opening a window")
    args = p.parse_args()

    import matplotlib
    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = build_figure(args.session, args.box_config)
    if args.save is not None:
        fig.savefig(args.save, dpi=130)
        print(f"Saved 3D scene to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
