import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Box dimensions (mm)
W, D, H = 80, 80, 30  # width, depth, height

# Box corners: centered in X/Y, Z from 0 (top) to -H (bottom)
x0, x1 = -W/2, W/2
y0, y1 = -D/2, D/2
z0, z1 = 0, -H

# Define 6 faces of the box
box_faces = [
    # top (z=0)
    [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
    # bottom (z=-H)
    [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
    # front (y=-D/2)
    [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
    # back (y=D/2)
    [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
    # left (x=-W/2)
    [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
    # right (x=W/2)
    [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
]

# Markers: id -> corners (list of [x,y,z])
markers = {
    0: {"corners": [
            [40.46, -7.46, -33.19],
            [40.37,  7.54, -33.08],
            [40.42,  7.65, -48.08],
            [40.52, -7.35, -48.19],
        ], "obs": 4, "anchor": False},
    1: {"corners": [
            [-7.5,  7.5, 0],
            [ 7.5,  7.5, 0],
            [ 7.5, -7.5, 0],
            [-7.5, -7.5, 0],
        ], "obs": 5, "anchor": True},
    2: {"corners": [
            [-42.02,  7.99, -49.72],
            [-42.14,  7.43, -34.73],
            [-41.51, -7.55, -35.29],
            [-41.39, -6.99, -50.28],
        ], "obs": 5, "anchor": False},
    3: {"corners": [
            [-7.72, 10.06, -81.04],
            [-6.92, -4.89, -82.02],
            [ 8.05, -4.05, -82.49],
            [ 7.25, 10.89, -81.51],
        ], "obs": 3, "anchor": False},
    4: {"corners": [
            [39.63, -12.86, -67.22],
            [39.31, -12.24, -82.21],
            [24.31, -12.20, -81.88],
            [24.63, -12.82, -66.90],
        ], "obs": 4, "anchor": False},
    5: {"corners": [
            [40.41, -14.42,  -0.73],
            [40.15, -14.24, -15.73],
            [25.16, -14.11, -15.47],
            [25.41, -14.29,  -0.47],
        ], "obs": 4, "anchor": False},
    6: {"corners": [
            [-32.65, -20.03, -71.49],
            [-39.74, -13.37, -82.91],
            [-28.35,  -3.72, -84.35],
            [-21.26, -10.38, -72.93],
        ], "obs": 1, "anchor": False},
    7: {"corners": [
            [-25.89, -15.38, -15.64],
            [-40.89, -15.48, -15.71],
            [-40.96, -15.88,  -0.72],
            [-25.96, -15.78,  -0.65],
        ], "obs": 5, "anchor": False},
}

# Color palette per marker
marker_colors = {
    0: '#e74c3c', 1: '#2ecc71', 2: '#3498db', 3: '#9b59b6',
    4: '#f39c12', 5: '#1abc9c', 6: '#e67e22', 7: '#e91e8c',
}

fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
fig.suptitle('ArUco Marker Box — 80 × 80 × 30 mm', color='white',
             fontsize=16, fontweight='bold', y=0.97)

# ── 3D view ──────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#16213e')

# Draw box faces
box_poly = Poly3DCollection(box_faces, alpha=0.08, linewidth=0.8,
                            edgecolor='#7f8c8d', facecolor='#2c3e50')
ax3.add_collection3d(box_poly)

# Draw markers
for mid, mdata in markers.items():
    c = np.array(mdata["corners"])
    quad = [c.tolist()]
    color = marker_colors[mid]
    alpha = 0.85 if mdata["anchor"] else 0.6

    poly = Poly3DCollection(quad, alpha=alpha, linewidth=1.5,
                            edgecolor=color, facecolor=color)
    ax3.add_collection3d(poly)

    # Label at centroid
    cx, cy, cz = c.mean(axis=0)
    lbl = f'#{mid}' + (' ⚓' if mdata["anchor"] else '')
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

# Expand limits to show markers outside box (e.g. marker 3 at z≈-82)
ax3.set_xlim(-55, 55)
ax3.set_ylim(-55, 55)
ax3.set_zlim(-95, 15)
ax3.set_title('3D View (isometric)', color='#ecf0f1', fontsize=11, pad=8)
ax3.view_init(elev=22, azim=-55)

# ── Orthographic projections ─────────────────────────────────────────────────
ax_top  = fig.add_axes([0.54, 0.55, 0.21, 0.38], facecolor='#16213e')
ax_side = fig.add_axes([0.78, 0.55, 0.21, 0.38], facecolor='#16213e')
ax_front= fig.add_axes([0.54, 0.08, 0.21, 0.38], facecolor='#16213e')
ax_leg  = fig.add_axes([0.78, 0.08, 0.21, 0.38], facecolor='#16213e')

def draw_box_rect(ax, rx, ry, rw, rh):
    rect = plt.Rectangle((rx, ry), rw, rh,
                          linewidth=1.2, edgecolor='#7f8c8d',
                          facecolor='#2c3e50', alpha=0.4)
    ax.add_patch(rect)

def style_ortho(ax, title, xlabel, ylabel, xlim, ylim):
    ax.set_title(title, color='#ecf0f1', fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color='#bdc3c7', fontsize=7)
    ax.set_ylabel(ylabel, color='#bdc3c7', fontsize=7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.tick_params(colors='#7f8c8d', labelsize=6)
    ax.grid(True, color='#2c3e50', linewidth=0.4, linestyle='--')
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_edgecolor('#34495e')

# TOP VIEW (XY plane, z=0 face)
draw_box_rect(ax_top, x0, y0, W, D)
for mid, mdata in markers.items():
    c = np.array(mdata["corners"])
    xs, ys = c[:,0].tolist()+[c[0,0]], c[:,1].tolist()+[c[0,1]]
    ax_top.fill(xs, ys, color=marker_colors[mid], alpha=0.55)
    ax_top.plot(xs, ys, color=marker_colors[mid], lw=1)
    cx, cy = c[:,0].mean(), c[:,1].mean()
    ax_top.text(cx, cy, str(mid), color='white', fontsize=6,
                ha='center', va='center', fontweight='bold')
style_ortho(ax_top, 'Top (X–Y)', 'X (mm)', 'Y (mm)', (-55,55), (-55,55))

# SIDE VIEW (YZ plane, looking along X)
draw_box_rect(ax_side, y0, -H, D, H)
for mid, mdata in markers.items():
    c = np.array(mdata["corners"])
    ys, zs = c[:,1].tolist()+[c[0,1]], c[:,2].tolist()+[c[0,2]]
    ax_side.fill(ys, zs, color=marker_colors[mid], alpha=0.55)
    ax_side.plot(ys, zs, color=marker_colors[mid], lw=1)
    cy, cz = c[:,1].mean(), c[:,2].mean()
    ax_side.text(cy, cz, str(mid), color='white', fontsize=6,
                 ha='center', va='center', fontweight='bold')
style_ortho(ax_side, 'Side (Y–Z)', 'Y (mm)', 'Z (mm)', (-55,55), (-95,15))

# FRONT VIEW (XZ plane, looking along Y)
draw_box_rect(ax_front, x0, -H, W, H)
for mid, mdata in markers.items():
    c = np.array(mdata["corners"])
    xs, zs = c[:,0].tolist()+[c[0,0]], c[:,2].tolist()+[c[0,2]]
    ax_front.fill(xs, zs, color=marker_colors[mid], alpha=0.55)
    ax_front.plot(xs, zs, color=marker_colors[mid], lw=1)
    cx, cz = c[:,0].mean(), c[:,2].mean()
    ax_front.text(cx, cz, str(mid), color='white', fontsize=6,
                  ha='center', va='center', fontweight='bold')
style_ortho(ax_front, 'Front (X–Z)', 'X (mm)', 'Z (mm)', (-55,55), (-95,15))

# LEGEND
ax_leg.axis('off')
ax_leg.set_title('Markers', color='#ecf0f1', fontsize=9, pad=4)
patches = []
for mid, mdata in markers.items():
    label = f'ID {mid}'
    if mdata["anchor"]: label += '  ⚓ anchor'
    label += f'  (n={mdata["obs"]})'
    patches.append(mpatches.Patch(color=marker_colors[mid], label=label))
ax_leg.legend(handles=patches, loc='center', fontsize=7.5,
              facecolor='#16213e', edgecolor='#34495e',
              labelcolor='white', framealpha=0.9,
              handlelength=1.2, handleheight=1.0)

# Box info text
info = (
    "Box: 80 × 80 × 30 mm\n"
    "ArUco: DICT_4X4_50\n"
    "Marker side: 15 mm\n"
    "Uncertainty: ±1.0 mm"
)
fig.text(0.54, 0.03, info, color='#95a5a6', fontsize=7.5,
         va='bottom', family='monospace')

plt.savefig('/mnt/user-data/outputs/box_aruco_viz.png',
            dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved.")