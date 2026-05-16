"""
Compare the measured 3D ball position against the simulator's predicted trap.

Both positions are expressed in the box coordinate frame for comparison.
The simulator output is transformed from the simulator frame to the box frame
using the box_to_sim transform (inverted) from box.yaml.

Outputs:
  - Offset vector (measured − simulated) in box frame (mm)
  - Mahalanobis distance: sqrt(rᵀ Σ⁻¹ r), χ²(3) distributed under the null
  - Pass/fail against a configurable Euclidean threshold
  - 3D scatter plot with error ellipsoid and simulated trap location
  - 2D slice plots (XY, XZ, YZ)

Usage:
    python -m comparison.compare \\
        --session sessions/session_001 \\
        --sim-output simulation_outputs/hardware_trap_runs/attempt_004/summary.json \\
        --box-config config/box.yaml \\
        --output sessions/session_001/comparison/ \\
        [--threshold-mm 2.0] [--sim-rank 1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ComparisonResult
from common.io_utils import (
    load_box_config, load_box_to_sim_transform, load_json, save_json,
)


# ── Load simulator output ─────────────────────────────────────────────────────

def load_sim_trap(sim_output_path: Path, rank: int = 1) -> np.ndarray:
    """Load the rank-N trap position (meters, simulator frame) from sim output.

    Supports both summary.json and final_candidates_*.csv from sim.py.
    """
    suffix = sim_output_path.suffix.lower()

    if suffix == ".json":
        data = load_json(sim_output_path)
        key = "ideal_final_candidates"
        if key not in data:
            key = next((k for k in data if "final_candidates" in k), None)
            if key is None:
                raise KeyError(f"No final_candidates key in {sim_output_path}")
        candidates = data[key]
        target = next((c for c in candidates if c.get("rank") == rank), None)
        if target is None:
            raise ValueError(f"Rank {rank} not found in {sim_output_path}.")
        return np.array([target["newton_x"], target["newton_y"], target["newton_z"]], dtype=np.float64)

    elif suffix == ".csv":
        with open(sim_output_path, newline="",encoding='UTF-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["rank"]) == rank:
                    return np.array([
                        float(row["newton_x"]),
                        float(row["newton_y"]),
                        float(row["newton_z"]),
                    ], dtype=np.float64)
        raise ValueError(f"Rank {rank} not found in {sim_output_path}.")

    else:
        raise ValueError(f"Unsupported sim output format: {suffix}. Use .json or .csv.")


# ── Frame transform ───────────────────────────────────────────────────────────

def sim_to_box(p_sim: np.ndarray, T_sim_box: np.ndarray) -> np.ndarray:
    """Transform a point from simulator frame to box frame.

    T_sim_box transforms box → sim, so box → sim: p_sim = T @ p_box
    To go sim → box: p_box = T⁻¹ @ p_sim
    """
    T_box_sim = np.linalg.inv(T_sim_box)
    p_h = np.append(p_sim, 1.0)
    return (T_box_sim @ p_h)[:3]


# ── Mahalanobis distance ──────────────────────────────────────────────────────

def mahalanobis(r: np.ndarray, cov: np.ndarray) -> float:
    """sqrt(rᵀ Σ⁻¹ r)."""
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    return float(np.sqrt(r @ cov_inv @ r))


# ── Main comparison ───────────────────────────────────────────────────────────

def compare(
    measured_position_box: np.ndarray,
    measured_covariance_box: np.ndarray,
    sim_output_path: Path,
    box_cfg: dict,
    threshold_mm: float = 2.0,
    sim_rank: int = 1,
) -> ComparisonResult:
    T_sim_box = load_box_to_sim_transform(box_cfg)
    p_sim = load_sim_trap(sim_output_path, rank=sim_rank)
    p_box_sim = sim_to_box(p_sim, T_sim_box)

    offset = measured_position_box - p_box_sim
    offset_mm = offset * 1000.0
    euclidean_mm = float(np.linalg.norm(offset_mm))
    m_dist = mahalanobis(offset, measured_covariance_box)
    passed = euclidean_mm <= threshold_mm

    print(f"\n{'='*60}")
    print(f"Comparison: measured vs. simulated trap (rank {sim_rank})")
    print(f"{'='*60}")
    print(f"  Measured position (box frame, mm):   {measured_position_box * 1000}")
    print(f"  Simulated position (sim frame, mm):  {p_sim * 1000}")
    print(f"  Simulated position (box frame, mm):  {p_box_sim * 1000}")
    print(f"  Offset (measured - simulated, mm):   {offset_mm}")
    print(f"  Euclidean offset:                    {euclidean_mm:.3f} mm")
    print(f"  Mahalanobis distance:                {m_dist:.3f}  (χ²(3) critical @ 95%: 2.80)")
    print(f"  Threshold:                           {threshold_mm:.1f} mm")
    print(f"  Result:                              {'PASS ✓' if passed else 'FAIL ✗'}")

    return ComparisonResult(
        measured_position_box=measured_position_box,
        measured_covariance_box=measured_covariance_box,
        simulated_position_box=p_box_sim,
        simulated_position_sim=p_sim,
        offset_vector_box=offset,
        mahalanobis_distance=m_dist,
        chi2_dof=3,
        passed=passed,
        threshold_mm=threshold_mm,
        sim_candidate_rank=sim_rank,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def _error_ellipsoid_points(cov: np.ndarray, center: np.ndarray,
                             n_sigma: float = 2.0, n_pts: int = 50
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric surface of the n_sigma error ellipsoid."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    radii = n_sigma * np.sqrt(eigvals)

    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi, n_pts)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (N, 3)
    pts_scaled = pts * radii[None, :]
    pts_rot = (eigvecs @ pts_scaled.T).T + center[None, :]
    X = pts_rot[:, 0].reshape(n_pts, n_pts)
    Y = pts_rot[:, 1].reshape(n_pts, n_pts)
    Z = pts_rot[:, 2].reshape(n_pts, n_pts)
    return X, Y, Z


def plot_comparison(result: ComparisonResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    meas_mm = result.measured_position_box * 1000.0
    sim_mm = result.simulated_position_box * 1000.0
    cov_mm2 = result.measured_covariance_box * 1e6

    # ── 3D plot with error ellipsoid ──────────────────────────────────────────
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    Xs, Ys, Zs = _error_ellipsoid_points(cov_mm2, meas_mm, n_sigma=2.0)
    ax.plot_surface(Xs, Ys, Zs, alpha=0.15, color="steelblue")
    ax.scatter(*meas_mm, color="steelblue", s=80, zorder=5, label="Measured (2σ ellipsoid)")
    ax.scatter(*sim_mm, color="crimson", s=80, marker="*", zorder=5, label=f"Simulated (rank {result.sim_candidate_rank})")

    offset_mm = result.offset_vector_box * 1000.0
    ax.quiver(*sim_mm, *offset_mm, color="gray", arrow_length_ratio=0.2, linewidth=1.5,
              label=f"Offset {np.linalg.norm(offset_mm):.2f} mm")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Measured vs. Simulated Trap\n"
        f"Offset: {np.linalg.norm(offset_mm):.2f} mm  |  "
        f"Mahalanobis: {result.mahalanobis_distance:.2f}  |  "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_3d.png", dpi=150)
    plt.close(fig)

    # ── 2D slice plots ────────────────────────────────────────────────────────
    planes = [
        ("XY", 0, 1, "X (mm)", "Y (mm)"),
        ("XZ", 0, 2, "X (mm)", "Z (mm)"),
        ("YZ", 1, 2, "Y (mm)", "Z (mm)"),
    ]
    for plane_name, ix, iy, xlabel, ylabel in planes:
        fig, ax = plt.subplots(figsize=(6, 5))
        cov2d = np.array([
            [cov_mm2[ix, ix], cov_mm2[ix, iy]],
            [cov_mm2[iy, ix], cov_mm2[iy, iy]],
        ])
        # Draw 2σ error ellipse.
        eigvals2, eigvecs2 = np.linalg.eigh(cov2d)
        eigvals2 = np.maximum(eigvals2, 0.0)
        t = np.linspace(0, 2 * np.pi, 200)
        ell = (eigvecs2 * (2.0 * np.sqrt(eigvals2))[None, :]) @ np.array([np.cos(t), np.sin(t)])
        ax.fill(ell[0] + meas_mm[ix], ell[1] + meas_mm[iy], alpha=0.2, color="steelblue")
        ax.plot(ell[0] + meas_mm[ix], ell[1] + meas_mm[iy], color="steelblue", lw=1.5)

        ax.scatter(meas_mm[ix], meas_mm[iy], s=60, color="steelblue", zorder=5, label="Measured")
        ax.scatter(sim_mm[ix], sim_mm[iy], s=80, color="crimson", marker="*", zorder=5, label="Simulated")
        ax.annotate(
            f"  {offset_mm[ix]:+.2f}, {offset_mm[iy]:+.2f} mm",
            xy=(sim_mm[ix], sim_mm[iy]), fontsize=8, color="gray",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{plane_name} projection — {'PASS' if result.passed else 'FAIL'}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", "datalim")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"comparison_{plane_name.lower()}.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {output_dir}")


# ── Session comparison ────────────────────────────────────────────────────────

def compare_session(
    session_dir: Path,
    sim_output_path: Path,
    box_config_path: Path,
    output_dir: Path,
    threshold_mm: float = 2.0,
    sim_rank: int = 1,
) -> ComparisonResult:
    box_cfg = load_box_config(box_config_path)

    tri_data = load_json(session_dir / "triangulation.json")
    measured_pos = np.array(tri_data["position_box_m"])

    # Use error budget covariance if available; fall back to triangulation.
    error_budget_path = session_dir / "error_budget.json"
    if error_budget_path.exists():
        budget_data = load_json(error_budget_path)
        measured_cov = np.array(budget_data["total_covariance_m2"])
    else:
        measured_cov = np.array(tri_data["covariance_box_m2"])

    result = compare(
        measured_pos, measured_cov, sim_output_path, box_cfg,
        threshold_mm=threshold_mm, sim_rank=sim_rank,
    )

    plot_comparison(result, output_dir)

    result_json = {
        "measured_position_box_mm": (result.measured_position_box * 1000).tolist(),
        "simulated_position_box_mm": (result.simulated_position_box * 1000).tolist(),
        "simulated_position_sim_mm": (result.simulated_position_sim * 1000).tolist(),
        "offset_mm": (result.offset_vector_box * 1000).tolist(),
        "euclidean_offset_mm": float(np.linalg.norm(result.offset_vector_box * 1000)),
        "mahalanobis_distance": result.mahalanobis_distance,
        "chi2_dof": result.chi2_dof,
        "passed": result.passed,
        "threshold_mm": result.threshold_mm,
        "sim_candidate_rank": result.sim_candidate_rank,
    }
    save_json(result_json, output_dir / "comparison_result.json")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare measured vs. simulated trap position")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--sim-output", type=Path, required=True,
                   help="Path to summary.json or final_candidates_*.csv from sim.py")
    p.add_argument("--box-config", type=Path, default=Path("config/box.yaml"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--threshold-mm", type=float, default=2.0)
    p.add_argument("--sim-rank", type=int, default=1,
                   help="Which sim.py candidate rank to compare against (1 = best)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or (args.session / "comparison")
    compare_session(
        session_dir=args.session,
        sim_output_path=args.sim_output,
        box_config_path=args.box_config,
        output_dir=output,
        threshold_mm=args.threshold_mm,
        sim_rank=args.sim_rank,
    )


if __name__ == "__main__":
    main()
