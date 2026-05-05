"""
Hardware-matched reflector trap simulator for a 5x5 ultrasonic phased array.

This copy keeps the real 5x5 hardware assumptions from verilog_generator.py,
but separates two ideas that were coupled in the original geometric simulator:

1. phase_focus_coord: the high-pressure point used only to set phases.
2. trap search volume: the 3D region where direct + reflected fields are
   evaluated for Gor'kov-potential well candidates.

The default phase focus is on the reflector plane, matching the paper-style
reflector focus description. The trap search is not constrained to a single
pressure-node plane; by default it scans the whole analysis grid below the
reflector and reports naturally formed low-pressure/stable-well candidates.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def _tqdm(it, **kw):  # type: ignore[misc]
        return it

from scipy.ndimage import minimum_filter

# Multiprocessing already fans out across CPU cores. Keep low-level numerical
# libraries single-threaded per process to avoid CPU oversubscription.
for _thread_env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import multiprocessing as _mp

print("INFO::LIB::TRYING_IMPORT_NUMBA_&_CUPY")
try:
    from numba import njit as _njit, prange as _prange
    import numba as _numba
    _HAS_NUMBA = True
    print("INFO::LIB::NUMBA_IMPORTED")
except ImportError:
    _HAS_NUMBA = False
    _numba = None
    print("WARN::LIB::NUMBA_IMPORT_FAILED")

try:
    import cupy as _cp      
    _HAS_CUPY = True
    print("INFO::LIB::CUPY_IMPORTED")
except ImportError:
    _HAS_CUPY = False
    _cp = None
    print("WARN::LIB::CUPY_IMPORT_FAILED")

# ============================================================
# Hardware and physical constants
# ============================================================

FREQUENCY_HZ = 40_000.0
SPEED_OF_SOUND = 343.0
WAVELENGTH = SPEED_OF_SOUND / FREQUENCY_HZ

PITCH = 0.011
CLK_HZ = 50_000_000
PERIOD_TICKS = int(round(CLK_HZ / FREQUENCY_HZ))
HALF_PERIOD_TICKS = PERIOD_TICKS // 2

if PERIOD_TICKS != 1250 or HALF_PERIOD_TICKS != 625:
    raise RuntimeError(f"Unexpected tick constants: {PERIOD_TICKS=} {HALF_PERIOD_TICKS=}")

TX_LAYOUT_1BASED = np.array(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ],
    dtype=int,
)

AMP_TABLE_VALUES = np.array(
    [
        100.000, 99.338, 98.681, 98.028, 97.380, 96.736, 96.096, 95.460,
        94.829, 94.201, 93.578, 92.959, 92.344, 91.733, 91.126, 90.479,
        89.994, 89.448, 88.899, 88.217, 87.277, 86.103, 84.774, 83.199,
        81.679, 80.375, 79.296, 78.233, 77.060, 75.761, 74.229, 72.472,
        70.805, 69.498, 68.265, 67.145, 66.168, 65.251, 64.374, 63.526,
        62.700, 61.767, 60.612, 59.575, 58.548, 57.895, 56.866, 55.593,
        53.833, 52.239, 50.215, 48.018, 45.531, 43.319, 40.942, 38.816,
        36.898, 35.548, 34.169, 32.826, 31.305, 29.995, 28.895, 28.101,
        27.647, 27.462, 27.509, 27.707, 27.760, 27.727, 27.509, 27.101,
        26.618, 26.202, 25.804, 25.368, 24.589, 23.650, 22.445, 21.098,
        19.806, 18.786, 18.404, 17.611, 16.567, 15.568, 13.726, 12.284,
        11.185, 10.404, 10.000,
    ],
    dtype=np.float64,
)

# LUT indexed by cos_theta in [0,1] — avoids arccos + rad2deg per chunk at runtime.
_DCOS_N = 9001
_dcos_samples = np.linspace(0.0, 1.0, _DCOS_N)
_dcos_theta = np.rad2deg(np.arccos(_dcos_samples))
_dcos_low = np.floor(_dcos_theta).astype(np.int64)
_dcos_high = np.clip(_dcos_low + 1, 0, len(AMP_TABLE_VALUES) - 1)
_dcos_frac = _dcos_theta - _dcos_low.astype(np.float64)
_DIRECTIVITY_COS_LUT: np.ndarray = (
    AMP_TABLE_VALUES[_dcos_low] + _dcos_frac * (AMP_TABLE_VALUES[_dcos_high] - AMP_TABLE_VALUES[_dcos_low])
)
del _dcos_samples, _dcos_theta, _dcos_low, _dcos_high, _dcos_frac


# ============================================================
# Project paths and defaults
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "simulation_outputs" / "hardware_trap_runs"
MODULE_NAME = "hardware_trap"

# ----------------------------
# User-editable experiment inputs.
# ----------------------------
# Reflector plane location used by the paper-style model.
REFLECTOR_Z = 0.045

# The high-pressure geometric point used only to set transducer phases.
# The default is the reflector plane, following the reflector-focus paper setup.
PHASE_FOCUS = (0.0, 0.0, REFLECTOR_Z)

# Optional center for limited trap searches. Set to None to scan the full grid.
SEARCH_CENTER: tuple[float, float, float] | None = None
SEARCH_RADIUS: float | None = None

# 3D analysis grid size.
# Larger = slower but more detailed. Keep an odd default so x=y=0 is sampled.
GRID_SIZE = 203 
SELECTED_FRACTION = 0.50
DEPTH_RADIUS_CELLS = 2
LOCAL_P_RADIUS = WAVELENGTH / 2.0
LOCAL_P_REFERENCE_PERCENTILE = 95.0
LOCAL_P_MIN_SAMPLES = 100
LOCAL_REFINE_RADIUS = WAVELENGTH / 8.0
LOCAL_REFINE_GRID_SIZE = 101 
CPU_COUNT = max(1, os.cpu_count() or 1)
if _HAS_NUMBA:
    _numba.set_num_threads(CPU_COUNT)
# Use all logical CPU cores by default. Lower these if memory pressure becomes
# visible during the 101^3 local-refine boxes.
FIELD_WORKERS = CPU_COUNT
LOCAL_REFINE_WORKERS = CPU_COUNT
PHYSICAL_TIE_RTOL = 1e-7
PHYSICAL_TIE_ATOL = 1e-12

# Exclude the transducer near-field singular layer from reported analyses.
# The current point-source model is not reliable arbitrarily close to z=0.
MIN_Z = 0.002


# ============================================================
# Data classes
# ============================================================


@dataclass
class SimulationConfig:
    phase_focus_coord: tuple[float, float, float] = PHASE_FOCUS
    search_center: tuple[float, float, float] | None = SEARCH_CENTER
    reflector_z: float = REFLECTOR_Z
    reflection_coeff: float = 1.0
    reflection_phase_rad: float = 0.0
    analysis_grid_size: int = GRID_SIZE
    local_derivative_step: float = 5e-4
    selected_candidate_fraction: float = SELECTED_FRACTION
    well_search_radius: float | None = SEARCH_RADIUS
    x_extent: tuple[float, float] = (-0.050, 0.050)
    y_extent: tuple[float, float] = (-0.050, 0.050)
    z_extent: tuple[float, float] = (MIN_Z, 0.044)
    chunk_points: int = 500_000
    field_workers: int = FIELD_WORKERS
    local_refine_workers: int = LOCAL_REFINE_WORKERS
    out_dir: Path = OUTPUT_ROOT / "attempt_001"
    module_name: str = MODULE_NAME


@dataclass
class FieldGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    dx: float
    dy: float
    dz: float


@dataclass
class SimulationDesign:
    requested_phase_focus: tuple[float, float, float]
    phase_focus_point: tuple[float, float, float]
    search_center: tuple[float, float, float]
    phase_focus_minus_reflector_z: float


# ============================================================
# Gor'kov potential model
# ============================================================


@dataclass
class GorkovPotentialConfig:
    rho0: float = 1.225
    c0: float = SPEED_OF_SOUND
    rho_particle: float = 100.0
    c_particle: float = 2400.0
    particle_radius: float = 1.3e-3 / 2.0
    frequency_hz: float = FREQUENCY_HZ
    amplitude_convention: str = "complex_amplitude_consistent_scale"
    velocity_model: str = "pressure_gradient_over_omega_rho0"


GORKOV_POTENTIAL_CONFIG = GorkovPotentialConfig()


def gorkov_potential_config_dict(cfg: GorkovPotentialConfig) -> dict[str, float | str]:
    return asdict(cfg)


def gorkov_coefficients(cfg: GorkovPotentialConfig) -> tuple[float, float]:
    volume = 4.0 / 3.0 * math.pi * cfg.particle_radius**3
    pressure_coeff = 0.25 * volume * (
        1.0 / (cfg.c0**2 * cfg.rho0)
        - 1.0 / (cfg.c_particle**2 * cfg.rho_particle)
    )
    # Positive for particles/droplets denser than air, so the velocity term is
    # subtracted in U = pressure_coeff * |p|^2 - velocity_coeff * |v|^2.
    velocity_coeff = 0.75 * volume * cfg.rho0 * (
        (cfg.rho_particle - cfg.rho0) / (2.0 * cfg.rho_particle + cfg.rho0)
    )
    return pressure_coeff, velocity_coeff


def velocity_sq_from_pressure_gradient(
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    dpdz: np.ndarray,
    cfg: GorkovPotentialConfig,
) -> np.ndarray:
    omega = 2.0 * math.pi * cfg.frequency_hz
    grad_p_sq = np.abs(dpdx) ** 2 + np.abs(dpdy) ** 2 + np.abs(dpdz) ** 2
    return grad_p_sq / (omega**2 * cfg.rho0**2)


# ============================================================
# Trap detection policy
# ============================================================


@dataclass
class TrapDetectionConfig:
    # Primary trap condition:
    #   1. U_G is a 3D local minimum on the 26-neighbor grid stencil.
    #   2. lambda_min(H_U) is positive, so the Gor'kov potential is stable
    #      along x, y, and z.
    criterion_name: str = "stable_3d_gorkov_potential_minimum"
    stable_min_lambda: float = 0.0
    depth_radius_cells: int = DEPTH_RADIUS_CELLS
    local_p_radius: float = LOCAL_P_RADIUS
    local_p_reference_percentile: float = LOCAL_P_REFERENCE_PERCENTILE
    local_p_min_samples: int = LOCAL_P_MIN_SAMPLES
    local_refine_radius: float = LOCAL_REFINE_RADIUS
    local_refine_grid_size: int = LOCAL_REFINE_GRID_SIZE
    local_refine_workers: int = LOCAL_REFINE_WORKERS
    physical_tie_rtol: float = PHYSICAL_TIE_RTOL
    physical_tie_atol: float = PHYSICAL_TIE_ATOL

def trap_detection_config_dict(cfg: TrapDetectionConfig) -> dict[str, Any]:
    return asdict(cfg)


def _analysis_grid(analysis: dict[str, Any]) -> FieldGrid:
    grid = analysis["grid"]
    assert isinstance(grid, FieldGrid)
    return grid


def local_minima_mask(values: np.ndarray) -> np.ndarray:
    """Return 3D grid points no larger than all 26 neighboring samples."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    safe = np.where(finite, values, np.inf)
    local_min = minimum_filter(safe, size=3, mode="constant", cval=np.inf)
    mask = finite & (values <= local_min)
    mask[[0, -1], :, :] = False
    mask[:, [0, -1], :] = False
    mask[:, :, [0, -1]] = False
    return mask


def local_shell_depth(values: np.ndarray, ix: int, iy: int, iz: int, radius_cells: int) -> float:
    """Estimate escape depth from the candidate to a local Chebyshev shell."""
    values = np.asarray(values)
    radius = max(1, int(radius_cells))
    center = float(values[ix, iy, iz])
    if not math.isfinite(center):
        return float("nan")

    x0 = max(ix - radius, 0)
    x1 = min(ix + radius + 1, values.shape[0])
    y0 = max(iy - radius, 0)
    y1 = min(iy + radius + 1, values.shape[1])
    z0 = max(iz - radius, 0)
    z1 = min(iz + radius + 1, values.shape[2])
    local = values[x0:x1, y0:y1, z0:z1]

    gx = np.arange(x0, x1)[:, None, None] - ix
    gy = np.arange(y0, y1)[None, :, None] - iy
    gz = np.arange(z0, z1)[None, None, :] - iz
    shell = np.maximum(np.maximum(np.abs(gx), np.abs(gy)), np.abs(gz)) == radius
    shell_values = local[shell & np.isfinite(local)]
    if shell_values.size == 0:
        return float("nan")
    return float(np.min(shell_values) - center)


def local_pressure_ratio(
    p_abs: np.ndarray,
    grid: FieldGrid,
    ix: int,
    iy: int,
    iz: int,
    cfg: TrapDetectionConfig,
) -> tuple[float, float, int]:
    """Compare candidate pressure to the local standing-wave pressure envelope."""
    p_abs = np.asarray(p_abs)
    radius = float(cfg.local_p_radius)
    if radius <= 0.0:
        return float("nan"), float("nan"), 0

    x = float(grid.x[ix])
    y = float(grid.y[iy])
    z = float(grid.z[iz])
    x0 = max(int(np.searchsorted(grid.x, x - radius, side="left")), 0)
    x1 = min(int(np.searchsorted(grid.x, x + radius, side="right")), len(grid.x))
    y0 = max(int(np.searchsorted(grid.y, y - radius, side="left")), 0)
    y1 = min(int(np.searchsorted(grid.y, y + radius, side="right")), len(grid.y))
    z0 = max(int(np.searchsorted(grid.z, z - radius, side="left")), 0)
    z1 = min(int(np.searchsorted(grid.z, z + radius, side="right")), len(grid.z))

    local = p_abs[x0:x1, y0:y1, z0:z1]
    dx = grid.x[x0:x1][:, None, None] - x
    dy = grid.y[y0:y1][None, :, None] - y
    dz = grid.z[z0:z1][None, None, :] - z
    sphere = (dx * dx + dy * dy + dz * dz) <= radius * radius
    samples = local[sphere & np.isfinite(local)]
    sample_count = int(samples.size)
    if sample_count < int(cfg.local_p_min_samples):
        return float("nan"), float("nan"), sample_count

    percentile = min(max(float(cfg.local_p_reference_percentile), 0.0), 100.0)
    ref = float(np.percentile(samples, percentile))
    candidate_p = float(p_abs[ix, iy, iz])
    if not math.isfinite(ref) or ref <= 0.0 or not math.isfinite(candidate_p):
        return float("nan"), ref, sample_count
    return float(candidate_p / ref), ref, sample_count


def search_mask_and_mode(
    analysis: dict[str, Any],
    search_center: tuple[float, float, float],
    search_radius: float | None,
) -> tuple[np.ndarray, str]:
    grid = _analysis_grid(analysis)
    U = np.asarray(analysis["U"])

    sx, sy, sz = search_center
    dist_to_search_center = np.sqrt((grid.X - sx) ** 2 + (grid.Y - sy) ** 2 + (grid.Z - sz) ** 2)
    if search_radius is None:
        return np.isfinite(U), "full_grid"
    return dist_to_search_center <= float(search_radius), "radius_limited"


def _candidate_rows_from_mask(
    analysis: dict[str, Any],
    candidate_mask: np.ndarray,
    search_mode: str,
    cfg: TrapDetectionConfig,
) -> list[dict[str, float | int | bool | str]]:
    grid = _analysis_grid(analysis)
    p_abs = np.asarray(analysis["p_abs"])
    U = np.asarray(analysis["U"])
    grad_norm = np.asarray(analysis["grad_norm"])
    lambda_min = np.asarray(analysis["lambda_min_conf"])

    flat_candidates = np.flatnonzero(candidate_mask)
    if flat_candidates.size == 0:
        return []

    rows: list[dict[str, float | int | bool | str]] = []
    for rank, flat_idx in enumerate(flat_candidates, start=1):
        ix, iy, iz = np.unravel_index(int(flat_idx), p_abs.shape)
        x = float(grid.x[ix])
        y = float(grid.y[iy])
        z = float(grid.z[iz])
        depth_value = local_shell_depth(U, int(ix), int(iy), int(iz), cfg.depth_radius_cells)
        local_p_ratio_value, local_p_ref_value, local_p_sample_count = local_pressure_ratio(p_abs, grid, int(ix), int(iy), int(iz), cfg)
        lambda_value = float(lambda_min[ix, iy, iz])
        rows.append(
            {
                "rank": rank,
                "search_mode": search_mode,
                "x": x,
                "y": y,
                "z": z,
                "p_abs": float(p_abs[ix, iy, iz]),
                "local_p_ratio": local_p_ratio_value,
                "local_p_ref": local_p_ref_value,
                "local_p_sample_count": local_p_sample_count,
                "U": float(U[ix, iy, iz]),
                "grad_norm": float(grad_norm[ix, iy, iz]),
                "well_depth": depth_value,
                "lambda_min_conf": lambda_value,
            }
        )
    return rows


def find_primary_trap_candidates(
    analysis: dict[str, Any],
    search_center: tuple[float, float, float],
    search_radius: float | None,
    cfg: TrapDetectionConfig | None = None,
) -> list[dict[str, float | int | bool | str]]:
    cfg = cfg or TrapDetectionConfig()
    U = np.asarray(analysis["U"])
    lambda_min = np.asarray(analysis["lambda_min_conf"])
    search_mask, search_mode = search_mask_and_mode(analysis, search_center, search_radius)

    candidate_mask = local_minima_mask(U) & search_mask & (lambda_min > cfg.stable_min_lambda)
    return _candidate_rows_from_mask(
        analysis,
        candidate_mask,
        search_mode,
        cfg,
    )


FILTER_STAGE_ORDER = (
    "primary",
    "primary+p",
    "primary+p+lambda_min",
    "primary+p+lambda_min+depth",
)
FINAL_FILTER_STAGE = FILTER_STAGE_ORDER[-1]
PHYSICAL_TIE_KEYS = (
    "p_abs",
    "local_p_ratio",
    "local_p_ref",
    "U",
    "grad_norm",
    "well_depth",
    "lambda_min_conf",
)


def _ranked_rows(
    rows: list[dict[str, float | int | bool | str]],
) -> list[dict[str, float | int | bool | str]]:
    ranked: list[dict[str, float | int | bool | str]] = []
    for rank, row in enumerate(rows, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = rank
        ranked.append(ranked_row)
    return ranked


def _finite_values(rows: list[dict[str, float | int | bool | str]], key: str) -> np.ndarray:
    values = [float(row.get(key, float("nan"))) for row in rows]
    return np.array([value for value in values if math.isfinite(value)], dtype=np.float64)


def _same_physical_tie(
    row: dict[str, float | int | bool | str],
    reference: dict[str, float | int | bool | str],
    rtol: float,
    atol: float,
) -> bool:
    for key in PHYSICAL_TIE_KEYS:
        a = float(row.get(key, float("nan")))
        b = float(reference.get(key, float("nan")))
        if math.isfinite(a) and math.isfinite(b):
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                return False
        elif math.isfinite(a) != math.isfinite(b):
            return False
    return True


def _add_boundary_physical_ties(
    rows: list[dict[str, float | int | bool | str]],
    survivors: list[dict[str, float | int | bool | str]],
    boundary_row: dict[str, float | int | bool | str] | None,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, float | int | bool | str]], int]:
    if boundary_row is None:
        return survivors, 0

    survivor_ids = {id(row) for row in survivors}
    tied_rows = [
        row
        for row in rows
        if id(row) not in survivor_ids and _same_physical_tie(row, boundary_row, rtol, atol)
    ]
    if not tied_rows:
        return survivors, 0
    return survivors + tied_rows, len(tied_rows)


def _fraction_filter(
    rows: list[dict[str, float | int | bool | str]],
    stage_name: str,
    key: str,
    keep: str,
    fraction: float,
    cfg: TrapDetectionConfig,
) -> tuple[list[dict[str, float | int | bool | str]], dict[str, float | int | str]]:
    if not rows:
        return [], {
            "stage": stage_name,
            "key": key,
            "mode": keep,
            "input_count": 0,
            "output_count": 0,
            "cutoff": float("nan"),
        }

    q = min(max(float(fraction), 0.0), 1.0)
    if q <= 0.0:
        return [], {
            "stage": stage_name,
            "key": key,
            "mode": keep,
            "input_count": len(rows),
            "output_count": 0,
            "cutoff": float("nan"),
        }

    finite = _finite_values(rows, key)
    if finite.size == 0:
        return [], {
            "stage": stage_name,
            "key": key,
            "mode": keep,
            "input_count": len(rows),
            "output_count": 0,
            "cutoff": float("nan"),
        }

    target_count = max(1, int(math.ceil(len(rows) * q)))
    finite_rows = [row for row in rows if math.isfinite(float(row.get(key, float("nan"))))]
    reverse = keep == "highest"
    finite_rows.sort(key=lambda row: float(row.get(key, float("nan"))), reverse=reverse)
    boundary_row = finite_rows[min(target_count - 1, len(finite_rows) - 1)] if finite_rows else None
    sorted_values = np.sort(finite)
    if keep == "lowest":
        cutoff = float(sorted_values[min(target_count - 1, sorted_values.size - 1)])
        survivors = [row for row in rows if float(row.get(key, float("nan"))) <= cutoff]
        survivors.sort(key=lambda row: float(row.get(key, float("inf"))))
    elif keep == "highest":
        cutoff = float(sorted_values[max(sorted_values.size - target_count, 0)])
        survivors = [row for row in rows if float(row.get(key, float("nan"))) >= cutoff]
        survivors.sort(key=lambda row: float(row.get(key, float("-inf"))), reverse=True)
    else:
        raise ValueError(f"Unknown filter direction: {keep}")

    survivors, added_ties = _add_boundary_physical_ties(
        rows,
        survivors,
        boundary_row,
        cfg.physical_tie_rtol,
        cfg.physical_tie_atol,
    )
    survivors.sort(key=lambda row: float(row.get(key, float("inf" if keep == "lowest" else "-inf"))), reverse=reverse)
    ranked = _ranked_rows(survivors)
    return ranked, {
        "stage": stage_name,
        "key": key,
        "mode": keep,
        "fraction": q,
        "input_count": len(rows),
        "target_count": target_count,
        "physical_tie_count": added_ties,
        "output_count": len(ranked),
        "cutoff": cutoff,
    }


def selected_candidates_by_filters(
    primary_rows: list[dict[str, float | int | bool | str]],
    fraction: float,
    cfg: TrapDetectionConfig | None = None,
) -> tuple[dict[str, list[dict[str, float | int | bool | str]]], dict[str, Any]]:
    cfg = cfg or TrapDetectionConfig()
    stages: dict[str, list[dict[str, float | int | bool | str]]] = {
        "primary": _ranked_rows(primary_rows),
    }
    stage_info: list[dict[str, float | int | str]] = []

    rows, info = _fraction_filter(stages["primary"], "primary+p", "local_p_ratio", "lowest", fraction, cfg)
    stages["primary+p"] = rows
    stage_info.append(info)

    rows, info = _fraction_filter(rows, "primary+p+lambda_min", "lambda_min_conf", "highest", fraction, cfg)
    stages["primary+p+lambda_min"] = rows
    stage_info.append(info)

    rows, info = _fraction_filter(rows, "primary+p+lambda_min+depth", "well_depth", "highest", fraction, cfg)
    stages["primary+p+lambda_min+depth"] = rows
    stage_info.append(info)

    return stages, {
        "selection_method": "staged_physical_filters",
        "filter_fraction": min(max(float(fraction), 0.0), 1.0),
        "depth_radius_cells": int(cfg.depth_radius_cells),
        "local_p_radius": float(cfg.local_p_radius),
        "local_p_reference_percentile": min(max(float(cfg.local_p_reference_percentile), 0.0), 100.0),
        "local_p_min_samples": int(cfg.local_p_min_samples),
        "local_refine_radius": float(cfg.local_refine_radius),
        "local_refine_grid_size": int(ensure_odd(cfg.local_refine_grid_size)),
        "local_refine_workers": int(cfg.local_refine_workers),
        "physical_tie_rtol": float(cfg.physical_tie_rtol),
        "physical_tie_atol": float(cfg.physical_tie_atol),
        "total_primary_count": len(primary_rows),
        "selected_count": len(stages[FINAL_FILTER_STAGE]),
        "stages": stage_info,
    }


def axis_scan_rows(
    analysis: dict[str, Any],
    axis_xy: tuple[float, float],
) -> list[dict[str, float | int | bool | str]]:
    grid = _analysis_grid(analysis)
    p_abs = np.asarray(analysis["p_abs"])
    U = np.asarray(analysis["U"])
    grad_norm = np.asarray(analysis["grad_norm"])
    lambda_min = np.asarray(analysis["lambda_min_conf"])

    ax, ay = axis_xy
    ix = int(np.argmin(np.abs(grid.x - ax)))
    iy = int(np.argmin(np.abs(grid.y - ay)))
    U_line = U[ix, iy, :]

    rows: list[dict[str, float | int | bool | str]] = []
    for iz, z in enumerate(grid.z):
        is_edge = iz == 0 or iz == len(grid.z) - 1
        U_min_1d = False if is_edge else bool(U_line[iz] <= U_line[iz - 1] and U_line[iz] <= U_line[iz + 1])
        lambda_value = float(lambda_min[ix, iy, iz])
        rows.append(
            {
                "index_z": int(iz),
                "x": float(grid.x[ix]),
                "y": float(grid.y[iy]),
                "z": float(z),
                "p_abs": float(p_abs[ix, iy, iz]),
                "U": float(U[ix, iy, iz]),
                "grad_norm": float(grad_norm[ix, iy, iz]),
                "lambda_min_conf": lambda_value,
                "potential_minimum_1d": U_min_1d,
            }
        )
    return rows


# ============================================================
# Geometry
# ============================================================


def build_transducer_positions() -> np.ndarray:
    idx = np.arange(-2, 3, dtype=np.float64)
    x_coords = idx * PITCH
    y_coords_top_to_bottom = idx[::-1] * PITCH

    positions = np.zeros((25, 3), dtype=np.float64)
    for r in range(5):
        for c in range(5):
            tx_num = TX_LAYOUT_1BASED[r, c]
            positions[tx_num - 1] = [x_coords[c], y_coords_top_to_bottom[r], 0.0]
    return positions


TRANSDUCER_POSITIONS = build_transducer_positions()
TRANSDUCER_NORMALS = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (25, 1))


# ============================================================
# Utility
# ============================================================


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_2pi(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) % (2.0 * np.pi)


def phase_from_ticks(ticks: np.ndarray, period_ticks: int = PERIOD_TICKS) -> np.ndarray:
    return wrap_to_pi((ticks.astype(np.float64) / float(period_ticks)) * 2.0 * np.pi)


def directivity_from_theta(theta_deg: np.ndarray) -> np.ndarray:
    theta = np.clip(theta_deg, 0.0, 90.0)
    low = np.floor(theta).astype(np.int64)
    high = np.clip(low + 1, 0, len(AMP_TABLE_VALUES) - 1)
    frac = theta - low.astype(np.float64)
    return AMP_TABLE_VALUES[low] + frac * (AMP_TABLE_VALUES[high] - AMP_TABLE_VALUES[low])


def _directivity_from_cos(cos_theta: np.ndarray) -> np.ndarray:
    """Directivity from cos_theta in [-1,1]. Values where cos_theta<=0 map to theta>=90 deg (LUT endpoint)."""
    idx = np.clip(
        np.round(np.clip(cos_theta, 0.0, 1.0) * float(_DCOS_N - 1)).astype(np.int64),
        0,
        _DCOS_N - 1,
    )
    return _DIRECTIVITY_COS_LUT[idx]


def create_attempt_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = root / f"attempt_{idx:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def make_simulation_design(cfg: SimulationConfig) -> SimulationDesign:
    fx, fy, fz = cfg.phase_focus_coord

    if fz <= 0.0:
        raise ValueError("phase_focus_z must be positive.")
    if fz > cfg.reflector_z:
        raise ValueError("phase_focus_z must be on or below reflector_z for the reflector model.")

    if cfg.search_center is None:
        # This marker is only for distance reporting and plots; the default
        # search still covers the full grid when well_search_radius is None.
        search_z = min(fz, cfg.reflector_z - WAVELENGTH / 4.0)
        search_center = (fx, fy, max(search_z, cfg.z_extent[0]))
    else:
        sx, sy, sz = cfg.search_center
        if sz <= 0.0 or sz >= cfg.reflector_z:
            raise ValueError("search_center_z must be inside the array-reflector gap.")
        search_center = (float(sx), float(sy), float(sz))

    return SimulationDesign(
        requested_phase_focus=(fx, fy, fz),
        phase_focus_point=(fx, fy, fz),
        search_center=search_center,
        phase_focus_minus_reflector_z=fz - cfg.reflector_z,
    )


def geometric_focus_phases(focus_point: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(focus_point, dtype=np.float64)
    distances = np.linalg.norm(target[None, :] - TRANSDUCER_POSITIONS, axis=1)
    k_val = 2.0 * math.pi / WAVELENGTH
    phases = -k_val * distances
    phases = wrap_to_pi(phases - phases[0])
    return phases, distances


def quantize_phases_to_ticks(phases_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ticks = np.mod(np.rint(phases_rad / (2.0 * np.pi) * PERIOD_TICKS).astype(np.int64), PERIOD_TICKS)
    quantized = phase_from_ticks(ticks, PERIOD_TICKS)
    return ticks.astype(np.int64), quantized


def ensure_odd(n: int) -> int:
    n = int(n)
    return n if (n % 2 == 1) else (n + 1)


def _resolve_worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    requested = int(requested_workers)
    if requested <= 0:
        requested = CPU_COUNT
    return min(max(1, requested), int(task_count))


def _split_even_ranges(length: int, part_count: int) -> list[tuple[int, int]]:
    part_count = _resolve_worker_count(part_count, length)
    edges = np.linspace(0, int(length), part_count + 1, dtype=np.int64)
    return [
        (int(edges[i]), int(edges[i + 1]))
        for i in range(part_count)
        if int(edges[i]) < int(edges[i + 1])
    ]


# ============================================================
# Acoustic field model
# ============================================================

_K_VAL = 2.0 * math.pi / WAVELENGTH


def _in_worker_process() -> bool:
    return _mp.current_process().name != "MainProcess"


def _use_gpu() -> bool:
    return _HAS_CUPY and not _in_worker_process()


def _use_numba() -> bool:
    return _HAS_NUMBA and not _use_gpu()


if _HAS_NUMBA:
    @_njit(parallel=True, fastmath=True, cache=True)
    def _nb_field_kernel(
        points: np.ndarray,
        src_pos: np.ndarray,
        src_normals: np.ndarray,
        src_phases: np.ndarray,
        src_scales: np.ndarray,
        k_val: float,
        dcos_lut: np.ndarray,
        dcos_n: int,
    ) -> np.ndarray:
        n_pts = points.shape[0]
        n_src = src_pos.shape[0]
        result = np.empty(n_pts, dtype=np.complex128)
        for i in _prange(n_pts):
            ar = 0.0
            ai = 0.0
            px = points[i, 0]
            py = points[i, 1]
            pz = points[i, 2]
            for j in range(n_src):
                dx = px - src_pos[j, 0]
                dy = py - src_pos[j, 1]
                dz = pz - src_pos[j, 2]
                R2 = dx * dx + dy * dy + dz * dz
                if R2 < 1e-18:
                    R = 1e-9
                else:
                    R = R2 ** 0.5
                inv_R = 1.0 / R
                cos_t = (dx * src_normals[j, 0] + dy * src_normals[j, 1] + dz * src_normals[j, 2]) * inv_R
                if cos_t <= 0.0:
                    continue
                if cos_t > 1.0:
                    cos_t = 1.0
                lut_idx = int(cos_t * (dcos_n - 1) + 0.5)
                if lut_idx >= dcos_n:
                    lut_idx = dcos_n - 1
                A = dcos_lut[lut_idx] * src_scales[j] * inv_R
                ph = src_phases[j] + k_val * R
                ar += A * math.cos(ph)
                ai += A * math.sin(ph)
            result[i] = complex(ar, ai)
        return result
else:
    _nb_field_kernel = None


def _gpu_field_kernel(
    points: np.ndarray,
    src_pos: np.ndarray,
    src_normals: np.ndarray,
    src_phases: np.ndarray,
    src_scales: np.ndarray,
    k_val: float,
) -> np.ndarray:
    pts = _cp.asarray(points)
    sp = _cp.asarray(src_pos)
    sn = _cp.asarray(src_normals)
    sph = _cp.asarray(src_phases)
    ss = _cp.asarray(src_scales)
    dcos = _cp.asarray(_DIRECTIVITY_COS_LUT)

    delta = pts[:, None, :] - sp[None, :, :]
    R = _cp.linalg.norm(delta, axis=2)
    R = _cp.maximum(R, 1e-9)
    cos_theta = _cp.clip(_cp.einsum("ijk,jk->ij", delta / R[:, :, None], sn), -1.0, 1.0)
    idx = _cp.clip(
        _cp.round(_cp.clip(cos_theta, 0.0, 1.0) * (_DCOS_N - 1)).astype(_cp.int64),
        0, _DCOS_N - 1,
    )
    A_theta = dcos[idx]
    amplitude = ss[None, :] * A_theta / R
    amplitude = _cp.where(cos_theta > 0.0, amplitude, 0.0)
    total_phase = sph[None, :] + k_val * R
    out = _cp.sum(amplitude * _cp.exp(1j * total_phase), axis=1)
    return _cp.asnumpy(out)


def _eigvalsh3_min_batch(H: np.ndarray) -> np.ndarray:
    """Minimum eigenvalue of batched real symmetric 3x3 matrices, shape (N, 3, 3).
    Uses analytic Cardano formula — avoids per-matrix LAPACK dispatch."""
    a = H[:, 0, 0]; b = H[:, 1, 1]; c = H[:, 2, 2]
    d = H[:, 0, 1]; e = H[:, 1, 2]; f = H[:, 0, 2]

    p1 = d * d + e * e + f * f
    q = (a + b + c) * (1.0 / 3.0)
    aq = a - q; bq = b - q; cq = c - q
    p2 = aq * aq + bq * bq + cq * cq + 2.0 * p1
    p = np.sqrt(np.maximum(p2 * (1.0 / 6.0), 0.0))

    det_A_qI = aq * (bq * cq - e * e) - d * (d * cq - e * f) + f * (d * e - bq * f)
    safe_p3 = np.where(p > 1e-10, p * p * p, 1.0)
    r = np.where(p > 1e-10, det_A_qI / (2.0 * safe_p3), 0.0)
    r = np.clip(r, -1.0, 1.0)

    phi = np.arccos(r) / 3.0
    eig_min = q + 2.0 * p * np.cos(phi + (2.0 * math.pi / 3.0))
    diag_min = np.minimum(np.minimum(a, b), c)
    return np.where(p > 1e-10, eig_min, diag_min)


def _field_contribution(
    points: np.ndarray,
    source_positions: np.ndarray,
    source_normals: np.ndarray,
    source_phases: np.ndarray,
    source_amplitude_scales: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute summed complex pressure at each point from all sources in one pass.

    source_phases: per-source total phase offset (base phase + any extra), shape (N_src,).
    source_amplitude_scales: per-source amplitude multiplier, shape (N_src,).
    Dispatches to GPU (cupy), numba JIT, or numpy based on available backends.
    """
    if _use_gpu():
        return _gpu_field_kernel(points, source_positions, source_normals, source_phases, source_amplitude_scales, _K_VAL)

    if _use_numba():
        return _nb_field_kernel(points, source_positions, source_normals, source_phases, source_amplitude_scales, _K_VAL, _DIRECTIVITY_COS_LUT, _DCOS_N)

    out = np.zeros(points.shape[0], dtype=np.complex128)
    for start in range(0, points.shape[0], cfg.chunk_points):
        stop = min(start + cfg.chunk_points, points.shape[0])
        pts = points[start:stop]
        delta = pts[:, None, :] - source_positions[None, :, :]
        R = np.linalg.norm(delta, axis=2)
        R = np.maximum(R, 1e-9)
        cos_theta = np.clip(
            np.einsum("ijk,jk->ij", delta / R[:, :, None], source_normals), -1.0, 1.0
        )
        forward_mask = cos_theta > 0.0
        A_theta = _directivity_from_cos(cos_theta)
        amplitude = source_amplitude_scales[None, :] * A_theta / R
        amplitude = np.where(forward_mask, amplitude, 0.0)
        total_phase = source_phases[None, :] + _K_VAL * R
        out[start:stop] = np.sum(amplitude * np.exp(1j * total_phase), axis=1)
    return out


def pressure_at_points(points: np.ndarray, phases: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    phases_wrapped = wrap_to_2pi(phases)

    image_positions = TRANSDUCER_POSITIONS.copy()
    image_positions[:, 2] = 2.0 * cfg.reflector_z - image_positions[:, 2]
    image_normals = TRANSDUCER_NORMALS.copy()
    image_normals[:, 2] *= -1.0

    all_positions = np.vstack([TRANSDUCER_POSITIONS, image_positions])
    all_normals = np.vstack([TRANSDUCER_NORMALS, image_normals])
    all_phases = np.concatenate([
        phases_wrapped,
        phases_wrapped + cfg.reflection_phase_rad,
    ])
    all_scales = np.concatenate([
        np.ones(25, dtype=np.float64),
        np.full(25, cfg.reflection_coeff, dtype=np.float64),
    ])

    return _field_contribution(points, all_positions, all_normals, all_phases, all_scales, cfg)


def make_grid(cfg: SimulationConfig) -> FieldGrid:
    n = ensure_odd(cfg.analysis_grid_size)
    z_lo = max(cfg.z_extent[0], MIN_Z)
    z_hi = min(cfg.z_extent[1], cfg.reflector_z - 1e-4)
    if z_hi <= z_lo:
        z_hi = cfg.reflector_z - 1e-4
    x = np.linspace(cfg.x_extent[0], cfg.x_extent[1], n, dtype=np.float64)
    y = np.linspace(cfg.y_extent[0], cfg.y_extent[1], n, dtype=np.float64)
    z = np.linspace(z_lo, z_hi, n, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
    return FieldGrid(x=x, y=y, z=z, X=X, Y=Y, Z=Z, dx=dx, dy=dy, dz=dz)


def pressure_on_grid(phases: np.ndarray, grid: FieldGrid, cfg: SimulationConfig) -> np.ndarray:
    if _use_gpu() or _use_numba():
        workers = 1
    else:
        workers = _resolve_worker_count(cfg.field_workers, len(grid.z))
    if workers <= 1:
        points = np.column_stack([grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel()])
        return pressure_at_points(points, phases, cfg).reshape(grid.X.shape)

    p = np.empty(grid.X.shape, dtype=np.complex128)
    serial_cfg = replace(cfg, field_workers=1)
    ranges = _split_even_ranges(len(grid.z), workers)
    payloads = [(z0, z1, grid.x, grid.y, grid.z, phases, serial_cfg) for z0, z1 in ranges]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        slabs = executor.map(_pressure_grid_slab_worker, payloads)
        for z_start, z_stop, p_slab in _tqdm(slabs, total=len(payloads), desc="    slabs", unit="slab", leave=False):
            p[:, :, z_start:z_stop] = p_slab
    return p


def _pressure_grid_slab_worker(
    payload: tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimulationConfig],
) -> tuple[int, int, np.ndarray]:
    z_start, z_stop, x, y, z, phases, cfg = payload
    z_slab = z[z_start:z_stop]
    X, Y, Z = np.meshgrid(x, y, z_slab, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    p_slab = pressure_at_points(points, phases, cfg).reshape(X.shape)
    return int(z_start), int(z_stop), p_slab


# ============================================================
# Gor'kov diagnostics (analysis only, not optimization)
# ============================================================


def gorkov_potential_from_pressure(p: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    pressure_coeff, velocity_coeff = gorkov_coefficients(GORKOV_POTENTIAL_CONFIG)
    dpdx, dpdy, dpdz = np.gradient(p, dx, dy, dz, edge_order=1)
    velocity_sq = velocity_sq_from_pressure_gradient(dpdx, dpdy, dpdz, GORKOV_POTENTIAL_CONFIG)
    return pressure_coeff * (np.abs(p) ** 2) - velocity_coeff * velocity_sq


def scalar_gradient(U: np.ndarray, dx: float, dy: float, dz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.gradient(U, dx, dy, dz, edge_order=1) # type: ignore


def hessian_matrix(U: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    dUx, dUy, dUz = scalar_gradient(U, dx, dy, dz)
    Hxx, Hxy_a, Hxz_a = np.gradient(dUx, dx, dy, dz, edge_order=1)
    Hyx_a, Hyy, Hyz_a = np.gradient(dUy, dx, dy, dz, edge_order=1)
    Hzx_a, Hzy_a, Hzz = np.gradient(dUz, dx, dy, dz, edge_order=1)

    Hxy = 0.5 * (Hxy_a + Hyx_a)
    Hxz = 0.5 * (Hxz_a + Hzx_a)
    Hyz = 0.5 * (Hyz_a + Hzy_a)

    H = np.empty(U.shape + (3, 3), dtype=np.float64)
    H[..., 0, 0] = Hxx
    H[..., 0, 1] = Hxy
    H[..., 0, 2] = Hxz
    H[..., 1, 0] = Hxy
    H[..., 1, 1] = Hyy
    H[..., 1, 2] = Hyz
    H[..., 2, 0] = Hxz
    H[..., 2, 1] = Hyz
    H[..., 2, 2] = Hzz

    return H


def analyze_field(phases: np.ndarray, cfg: SimulationConfig, _label: str = "field") -> dict[str, np.ndarray | FieldGrid]:
    grid = make_grid(cfg)
    n = ensure_odd(cfg.analysis_grid_size)
    n_pts = n ** 3
    backend = "gpu" if _use_gpu() else ("numba" if _use_numba() else "numpy")
    workers_active = 1 if (_use_gpu() or _use_numba()) else _resolve_worker_count(cfg.field_workers, len(grid.z))

    print(
        f"  [{_label}] pressure  {n}³={n_pts:,} pts × 50 src"
        f"  backend={backend}"
        + (f"/{workers_active}T" if workers_active > 1 else ""),
        flush=True,
    )
    t0 = time.perf_counter()
    p = pressure_on_grid(phases, grid, cfg)
    t_p = time.perf_counter() - t0
    print(f"  [{_label}] pressure  done  {t_p:.2f}s", flush=True)

    print(f"  [{_label}] Gorkov + gradient + Hessian ...", end="", flush=True)
    t1 = time.perf_counter()
    p_abs = np.abs(p)
    U = gorkov_potential_from_pressure(p, grid.dx, grid.dy, grid.dz)
    dUx, dUy, dUz = scalar_gradient(U, grid.dx, grid.dy, grid.dz)
    grad_norm = np.sqrt(dUx**2 + dUy**2 + dUz**2)
    H = hessian_matrix(U, grid.dx, grid.dy, grid.dz)
    print(f"  {time.perf_counter()-t1:.2f}s", flush=True)

    print(f"  [{_label}] eigenvalues (analytic 3×3) ...", end="", flush=True)
    t2 = time.perf_counter()
    lambda_min = _eigvalsh3_min_batch(H.reshape(-1, 3, 3)).reshape(U.shape)
    print(f"  {time.perf_counter()-t2:.2f}s", flush=True)

    print(
        f"  [{_label}] total {time.perf_counter()-t0:.2f}s"
        f"  |  p_max={float(np.max(np.abs(p))):.3e}"
        f"  U_min={float(np.nanmin(U)):.3e}",
        flush=True,
    )
    return {
        "grid": grid,
        "p": p,
        "p_abs": p_abs,
        "U": U,
        "grad_norm": grad_norm,
        "lambda_min_conf": lambda_min,
    }


def local_metrics_at_coord(coord: tuple[float, float, float], phases: np.ndarray, cfg: SimulationConfig) -> dict[str, float]:
    half = 2
    dx = dy = dz = float(cfg.local_derivative_step)
    cx, cy, cz = coord
    x = np.linspace(cx - half * dx, cx + half * dx, 2 * half + 1, dtype=np.float64)
    y = np.linspace(cy - half * dy, cy + half * dy, 2 * half + 1, dtype=np.float64)
    z = np.linspace(cz - half * dz, cz + half * dz, 2 * half + 1, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    p = pressure_at_points(points, phases, cfg).reshape(X.shape)
    U = gorkov_potential_from_pressure(p, dx, dy, dz)
    dUx, dUy, dUz = scalar_gradient(U, dx, dy, dz)
    H = hessian_matrix(U, dx, dy, dz)

    c = (half, half, half)
    Hc = H[c]
    lambda_min = float(np.linalg.eigvalsh(Hc)[0])

    return {
        "x": float(cx),
        "y": float(cy),
        "z": float(cz),
        "p_abs": float(np.abs(p[c])),
        "U": float(U[c]),
        "grad_norm": float(np.sqrt(dUx[c] ** 2 + dUy[c] ** 2 + dUz[c] ** 2)),
        "lambda_min_conf": lambda_min,
    }


def _local_minimum_26_at(U: np.ndarray, ix: int, iy: int, iz: int) -> bool:
    if ix <= 0 or iy <= 0 or iz <= 0:
        return False
    if ix >= U.shape[0] - 1 or iy >= U.shape[1] - 1 or iz >= U.shape[2] - 1:
        return False
    center = float(U[ix, iy, iz])
    if not math.isfinite(center):
        return False
    neighborhood = U[ix - 1 : ix + 2, iy - 1 : iy + 2, iz - 1 : iz + 2]
    return bool(center <= float(np.nanmin(neighborhood)))


def _point_gradient_hessian(
    U: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if ix <= 0 or iy <= 0 or iz <= 0:
        nan_g = np.full(3, np.nan, dtype=np.float64)
        nan_h = np.full((3, 3), np.nan, dtype=np.float64)
        return nan_g, nan_h
    if ix >= U.shape[0] - 1 or iy >= U.shape[1] - 1 or iz >= U.shape[2] - 1:
        nan_g = np.full(3, np.nan, dtype=np.float64)
        nan_h = np.full((3, 3), np.nan, dtype=np.float64)
        return nan_g, nan_h

    center = float(U[ix, iy, iz])
    dUx = (float(U[ix + 1, iy, iz]) - float(U[ix - 1, iy, iz])) / (2.0 * dx)
    dUy = (float(U[ix, iy + 1, iz]) - float(U[ix, iy - 1, iz])) / (2.0 * dy)
    dUz = (float(U[ix, iy, iz + 1]) - float(U[ix, iy, iz - 1])) / (2.0 * dz)

    Hxx = (float(U[ix + 1, iy, iz]) - 2.0 * center + float(U[ix - 1, iy, iz])) / (dx * dx)
    Hyy = (float(U[ix, iy + 1, iz]) - 2.0 * center + float(U[ix, iy - 1, iz])) / (dy * dy)
    Hzz = (float(U[ix, iy, iz + 1]) - 2.0 * center + float(U[ix, iy, iz - 1])) / (dz * dz)
    Hxy = (
        float(U[ix + 1, iy + 1, iz])
        - float(U[ix + 1, iy - 1, iz])
        - float(U[ix - 1, iy + 1, iz])
        + float(U[ix - 1, iy - 1, iz])
    ) / (4.0 * dx * dy)
    Hxz = (
        float(U[ix + 1, iy, iz + 1])
        - float(U[ix + 1, iy, iz - 1])
        - float(U[ix - 1, iy, iz + 1])
        + float(U[ix - 1, iy, iz - 1])
    ) / (4.0 * dx * dz)
    Hyz = (
        float(U[ix, iy + 1, iz + 1])
        - float(U[ix, iy + 1, iz - 1])
        - float(U[ix, iy - 1, iz + 1])
        + float(U[ix, iy - 1, iz - 1])
    ) / (4.0 * dy * dz)

    g = np.array([dUx, dUy, dUz], dtype=np.float64)
    H = np.array(
        [
            [Hxx, Hxy, Hxz],
            [Hxy, Hyy, Hyz],
            [Hxz, Hyz, Hzz],
        ],
        dtype=np.float64,
    )
    return g, H


def refine_candidate_local_minimum(
    candidate: dict[str, float | int | bool | str],
    phases: np.ndarray,
    cfg: SimulationConfig,
    trap_cfg: TrapDetectionConfig,
) -> dict[str, float | int | bool | str]:
    n = ensure_odd(trap_cfg.local_refine_grid_size)
    radius = float(trap_cfg.local_refine_radius)
    cx = float(candidate["x"])
    cy = float(candidate["y"])
    cz = float(candidate["z"])

    x = np.linspace(cx - radius, cx + radius, n, dtype=np.float64)
    y = np.linspace(cy - radius, cy + radius, n, dtype=np.float64)
    z_lo = max(cz - radius, MIN_Z)
    z_hi = min(cz + radius, cfg.reflector_z - 1e-4)
    if z_hi <= z_lo:
        z_lo = max(cz - radius, MIN_Z)
        z_hi = z_lo + max(radius, 1e-6)
    z = np.linspace(z_lo, z_hi, n, dtype=np.float64)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    p = pressure_at_points(points, phases, cfg).reshape(X.shape)

    dx = float(x[1] - x[0]) if n > 1 else 1.0
    dy = float(y[1] - y[0]) if n > 1 else 1.0
    dz = float(z[1] - z[0]) if n > 1 else 1.0
    local_grid = FieldGrid(x=x, y=y, z=z, X=X, Y=Y, Z=Z, dx=dx, dy=dy, dz=dz)
    U = gorkov_potential_from_pressure(p, dx, dy, dz)

    finite_U = np.where(np.isfinite(U), U, np.inf)
    ix, iy, iz = np.unravel_index(int(np.argmin(finite_U)), finite_U.shape)
    refined_x = float(x[ix])
    refined_y = float(y[iy])
    refined_z = float(z[iz])
    g, Hc = _point_gradient_hessian(U, ix, iy, iz, dx, dy, dz)  # type: ignore
    if bool(np.all(np.isfinite(Hc))):
        eigvals = np.linalg.eigvalsh(Hc)
    else:
        eigvals = np.array([float("nan"), float("nan"), float("nan")], dtype=np.float64)
    refined_hessian_positive = bool(float(eigvals[0]) > 0.0)

    newton_dx = float("nan")
    newton_dy = float("nan")
    newton_dz = float("nan")
    newton_offset_norm = float("nan")
    newton_x = float("nan")
    newton_y = float("nan")
    newton_z = float("nan")
    newton_in_box = False
    if bool(np.all(np.isfinite(Hc))) and bool(np.all(np.isfinite(g))) and refined_hessian_positive:
        try:
            delta = np.linalg.solve(Hc, -g)
            if bool(np.all(np.isfinite(delta))):
                newton_dx, newton_dy, newton_dz = (float(delta[0]), float(delta[1]), float(delta[2]))
                newton_offset_norm = float(np.linalg.norm(delta))
                newton_x = refined_x + newton_dx
                newton_y = refined_y + newton_dy
                newton_z = refined_z + newton_dz
                newton_in_box = bool(
                    x[0] <= newton_x <= x[-1]
                    and y[0] <= newton_y <= y[-1]
                    and z[0] <= newton_z <= z[-1]
                )
        except np.linalg.LinAlgError:
            pass

    coarse_grad = float(candidate.get("grad_norm", float("nan")))
    refined_grad = float(np.linalg.norm(g))
    edge_minimum = bool(ix in (0, n - 1) or iy in (0, n - 1) or iz in (0, n - 1))
    refined_local_minimum_26 = _local_minimum_26_at(U, ix, iy, iz)      # type: ignore
    refined_p_abs_grid = np.abs(p)
    refined_p_abs = float(refined_p_abs_grid[ix, iy, iz])
    refined_box_p_ratio, refined_box_p_ref, refined_box_p_sample_count = local_pressure_ratio(
        refined_p_abs_grid,
        local_grid,
        ix,     # type: ignore
        iy,      # type: ignore
        iz,     # type: ignore
        trap_cfg,
    )
    refined_box_well_depth = local_shell_depth(U, ix, iy, iz, trap_cfg.depth_radius_cells)      # type: ignore
    validation_failures: list[str] = []
    if not refined_local_minimum_26:
        validation_failures.append("not_26_neighbor_minimum")
    if not refined_hessian_positive:
        validation_failures.append("hessian_not_positive_definite")
    if edge_minimum:
        validation_failures.append("minimum_on_refine_box_edge")
    if not newton_in_box:
        validation_failures.append("newton_point_outside_refine_box")
    refine_validation_pass = len(validation_failures) == 0
    refine_validation_reason = "pass" if refine_validation_pass else ";".join(validation_failures)
    coarse_to_refined_distance = float(
        math.sqrt((refined_x - cx) ** 2 + (refined_y - cy) ** 2 + (refined_z - cz) ** 2)
    )

    return {
        "rank": int(candidate["rank"]),
        "coarse_x": cx,
        "coarse_y": cy,
        "coarse_z": cz,
        "coarse_p_abs": float(candidate.get("p_abs", float("nan"))),
        "coarse_local_p_ratio": float(candidate.get("local_p_ratio", float("nan"))),
        "coarse_U": float(candidate.get("U", float("nan"))),
        "coarse_grad_norm": coarse_grad,
        "coarse_well_depth": float(candidate.get("well_depth", float("nan"))),
        "coarse_lambda_min_conf": float(candidate.get("lambda_min_conf", float("nan"))),
        "refined_x": refined_x,
        "refined_y": refined_y,
        "refined_z": refined_z,
        "refined_distance_from_coarse": coarse_to_refined_distance,
        "refined_p_abs": refined_p_abs,
        "refined_box_p_ratio": refined_box_p_ratio,
        "refined_box_p_ref": refined_box_p_ref,
        "refined_box_p_sample_count": refined_box_p_sample_count,
        "refined_U": float(U[ix, iy, iz]),
        "refined_grad_norm": refined_grad,
        "refined_grad_ratio_to_coarse": float(refined_grad / coarse_grad) if coarse_grad > 0.0 else float("nan"),
        "refined_box_well_depth": refined_box_well_depth,
        "refined_lambda_min_conf": float(eigvals[0]),
        "refined_local_minimum_26": refined_local_minimum_26,
        "refined_hessian_positive": refined_hessian_positive,
        "refined_minimum_on_edge": edge_minimum,
        "newton_dx": newton_dx,
        "newton_dy": newton_dy,
        "newton_dz": newton_dz,
        "newton_offset_norm": newton_offset_norm,
        "newton_x": newton_x,
        "newton_y": newton_y,
        "newton_z": newton_z,
        "newton_in_refine_box": newton_in_box,
        "refine_validation_pass": refine_validation_pass,
        "refine_validation_reason": refine_validation_reason,
        "local_refine_radius": radius,
        "local_refine_grid_size": n,
        "local_refine_dx": dx,
        "local_refine_dy": dy,
        "local_refine_dz": dz,
    }


def refine_candidates_local_minima(
    candidates: list[dict[str, float | int | bool | str]],
    phases: np.ndarray,
    cfg: SimulationConfig,
    trap_cfg: TrapDetectionConfig,
    _label: str = "refine",
) -> list[dict[str, float | int | bool | str]]:
    n = len(candidates)
    workers = _resolve_worker_count(trap_cfg.local_refine_workers, n)
    bar_kw: dict = dict(total=n, desc=f"  [{_label}]", unit="cand", leave=True)
    if workers <= 1:
        results = []
        for candidate in _tqdm(candidates, **bar_kw):
            results.append(refine_candidate_local_minimum(candidate, phases, cfg, trap_cfg))
        return results

    payloads = [(candidate, phases, cfg, trap_cfg) for candidate in candidates]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        mapped = executor.map(_refine_candidate_worker, payloads, chunksize=1)
        return list(_tqdm(mapped, **bar_kw))


def _refine_candidate_worker(
    payload: tuple[
        dict[str, float | int | bool | str],
        np.ndarray,
        SimulationConfig,
        TrapDetectionConfig,
    ],
) -> dict[str, float | int | bool | str]:
    candidate, phases, cfg, trap_cfg = payload
    return refine_candidate_local_minimum(candidate, phases, cfg, trap_cfg)


def validated_refined_candidates(
    rows: list[dict[str, float | int | bool | str]],
) -> list[dict[str, float | int | bool | str]]:
    return [row for row in rows if bool(row.get("refine_validation_pass", False))]


def summarize_refine_validation(rows: list[dict[str, float | int | bool | str]]) -> dict[str, int]:
    return {
        "input_count": len(rows),
        "valid_count": sum(bool(row.get("refine_validation_pass", False)) for row in rows),
        "not_26_neighbor_minimum_count": sum(
            not bool(row.get("refined_local_minimum_26", False)) for row in rows
        ),
        "hessian_not_positive_definite_count": sum(
            not bool(row.get("refined_hessian_positive", False)) for row in rows
        ),
        "minimum_on_refine_box_edge_count": sum(bool(row.get("refined_minimum_on_edge", False)) for row in rows),
        "newton_point_outside_refine_box_count": sum(
            not bool(row.get("newton_in_refine_box", False)) for row in rows
        ),
    }


# ============================================================
# Plotting
# ============================================================


def save_phase_grid_plot(phases_rad: np.ndarray, path: Path, title: str) -> None:
    phase_grid = phases_rad[TX_LAYOUT_1BASED - 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(phase_grid, origin="upper", cmap="twilight", vmin=-np.pi, vmax=np.pi)

    for r in range(5):
        for c in range(5):
            tx_num = TX_LAYOUT_1BASED[r, c]
            ax.text(
                c,
                r,
                f"{tx_num}\n{phase_grid[r, c]:+.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(title)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([1, 2, 3, 4, 5]) # type: ignore      
    ax.set_yticklabels([1, 2, 3, 4, 5])# type: ignore
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, label="Phase (rad)")
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_three_slice_plot(
    analysis: dict[str, np.ndarray | FieldGrid],
    marker_coord: tuple[float, float, float],
    path: Path,
    title: str,
) -> None:
    grid = analysis["grid"]
    assert isinstance(grid, FieldGrid)
    p_abs = np.asarray(analysis["p_abs"])

    ex, ey, ez = marker_coord
    x_idx = int(np.argmin(np.abs(grid.x - ex)))
    y_idx = int(np.argmin(np.abs(grid.y - ey)))
    z_idx = int(np.argmin(np.abs(grid.z - ez)))

    yz_abs = p_abs[x_idx, :, :]
    xz_abs = p_abs[:, y_idx, :]
    xy_abs = p_abs[:, :, z_idx]

    all_vals = np.concatenate([yz_abs.ravel(), xz_abs.ravel(), xy_abs.ravel()])
    vmax = float(np.quantile(all_vals[np.isfinite(all_vals)], 0.995))
    vmax = max(vmax, 1e-12)
    norm = colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(title, fontsize=15)

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(
        yz_abs.T,
        origin="lower",
        extent=[grid.y[0], grid.y[-1], grid.z[0], grid.z[-1]], # type: ignore 
        aspect="auto",
        cmap="viridis",
        norm=norm,
    )
    ax1.scatter([ey], [ez], c="red", marker="x", s=70, linewidths=2)
    ax1.set_title(f"YZ | x = {grid.x[x_idx]:.4f} m")
    ax1.set_xlabel("y (m)")
    ax1.set_ylabel("z (m)")
    plt.colorbar(im1, ax=ax1, label="|p|")

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(
        xz_abs.T,
        origin="lower",
        extent=[grid.x[0], grid.x[-1], grid.z[0], grid.z[-1]], # type: ignore 
        aspect="auto",
        cmap="viridis",
        norm=norm,
    )
    ax2.scatter([ex], [ez], c="red", marker="x", s=70, linewidths=2)
    ax2.set_title(f"XZ | y = {grid.y[y_idx]:.4f} m")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("z (m)")
    plt.colorbar(im2, ax=ax2, label="|p|")

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(
        xy_abs.T,
        origin="lower",
        extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], # type: ignore 
        aspect="auto",
        cmap="viridis",
        norm=norm,
    )
    ax3.scatter([ex], [ey], c="red", marker="x", s=70, linewidths=2)
    ax3.set_title(f"XY | z = {grid.z[z_idx]:.4f} m")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    plt.colorbar(im3, ax=ax3, label="|p|")

    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sim_coordinate_central_xz_plot(
    analysis: dict[str, np.ndarray | FieldGrid],
    reflector_z: float,
    path: Path,
    title: str,
) -> None:
    grid = analysis["grid"]
    assert isinstance(grid, FieldGrid)
    p_abs = np.asarray(analysis["p_abs"])

    y_idx = int(np.argmin(np.abs(grid.y)))
    x_mm = grid.x * 1000.0

    # Keep the simulator coordinate system: z=0 is the transducer plane and
    # z=reflector_z is the reflector. Compare with the paper figure inverted.
    z_mm = grid.z * 1000.0
    xz_abs = p_abs[:, y_idx, :]
    xz_norm = xz_abs / (float(np.nanmax(xz_abs)) + 1e-32)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        xz_norm.T,
        origin="lower",
        extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]], # type: ignore 
        aspect="auto",
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    ax.axhline(0.0, color="tab:gray", linewidth=2.0, alpha=0.35, label="transducer plane")
    ax.axhline(float(reflector_z) * 1000.0, color="tab:gray", linewidth=2.0, alpha=0.55, label="reflector")
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z from transducer plane [mm]")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="normalized |p|")
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_well_candidates_3d_plot(
    analysis: dict[str, np.ndarray | FieldGrid],
    phase_focus_coord: tuple[float, float, float],
    search_center: tuple[float, float, float],
    candidates: list[dict[str, float | int | bool | str]],
    reflector_z: float,
    path: Path,
    title: str,
) -> None:
    grid = analysis["grid"]
    assert isinstance(grid, FieldGrid)
    p_abs = np.asarray(analysis["p_abs"])

    threshold = float(np.quantile(p_abs[np.isfinite(p_abs)], 0.02))
    low_pressure_idx = np.argwhere(p_abs <= threshold)
    if low_pressure_idx.shape[0] > 3000:
        sample_idx = np.linspace(0, low_pressure_idx.shape[0] - 1, 3000).astype(np.int64)
        low_pressure_idx = low_pressure_idx[sample_idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    if low_pressure_idx.size > 0:
        lx = grid.x[low_pressure_idx[:, 0]]
        ly = grid.y[low_pressure_idx[:, 1]]
        lz = grid.z[low_pressure_idx[:, 2]]
        ax.scatter(lx, ly, lz, s=4, c="lightgray", alpha=0.18, label="lowest 2% |p| grid points") # type: ignore 

    if candidates:
        cx = np.array([float(row.get("refined_x", row.get("x", float("nan")))) for row in candidates], dtype=np.float64)
        cy = np.array([float(row.get("refined_y", row.get("y", float("nan")))) for row in candidates], dtype=np.float64)
        cz = np.array([float(row.get("refined_z", row.get("z", float("nan")))) for row in candidates], dtype=np.float64)
        cr = np.array([int(row["rank"]) for row in candidates], dtype=np.float64)
        scatter = ax.scatter(cx, cy, cz, s=42, c=cr, cmap="plasma_r", depthshade=True, label="well candidates") # type: ignore 
        plt.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1, label="candidate rank")

    fx, fy, fz = phase_focus_coord
    sx, sy, sz = search_center
    ax.scatter([fx], [fy], [fz], s=130, c="red", marker="*", label="phase focus") # type: ignore 
    ax.scatter([sx], [sy], [sz], s=80, c="tab:green", marker="x", label="search center") # type: ignore 

    px, py = np.meshgrid(
        [grid.x[0], grid.x[-1]],
        [grid.y[0], grid.y[-1]],
        indexing="ij",
    )
    pz = np.full_like(px, reflector_z, dtype=np.float64)
    ax.plot_surface(px, py, pz, color="tab:blue", alpha=0.08, linewidth=0)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim(grid.x[0], grid.x[-1])
    ax.set_ylim(grid.y[0], grid.y[-1])
    ax.set_zlim(grid.z[0], reflector_z)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_z_scan_plot(
    z_vals: np.ndarray,
    ideal_abs: np.ndarray,
    quant_abs: np.ndarray,
    design: SimulationDesign,
    best_well_z: float | None,
    reflector_z: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_vals, ideal_abs, label="ideal")
    ax.plot(z_vals, quant_abs, label="quantized", linestyle="--")
    ax.axvline(design.phase_focus_point[2], color="tab:red", linestyle=":", label="phase focus z")
    ax.axvline(design.search_center[2], color="tab:green", linestyle=":", label="search center z")
    if best_well_z is not None:
        ax.axvline(best_well_z, color="tab:orange", linestyle="-.", label="best detected well z")
    ax.axvline(reflector_z, color="tab:gray", linestyle="--", label="reflector z")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("|p| at phase-focus x,y")
    ax.set_title("Axial scan at phase-focus x,y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ============================================================
# Output writers
# ============================================================


def write_phase_table_csv(
    path: Path,
    distances: np.ndarray,
    phases_ideal: np.ndarray,
    ticks: np.ndarray,
    phases_quant: np.ndarray,
) -> None:
    phase_error = wrap_to_pi(phases_quant - phases_ideal)
    header = "tx_num,x_m,y_m,z_m,path_to_focus_m,phase_ideal_rad,tick,phase_quant_rad,phase_error_rad"
    rows = [header]
    for i in range(25):
        x, y, z = TRANSDUCER_POSITIONS[i]
        rows.append(
            ",".join(
                [
                    str(i + 1),
                    f"{x:.9f}",
                    f"{y:.9f}",
                    f"{z:.9f}",
                    f"{distances[i]:.9f}",
                    f"{phases_ideal[i]:.9f}",
                    str(int(ticks[i])),
                    f"{phases_quant[i]:.9f}",
                    f"{phase_error[i]:.9f}",
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_well_candidates_csv(
    path: Path,
    candidates: list[dict[str, float | int | bool | str]],
) -> None:
    columns = [
        "rank",
        "search_mode",
        "x",
        "y",
        "z",
        "p_abs",
        "local_p_ratio",
        "local_p_ref",
        "local_p_sample_count",
        "U",
        "grad_norm",
        "well_depth",
        "lambda_min_conf",
    ]
    rows = [",".join(columns)]
    for row in candidates:
        rows.append(
            ",".join(
                [
                    str(row["rank"]),
                    str(row["search_mode"]),
                    f"{float(row['x']):.9f}",
                    f"{float(row['y']):.9f}",
                    f"{float(row['z']):.9f}",
                    f"{float(row['p_abs']):.9e}",
                    f"{float(row.get('local_p_ratio', float('nan'))):.9e}",
                    f"{float(row.get('local_p_ref', float('nan'))):.9e}",
                    str(int(row.get("local_p_sample_count", 0))),
                    f"{float(row['U']):.9e}",
                    f"{float(row['grad_norm']):.9e}",
                    f"{float(row.get('well_depth', float('nan'))):.9e}",
                    f"{float(row['lambda_min_conf']):.9e}",
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_local_refine_csv(
    path: Path,
    rows_in: list[dict[str, float | int | bool | str]],
) -> None:
    columns = [
        "rank",
        "coarse_x",
        "coarse_y",
        "coarse_z",
        "coarse_p_abs",
        "coarse_local_p_ratio",
        "coarse_U",
        "coarse_grad_norm",
        "coarse_well_depth",
        "coarse_lambda_min_conf",
        "refined_x",
        "refined_y",
        "refined_z",
        "refined_distance_from_coarse",
        "refined_p_abs",
        "refined_box_p_ratio",
        "refined_box_p_ref",
        "refined_box_p_sample_count",
        "refined_U",
        "refined_grad_norm",
        "refined_grad_ratio_to_coarse",
        "refined_box_well_depth",
        "refined_lambda_min_conf",
        "refined_local_minimum_26",
        "refined_hessian_positive",
        "refined_minimum_on_edge",
        "newton_dx",
        "newton_dy",
        "newton_dz",
        "newton_offset_norm",
        "newton_x",
        "newton_y",
        "newton_z",
        "newton_in_refine_box",
        "refine_validation_pass",
        "refine_validation_reason",
        "local_refine_radius",
        "local_refine_grid_size",
        "local_refine_dx",
        "local_refine_dy",
        "local_refine_dz",
    ]

    def format_cell(value: float | int | bool | str) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.9e}"
        return str(value)

    rows = [",".join(columns)]
    for row in rows_in:
        rows.append(",".join(format_cell(row.get(column, "")) for column in columns))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_axis_scan_csv(
    path: Path,
    rows_in: list[dict[str, float | int | bool | str]],
) -> None:
    columns = [
        "index_z",
        "x",
        "y",
        "z",
        "p_abs",
        "U",
        "grad_norm",
        "lambda_min_conf",
        "potential_minimum_1d",
    ]
    rows = [",".join(columns)]
    for row in rows_in:
        rows.append(
            ",".join(
                [
                    str(int(row["index_z"])),
                    f"{float(row['x']):.9f}",
                    f"{float(row['y']):.9f}",
                    f"{float(row['z']):.9f}",
                    f"{float(row['p_abs']):.9e}",
                    f"{float(row['U']):.9e}",
                    f"{float(row['grad_norm']):.9e}",
                    f"{float(row['lambda_min_conf']):.9e}",
                    str(bool(row["potential_minimum_1d"])),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_ticks_txt(path: Path, ticks: np.ndarray) -> None:
    lines = ["Quantized ticks (Tx01..Tx25)", "=" * 36]
    for i, tick in enumerate(ticks, start=1):
        lines.append(f"Tx{i:02d}: {int(tick):4d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verilog(path: Path, module_name: str, ticks: np.ndarray) -> None:
    lines: list[str] = []
    lines.append(f"module {module_name} (")
    lines.append("    input  wire        CLOCK_50,")
    lines.append("    output wire [33:0] GPIO_1,")
    lines.append("    output wire [15:0] GPIO_0")
    lines.append(");")
    lines.append("")
    lines.append("    localparam integer HALF_PERIOD = 625;")
    lines.append("    localparam integer PERIOD      = 1250;")
    lines.append("")
    lines.append("    reg [31:0] div_counter = 32'd0;")
    lines.append("    reg        out_signal  = 1'b0;")
    lines.append("")
    lines.append("    always @(posedge CLOCK_50) begin")
    lines.append("        if (div_counter >= HALF_PERIOD - 1) begin")
    lines.append("            div_counter <= 32'd0;")
    lines.append("            out_signal  <= ~out_signal;")
    lines.append("        end else begin")
    lines.append("            div_counter <= div_counter + 32'd1;")
    lines.append("        end")
    lines.append("    end")
    lines.append("")
    lines.append("    wire [10:0] base_phase =")
    lines.append("        (out_signal) ? div_counter[10:0] : (div_counter[10:0] + 11'd625);")
    lines.append("")

    for i in range(25):
        lines.append(f"    localparam [10:0] TICK_TX{i+1:02d} = 11'd{int(ticks[i])};")

    lines.append("")
    lines.append("    function [10:0] phase_mod;")
    lines.append("        input [10:0] a;")
    lines.append("        input [10:0] b;")
    lines.append("        reg   [11:0] s;")
    lines.append("        begin")
    lines.append("            s = {1'b0, a} + {1'b0, b};")
    lines.append("            if (s >= 12'd1250)")
    lines.append("                phase_mod = s - 12'd1250;")
    lines.append("            else")
    lines.append("                phase_mod = s[10:0];")
    lines.append("        end")
    lines.append("    endfunction")
    lines.append("")
    lines.append("    wire [24:0] wav_tx;")
    lines.append("")

    for i in range(1, 26):
        wav_idx = i - 1
        pad = "  " if wav_idx < 10 else " "
        lines.append(
            f"    wire [10:0] phi{i:02d} = phase_mod(base_phase, TICK_TX{i:02d});"
            f"  assign wav_tx[{wav_idx}]{pad}= (phi{i:02d} < 11'd625);"
        )

    lines.append("")
    lines.append("    genvar i;")
    lines.append("    generate")
    lines.append("        for (i = 0; i < 17; i = i + 1) begin : trans_1_to_17")
    lines.append("            assign GPIO_1[2*i]     =  wav_tx[i];")
    lines.append("            assign GPIO_1[2*i + 1] = ~wav_tx[i];")
    lines.append("        end")
    lines.append("")
    lines.append("        for (i = 0; i < 8; i = i + 1) begin : trans_18_to_25")
    lines.append("            assign GPIO_0[2*i]     =  wav_tx[17 + i];")
    lines.append("            assign GPIO_0[2*i + 1] = ~wav_tx[17 + i];")
    lines.append("        end")
    lines.append("    endgenerate")
    lines.append("")
    lines.append("endmodule")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_output_readme(
    path: Path,
    module_name: str,
    trap_cfg: TrapDetectionConfig,
    selected_fraction: float,
    field_workers: int,
) -> None:
    lines = [
        "Output folder guide",
        "=" * 19,
        "",
        "Root",
        "- summary.json: full machine-readable run config and results.",
        "- README_outputs.txt: this guide.",
        f"- Field, trap, and z-scan outputs exclude z < {MIN_Z * 1000.0:.1f} mm near the transducer plane.",
        "",
        "Trap detection and filtering",
        "- Primary condition: U_G is a 26-neighbor local minimum and lambda_min(H_U) > 0.",
        "- Secondary filters: primary -> primary+p -> primary+p+lambda_min -> primary+p+lambda_min+depth.",
        "- primary+p keeps the lowest local_p_ratio, where local_p_ratio = |p(candidate)| / percentile95(|p| within local_p_radius).",
        "- lambda_min keeps the highest filter_fraction, and depth keeps the highest filter_fraction.",
        "- refined/refined_candidates_*.csv refines each secondary candidate in a high-resolution local box and reports the refined minimum and Newton equilibrium estimate.",
        "- Local-refine validation passes when the refined point is a 26-neighbor local minimum, lambda_min(H_U) > 0, the minimum is not on the refine-box edge, and the Newton equilibrium estimate remains inside the refine box.",
        "- newton_offset_norm = ||-H_U^-1 grad(U_G)|| is a diagnostic displacement estimate, not a hard cutoff.",
        "- final_candidates_*.csv and final_candidates_3d_*.png contain only candidates passing the local-refine validation checks.",
        "- Boundary ties are preserved when p, local_p_ratio, U_G, grad_norm, depth, and lambda_min match within the physical tie tolerance.",
        "- well_depth is estimated as min(U_G on a local Chebyshev shell) - U_G(candidate).",
        f"- Hyperparameters: filter_fraction = {selected_fraction:.3g}, depth_radius_cells = {trap_cfg.depth_radius_cells}, "
        f"local_p_radius = {trap_cfg.local_p_radius:.6g} m, local_p_reference_percentile = {trap_cfg.local_p_reference_percentile:.3g}, "
        f"local_p_min_samples = {trap_cfg.local_p_min_samples}, local_refine_radius = {trap_cfg.local_refine_radius:.6g} m, "
        f"field_workers = {field_workers}, local_refine_grid_size = {ensure_odd(trap_cfg.local_refine_grid_size)}, "
        f"local_refine_workers = {trap_cfg.local_refine_workers}, "
        f"physical_tie_rtol = {trap_cfg.physical_tie_rtol:.3g}, physical_tie_atol = {trap_cfg.physical_tie_atol:.3g}.",
        "",
        "01_hardware_phase/",
        f"- {module_name}.v: generated Verilog for the DE0-Nano output phase pattern.",
        "- phase_table.csv: Tx position, path length, ideal phase, quantized tick, and phase error.",
        "- quantized_ticks.txt: compact Tx01..Tx25 tick list.",
        "- phases_ideal.png / phases_quantized.png: 5x5 phase maps.",
        "",
        "02_trap_candidates/",
        "- final_candidates_*.csv: final trusted trap candidates.",
        "- coarse/: non-refined primary and secondary-stage candidate CSVs.",
        "- refined/: high-resolution local-refine diagnostic CSVs.",
        "- axis/: center-axis z scan diagnostic CSVs.",
        "",
        "03_figures/field/",
        "- field_slices_*.png: three pressure-field slices around the selected well.",
        "- central_xz_sim_coords_*.png: paper-style central XZ normalized |p| view.",
        "- z_scan_focus_xy.png: axial |p| scan at the phase-focus x,y position.",
        "",
        "03_figures/traps_3d/",
        "- primary_candidates_3d_*.png: 3D view of all primary trap candidates.",
        "- primary+p*_3d_*.png: 3D view after each selected filter.",
        "- final_candidates_3d_*.png: 3D view of final trusted refined candidates.",
        "",
        "*_ideal uses continuous ideal phases. *_quantized uses FPGA tick-quantized phases.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Reporting helpers
# ============================================================


def compute_z_scan(phases: np.ndarray, x: float, y: float, z_lo: float, z_hi: float, n: int, cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(z_lo, z_hi, ensure_odd(n), dtype=np.float64)
    pts = np.column_stack([np.full_like(z, x), np.full_like(z, y), z])
    p = pressure_at_points(pts, phases, cfg)
    return z, np.abs(p)


def print_summary(label: str, focus_metrics: dict[str, float], candidates: list[dict[str, float | int | bool | str]]) -> None:
    print(f"[{label}] search-center local metrics")
    print(
        "  |p|={p_abs:.3e}, |gradU|={grad_norm:.3e}, "
        "lambda_min={lambda_min_conf:.3e}".format(**focus_metrics)
    )
    if not candidates:
        print(f"[{label}] no well candidates found")
        return

    best = candidates[0]
    print(f"[{label}] first primary well candidate")
    print(
        "  coord=({x:.5f}, {y:.5f}, {z:.5f}) m, "
        "|p|={p_abs:.3e}, local_p_ratio={local_p_ratio:.3e}, "
        "|gradU|={grad_norm:.3e}, lambda_min={lambda_min_conf:.3e}".format(**best)
    )


# ============================================================
# Main workflow
# ============================================================


def run(cfg: SimulationConfig) -> dict[str, object]:
    _t_run = time.perf_counter()
    backend = "gpu" if _use_gpu() else ("numba" if _use_numba() else "numpy")
    n = ensure_odd(cfg.analysis_grid_size)
    print(
        f"\n{'='*60}\n"
        f"  Sim start  grid={n}³  backend={backend}  CPUs={CPU_COUNT}\n"
        f"  focus={cfg.phase_focus_coord}  reflector_z={cfg.reflector_z}\n"
        f"{'='*60}",
        flush=True,
    )

    design = make_simulation_design(cfg)
    phases_ideal, distances = geometric_focus_phases(design.phase_focus_point)
    ticks, phases_quant = quantize_phases_to_ticks(phases_ideal)

    print("\n--- [1/6] Ideal field analysis ---", flush=True)
    _t = time.perf_counter()
    ideal_analysis = analyze_field(phases_ideal, cfg, _label="IDEAL ")
    print(f"  [IDEAL ] field analysis  {time.perf_counter()-_t:.2f}s\n", flush=True)

    print("--- [2/6] Quantized field analysis ---", flush=True)
    _t = time.perf_counter()
    quant_analysis = analyze_field(phases_quant, cfg, _label="QUANT ")
    print(f"  [QUANT ] field analysis  {time.perf_counter()-_t:.2f}s\n", flush=True)

    print("--- [3/6] Local metrics at search center ---", flush=True)
    ideal_search_center_metrics = local_metrics_at_coord(design.search_center, phases_ideal, cfg)
    quant_search_center_metrics = local_metrics_at_coord(design.search_center, phases_quant, cfg)

    trap_cfg = TrapDetectionConfig(local_refine_workers=cfg.local_refine_workers)

    print("--- [4/6] Primary trap candidates ---", flush=True)
    _t = time.perf_counter()
    ideal_candidates = find_primary_trap_candidates(
        ideal_analysis,
        design.search_center,
        cfg.well_search_radius,
        trap_cfg,
    )
    quant_candidates = find_primary_trap_candidates(
        quant_analysis,
        design.search_center,
        cfg.well_search_radius,
        trap_cfg,
    )
    ideal_stages, ideal_selection = selected_candidates_by_filters(
        ideal_candidates,
        cfg.selected_candidate_fraction,
        trap_cfg,
    )
    quant_stages, quant_selection = selected_candidates_by_filters(
        quant_candidates,
        cfg.selected_candidate_fraction,
        trap_cfg,
    )
    ideal_selected_candidates = ideal_stages[FINAL_FILTER_STAGE]
    quant_selected_candidates = quant_stages[FINAL_FILTER_STAGE]
    print(
        f"  ideal:  primary={len(ideal_candidates)}  selected={len(ideal_selected_candidates)}\n"
        f"  quant:  primary={len(quant_candidates)}  selected={len(quant_selected_candidates)}\n"
        f"  {time.perf_counter()-_t:.2f}s\n",
        flush=True,
    )

    print("--- [5/6] Local refinement ---", flush=True)
    _t = time.perf_counter()
    ideal_refined_candidates = refine_candidates_local_minima(
        ideal_selected_candidates, phases_ideal, cfg, trap_cfg, _label="IDEAL  refine"
    )
    quant_refined_candidates = refine_candidates_local_minima(
        quant_selected_candidates, phases_quant, cfg, trap_cfg, _label="QUANT  refine"
    )
    print(f"  refine done  {time.perf_counter()-_t:.2f}s\n", flush=True)
    ideal_refined_valid_candidates = validated_refined_candidates(ideal_refined_candidates)
    quant_refined_valid_candidates = validated_refined_candidates(quant_refined_candidates)
    ideal_final_candidates = ideal_refined_valid_candidates
    quant_final_candidates = quant_refined_valid_candidates
    ideal_axis_scan = axis_scan_rows(ideal_analysis, (design.phase_focus_point[0], design.phase_focus_point[1]))
    quant_axis_scan = axis_scan_rows(quant_analysis, (design.phase_focus_point[0], design.phase_focus_point[1]))

    best_source = quant_selected_candidates or quant_candidates
    best_refined_source = quant_refined_valid_candidates or quant_refined_candidates
    best_marker = (
        (
            float(best_refined_source[0]["refined_x"]),
            float(best_refined_source[0]["refined_y"]),
            float(best_refined_source[0]["refined_z"]),
        )
        if best_refined_source
        else (float(best_source[0]["x"]), float(best_source[0]["y"]), float(best_source[0]["z"]))
        if best_source
        else design.search_center
    )

    z_lo = max(cfg.z_extent[0], MIN_Z)
    z_hi = min(cfg.reflector_z - 1e-4, cfg.z_extent[1])
    z_scan, ideal_z_abs = compute_z_scan(
        phases_ideal,
        x=design.phase_focus_point[0],
        y=design.phase_focus_point[1],
        z_lo=z_lo,
        z_hi=z_hi,
        n=max(401, cfg.analysis_grid_size),
        cfg=cfg,
    )
    _, quant_z_abs = compute_z_scan(
        phases_quant,
        x=design.phase_focus_point[0],
        y=design.phase_focus_point[1],
        z_lo=z_lo,
        z_hi=z_hi,
        n=max(401, cfg.analysis_grid_size),
        cfg=cfg,
    )

    phase_error = wrap_to_pi(phases_quant - phases_ideal)
    quant_summary = {
        "max_abs_phase_error_rad": float(np.max(np.abs(phase_error))),
        "rms_phase_error_rad": float(np.sqrt(np.mean(phase_error**2))),
    }

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    hardware_phase_dir = cfg.out_dir / "01_hardware_phase"
    trap_candidates_dir = cfg.out_dir / "02_trap_candidates"
    coarse_candidates_dir = trap_candidates_dir / "coarse"
    refined_candidates_dir = trap_candidates_dir / "refined"
    axis_candidates_dir = trap_candidates_dir / "axis"
    field_figures_dir = cfg.out_dir / "03_figures" / "field"
    trap_figures_dir = cfg.out_dir / "03_figures" / "traps_3d"
    for output_dir in (
        hardware_phase_dir,
        trap_candidates_dir,
        coarse_candidates_dir,
        refined_candidates_dir,
        axis_candidates_dir,
        field_figures_dir,
        trap_figures_dir,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

    phase_table_path = hardware_phase_dir / "phase_table.csv"
    ticks_txt_path = hardware_phase_dir / "quantized_ticks.txt"
    verilog_path = hardware_phase_dir / f"{cfg.module_name}.v"
    ideal_phase_png = hardware_phase_dir / "phases_ideal.png"
    quant_phase_png = hardware_phase_dir / "phases_quantized.png"

    def stage_csv_path(stage: str, variant: str) -> Path:
        if stage == "primary":
            return coarse_candidates_dir / f"primary_candidates_{variant}.csv"
        return coarse_candidates_dir / f"{stage}_candidates_{variant}.csv"

    def stage_3d_path(stage: str, variant: str) -> Path:
        if stage == "primary":
            return trap_figures_dir / f"primary_candidates_3d_{variant}.png"
        return trap_figures_dir / f"{stage}_3d_{variant}.png"

    ideal_stage_csv_paths = {stage: stage_csv_path(stage, "ideal") for stage in FILTER_STAGE_ORDER}
    quant_stage_csv_paths = {stage: stage_csv_path(stage, "quantized") for stage in FILTER_STAGE_ORDER}
    ideal_selected_candidates_csv = ideal_stage_csv_paths[FINAL_FILTER_STAGE]
    quant_selected_candidates_csv = quant_stage_csv_paths[FINAL_FILTER_STAGE]
    ideal_refined_candidates_csv = refined_candidates_dir / "refined_candidates_ideal.csv"
    quant_refined_candidates_csv = refined_candidates_dir / "refined_candidates_quantized.csv"
    ideal_final_candidates_csv = trap_candidates_dir / "final_candidates_ideal.csv"
    quant_final_candidates_csv = trap_candidates_dir / "final_candidates_quantized.csv"
    ideal_axis_scan_csv = axis_candidates_dir / "axis_scan_ideal.csv"
    quant_axis_scan_csv = axis_candidates_dir / "axis_scan_quantized.csv"

    ideal_slice_png = field_figures_dir / "field_slices_ideal.png"
    quant_slice_png = field_figures_dir / "field_slices_quantized.png"
    ideal_central_xz_png = field_figures_dir / "central_xz_sim_coords_ideal.png"
    quant_central_xz_png = field_figures_dir / "central_xz_sim_coords_quantized.png"
    z_scan_png = field_figures_dir / "z_scan_focus_xy.png"

    ideal_stage_3d_paths = {stage: stage_3d_path(stage, "ideal") for stage in FILTER_STAGE_ORDER}
    quant_stage_3d_paths = {stage: stage_3d_path(stage, "quantized") for stage in FILTER_STAGE_ORDER}
    ideal_selected_candidates_3d_png = ideal_stage_3d_paths[FINAL_FILTER_STAGE]
    quant_selected_candidates_3d_png = quant_stage_3d_paths[FINAL_FILTER_STAGE]
    ideal_final_candidates_3d_png = trap_figures_dir / "final_candidates_3d_ideal.png"
    quant_final_candidates_3d_png = trap_figures_dir / "final_candidates_3d_quantized.png"

    summary_json_path = cfg.out_dir / "summary.json"
    output_readme_path = cfg.out_dir / "README_outputs.txt"

    write_phase_table_csv(phase_table_path, distances, phases_ideal, ticks, phases_quant)
    for stage in FILTER_STAGE_ORDER:
        write_well_candidates_csv(ideal_stage_csv_paths[stage], ideal_stages[stage])
        write_well_candidates_csv(quant_stage_csv_paths[stage], quant_stages[stage])
    write_local_refine_csv(ideal_refined_candidates_csv, ideal_refined_candidates)
    write_local_refine_csv(quant_refined_candidates_csv, quant_refined_candidates)
    write_local_refine_csv(ideal_final_candidates_csv, ideal_final_candidates)
    write_local_refine_csv(quant_final_candidates_csv, quant_final_candidates)
    write_axis_scan_csv(ideal_axis_scan_csv, ideal_axis_scan)
    write_axis_scan_csv(quant_axis_scan_csv, quant_axis_scan)
    write_ticks_txt(ticks_txt_path, ticks)
    write_verilog(verilog_path, cfg.module_name, ticks)
    save_phase_grid_plot(phases_ideal, ideal_phase_png, "Ideal phase-focus phases")
    save_phase_grid_plot(phases_quant, quant_phase_png, "Quantized hardware phases")
    save_three_slice_plot(ideal_analysis, best_marker, ideal_slice_png, "Ideal field |p| slices around best quantized well")
    save_three_slice_plot(quant_analysis, best_marker, quant_slice_png, "Quantized field |p| slices around best well")
    save_sim_coordinate_central_xz_plot(
        ideal_analysis,
        cfg.reflector_z,
        ideal_central_xz_png,
        "Ideal central XZ normalized |p|, simulator coordinates",
    )
    save_sim_coordinate_central_xz_plot(
        quant_analysis,
        cfg.reflector_z,
        quant_central_xz_png,
        "Quantized central XZ normalized |p|, simulator coordinates",
    )
    for stage in FILTER_STAGE_ORDER:
        save_well_candidates_3d_plot(
            ideal_analysis,
            design.phase_focus_point,
            design.search_center,
            ideal_stages[stage],
            cfg.reflector_z,
            ideal_stage_3d_paths[stage],
            f"Ideal {stage} candidates in 3D field",
        )
        save_well_candidates_3d_plot(
            quant_analysis,
            design.phase_focus_point,
            design.search_center,
            quant_stages[stage],
            cfg.reflector_z,
            quant_stage_3d_paths[stage],
            f"Quantized {stage} candidates in 3D field",
        )
    save_well_candidates_3d_plot(
        ideal_analysis,
        design.phase_focus_point,
        design.search_center,
        ideal_final_candidates,
        cfg.reflector_z,
        ideal_final_candidates_3d_png,
        "Ideal final refined-valid candidates in 3D field",
    )
    save_well_candidates_3d_plot(
        quant_analysis,
        design.phase_focus_point,
        design.search_center,
        quant_final_candidates,
        cfg.reflector_z,
        quant_final_candidates_3d_png,
        "Quantized final refined-valid candidates in 3D field",
    )
    save_z_scan_plot(
        z_scan,
        ideal_z_abs,
        quant_z_abs,
        design,
        best_marker[2] if quant_candidates else None,
        cfg.reflector_z,
        z_scan_png,
    )

    summary = {
        "config": {
            **asdict(cfg),
            "out_dir": str(cfg.out_dir),
        },
        "design": asdict(design),
        "gorkov_potential": gorkov_potential_config_dict(GORKOV_POTENTIAL_CONFIG),
        "trap_detection": trap_detection_config_dict(trap_cfg),
        "phase_quantization": quant_summary,
        "ideal_search_center_metrics": ideal_search_center_metrics,
        "ideal_selection": ideal_selection,
        "ideal_filter_stages": ideal_stages,
        "ideal_selected_candidates": ideal_selected_candidates,
        "ideal_refined_candidates": ideal_refined_candidates,
        "ideal_refined_valid_candidates": ideal_refined_valid_candidates,
        "ideal_final_candidates": ideal_final_candidates,
        "ideal_refine_validation": summarize_refine_validation(ideal_refined_candidates),
        "ideal_primary_candidates": ideal_candidates,
        "ideal_axis_scan": ideal_axis_scan,
        "quantized_search_center_metrics": quant_search_center_metrics,
        "quantized_selection": quant_selection,
        "quantized_filter_stages": quant_stages,
        "quantized_selected_candidates": quant_selected_candidates,
        "quantized_refined_candidates": quant_refined_candidates,
        "quantized_refined_valid_candidates": quant_refined_valid_candidates,
        "quantized_final_candidates": quant_final_candidates,
        "quantized_refine_validation": summarize_refine_validation(quant_refined_candidates),
        "quantized_primary_candidates": quant_candidates,
        "quantized_axis_scan": quant_axis_scan,
        "files": {
            "output_readme": str(output_readme_path),
            "hardware_phase_dir": str(hardware_phase_dir),
            "trap_candidates_dir": str(trap_candidates_dir),
            "coarse_candidates_dir": str(coarse_candidates_dir),
            "refined_candidates_dir": str(refined_candidates_dir),
            "axis_candidates_dir": str(axis_candidates_dir),
            "field_figures_dir": str(field_figures_dir),
            "trap_figures_dir": str(trap_figures_dir),
            "ideal_stage_candidate_csvs": {stage: str(path) for stage, path in ideal_stage_csv_paths.items()},
            "quant_stage_candidate_csvs": {stage: str(path) for stage, path in quant_stage_csv_paths.items()},
            "ideal_stage_3d_pngs": {stage: str(path) for stage, path in ideal_stage_3d_paths.items()},
            "quant_stage_3d_pngs": {stage: str(path) for stage, path in quant_stage_3d_paths.items()},
            "phase_table_csv": str(phase_table_path),
            "ideal_primary_candidates_csv": str(ideal_stage_csv_paths["primary"]),
            "quant_primary_candidates_csv": str(quant_stage_csv_paths["primary"]),
            "ideal_selected_candidates_csv": str(ideal_selected_candidates_csv),
            "quant_selected_candidates_csv": str(quant_selected_candidates_csv),
            "ideal_refined_candidates_csv": str(ideal_refined_candidates_csv),
            "quant_refined_candidates_csv": str(quant_refined_candidates_csv),
            "ideal_final_candidates_csv": str(ideal_final_candidates_csv),
            "quant_final_candidates_csv": str(quant_final_candidates_csv),
            "ideal_axis_scan_csv": str(ideal_axis_scan_csv),
            "quant_axis_scan_csv": str(quant_axis_scan_csv),
            "ticks_txt": str(ticks_txt_path),
            "verilog": str(verilog_path),
            "ideal_phase_png": str(ideal_phase_png),
            "quant_phase_png": str(quant_phase_png),
            "ideal_slice_png": str(ideal_slice_png),
            "quant_slice_png": str(quant_slice_png),
            "ideal_central_xz_sim_coords_png": str(ideal_central_xz_png),
            "quant_central_xz_sim_coords_png": str(quant_central_xz_png),
            "ideal_primary_candidates_3d_png": str(ideal_stage_3d_paths["primary"]),
            "quant_primary_candidates_3d_png": str(quant_stage_3d_paths["primary"]),
            "ideal_selected_candidates_3d_png": str(ideal_selected_candidates_3d_png),
            "quant_selected_candidates_3d_png": str(quant_selected_candidates_3d_png),
            "ideal_final_candidates_3d_png": str(ideal_final_candidates_3d_png),
            "quant_final_candidates_3d_png": str(quant_final_candidates_3d_png),
            "z_scan_png": str(z_scan_png),
        },
    }
    print("--- [6/6] Writing outputs ---", flush=True)
    _t = time.perf_counter()
    write_output_readme(output_readme_path, cfg.module_name, trap_cfg, cfg.selected_candidate_fraction, cfg.field_workers)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  outputs written  {time.perf_counter()-_t:.2f}s", flush=True)

    _t_total = time.perf_counter() - _t_run
    print(
        f"\n{'='*60}\n"
        f"  DONE  total={_t_total:.1f}s\n"
        f"{'='*60}",
        flush=True,
    )
    print("[INFO] Requested phase focus:", design.requested_phase_focus)
    print("[INFO] Phase focus point:", design.phase_focus_point)
    print("[INFO] Search center:", design.search_center)
    print(f"[INFO] Phase focus - reflector z: {design.phase_focus_minus_reflector_z:+.6e} m")
    if cfg.well_search_radius is None:
        print("[INFO] Well search mode: full 3D grid")
    else:
        print(f"[INFO] Well search radius around search center: {cfg.well_search_radius:.6e} m")
    print(
        "[INFO] Quantization error: "
        f"max={quant_summary['max_abs_phase_error_rad']:.3e} rad, "
        f"rms={quant_summary['rms_phase_error_rad']:.3e} rad"
    )
    print(
        "[INFO] Candidates:\n"
        f"  ideal:  primary={len(ideal_candidates)}  selected={len(ideal_selected_candidates)}"
        f"  refined={len(ideal_refined_candidates)}  valid={len(ideal_refined_valid_candidates)}  final={len(ideal_final_candidates)}\n"
        f"  quant:  primary={len(quant_candidates)}  selected={len(quant_selected_candidates)}"
        f"  refined={len(quant_refined_candidates)}  valid={len(quant_refined_valid_candidates)}  final={len(quant_final_candidates)}"
    )
    print(
        f"[INFO] backend={backend}  grid={n}³  refine_grid={ideal_selection['local_refine_grid_size']}"
        f"  keep_fraction={ideal_selection['filter_fraction']:.2f}"
        f"  refine_workers={_resolve_worker_count(trap_cfg.local_refine_workers, max(1, len(quant_selected_candidates)))}"
    )
    print_summary("IDEAL PRIMARY", ideal_search_center_metrics, ideal_candidates)
    print_summary("QUANTIZED PRIMARY", quant_search_center_metrics, quant_candidates)
    print(f"[INFO] Outputs written under: {cfg.out_dir}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware-matched reflector trap simulator for the 5x5 PAT.")
    parser.add_argument("--phase-focus-x", "--focus-x", "--target-x", dest="phase_focus_x", type=float, default=PHASE_FOCUS[0], help="Phase-focus x used for geometric path compensation (m)")
    parser.add_argument("--phase-focus-y", "--focus-y", "--target-y", dest="phase_focus_y", type=float, default=PHASE_FOCUS[1], help="Phase-focus y used for geometric path compensation (m)")
    parser.add_argument("--phase-focus-z", "--focus-z", "--target-z", dest="phase_focus_z", type=float, default=PHASE_FOCUS[2], help="Phase-focus z used for geometric path compensation (m)")
    parser.add_argument("--search-center-x", type=float, default=None, help="Optional well-search center x (m)")
    parser.add_argument("--search-center-y", type=float, default=None, help="Optional well-search center y (m)")
    parser.add_argument("--search-center-z", type=float, default=None, help="Optional well-search center z (m)")
    parser.add_argument("--reflector-z", type=float, default=REFLECTOR_Z, help="Reflector plane z (m)")
    parser.add_argument("--analysis-grid", type=int, default=GRID_SIZE, help="Odd-sized 3D analysis grid resolution")
    parser.add_argument("--selected-candidate-fraction", type=float, default=SELECTED_FRACTION, help="Fraction kept by the p, lambda_min, and depth selected-candidate filters.")
    parser.add_argument("--well-search-radius", type=float, default=SEARCH_RADIUS, help="Optional 3D radius around the search center. Omit for full-grid search (m)")
    parser.add_argument("--reflection-coeff", type=float, default=1.0, help="Pressure reflection coefficient")
    parser.add_argument("--reflection-phase-rad", type=float, default=0.0, help="Additional reflector phase shift (rad)")
    parser.add_argument("--local-derivative-step", type=float, default=5e-4, help="Finite-difference step for exact local diagnostics (m)")
    parser.add_argument(
        "--field-workers",
        type=int,
        default=FIELD_WORKERS,
        help="Parallel worker count for full-grid pressure calculation. Use 1 for serial, 0 for every logical CPU core.",
    )
    parser.add_argument(
        "--local-refine-workers",
        type=int,
        default=LOCAL_REFINE_WORKERS,
        help="Parallel worker count for local refine. Use 1 for serial, 0 for every logical CPU core.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory. Default creates attempt_### automatically.")
    parser.add_argument("--module-name", type=str, default=MODULE_NAME, help="Verilog module name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else create_attempt_dir(OUTPUT_ROOT)
    z_hi = max(args.reflector_z - 1e-4, 0.006)
    search_args = (args.search_center_x, args.search_center_y, args.search_center_z)
    if any(v is not None for v in search_args):
        if any(v is None for v in search_args):
            raise ValueError("Set all of --search-center-x/y/z, or omit all three for the default marker.")
        search_center = (float(args.search_center_x), float(args.search_center_y), float(args.search_center_z))
    else:
        search_center = SEARCH_CENTER

    cfg = SimulationConfig(
        phase_focus_coord=(args.phase_focus_x, args.phase_focus_y, args.phase_focus_z),
        search_center=search_center,
        reflector_z=args.reflector_z,
        reflection_coeff=args.reflection_coeff,
        reflection_phase_rad=args.reflection_phase_rad,
        analysis_grid_size=ensure_odd(args.analysis_grid),
        local_derivative_step=args.local_derivative_step,
        selected_candidate_fraction=args.selected_candidate_fraction,
        well_search_radius=args.well_search_radius,
        field_workers=args.field_workers,
        local_refine_workers=args.local_refine_workers,
        z_extent=(MIN_Z, z_hi),
        out_dir=out_dir,
        module_name=args.module_name,
    )
    run(cfg)


if __name__ == "__main__":
    main()
