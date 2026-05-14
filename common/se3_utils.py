"""SE(3) Lie algebra utilities shared across pipeline modules."""

from __future__ import annotations

import numpy as np


def _hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) matrix → 6-vector [u; omega] (translation part first)."""
    R = T[:3, :3]
    t = T[:3, 3]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        omega = np.zeros(3)
        V_inv = np.eye(3)
    else:
        log_R = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        omega = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
        omega_hat = log_R
        c = (1.0 / theta**2) * (1.0 - theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta))))
        V_inv = np.eye(3) - 0.5 * omega_hat + c * (omega_hat @ omega_hat)

    u = V_inv @ t
    return np.concatenate([u, omega])


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """6-vector [u; omega] → SE(3) matrix."""
    u, omega = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))

    if theta < 1e-8:
        R = np.eye(3)
        V = np.eye(3)
    else:
        omega_hat = _hat(omega)
        s, c = np.sin(theta), np.cos(theta)
        R = np.eye(3) + (s / theta) * omega_hat + ((1.0 - c) / theta**2) * (omega_hat @ omega_hat)
        V = np.eye(3) + ((1.0 - c) / theta**2) * omega_hat + ((theta - s) / theta**3) * (omega_hat @ omega_hat)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ u
    return T


def _average_se3(transforms: list[np.ndarray]) -> np.ndarray:
    """Lie algebra mean of a list of SE(3) matrices."""
    if len(transforms) == 1:
        return transforms[0].copy()
    mean_T = transforms[0].copy()
    for _ in range(100):
        deltas = [_se3_log(np.linalg.inv(mean_T) @ T) for T in transforms]
        mean_delta = np.mean(deltas, axis=0)
        mean_T = mean_T @ _se3_exp(mean_delta)
        if np.linalg.norm(mean_delta) < 1e-10:
            break
    return mean_T
