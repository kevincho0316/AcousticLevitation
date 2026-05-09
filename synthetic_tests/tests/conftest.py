"""Shared fixtures for synthetic pipeline tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so pipeline modules resolve.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_tests.synth.cameras import make_default_scene
from synthetic_tests.synth.scene import SyntheticScene


@pytest.fixture(scope="session")
def default_scene() -> SyntheticScene:
    """4-camera ring scene, box centre ball, no noise."""
    return make_default_scene(n_cameras=4)


@pytest.fixture(scope="session")
def centre_ball_pos() -> np.ndarray:
    W, D, H = 0.12, 0.12, 0.06
    return np.array([W / 2, H / 2, D / 2])
