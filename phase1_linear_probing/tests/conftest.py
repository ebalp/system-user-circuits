"""Shared fixtures for phase1_linear_probing tests.

All fixtures use synthetic data — no dependency on real experiment files.
"""

import sys
from pathlib import Path

# Add phase1_linear_probing/ to sys.path so tests can import `data`, `probe`,
# etc. directly — matching the notebook import pattern.  Also add the repo root
# so that `phase1_linear_probing.data` style imports work for the CLI check.
_phase1_dir = Path(__file__).resolve().parent.parent
_repo_root = _phase1_dir.parent
for _p in (_phase1_dir, _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_activations():
    """50 samples, 4 layers, 16 dims, 2 positions."""
    rng = np.random.default_rng(42)
    return {
        "last_prompt": rng.standard_normal((50, 4, 16)).astype(np.float32),
        "last_system": rng.standard_normal((50, 4, 16)).astype(np.float32),
    }


@pytest.fixture
def synthetic_labels():
    """50 binary labels, ~balanced."""
    return np.array([0, 1] * 25)


@pytest.fixture
def synthetic_groups():
    """50 samples across 4 constraint types."""
    return np.array(
        ["type_a"] * 13
        + ["type_b"] * 12
        + ["type_c"] * 13
        + ["type_d"] * 12
    )


@pytest.fixture
def synthetic_condition_c_df():
    """Minimal DataFrame mimicking Condition C data (50 rows)."""
    rng = np.random.default_rng(42)
    n = 50
    constraint_types = ["format", "language", "emoji", "capitalization"]
    return pd.DataFrame(
        {
            "condition": ["C"] * n,
            "label": rng.choice(
                ["followed_system", "followed_user"], size=n
            ),
            "constraint_type": rng.choice(constraint_types, size=n),
            "strength": rng.choice(["weak", "medium", "strong"], size=n),
            "user_style": rng.choice(
                ["with_instruction", "jailbreak"], size=n
            ),
            "task_id": rng.choice(["photosynthesis", "recipe"], size=n),
            "direction": rng.choice(["a_to_b", "b_to_a"], size=n),
            "system_prompt": ["Respond in Spanish."] * n,
            "user_prompt": ["Explain photosynthesis."] * n,
        }
    )
