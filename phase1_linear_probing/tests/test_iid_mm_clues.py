"""Pure-function tests for iid_mm_steering_clues.

No HTTP, no fixtures — just synthetic stats/scales dicts feeding the
helper and asserting the math.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_utils import iid_mm_steering_clues  # noqa: E402


def _make_inputs(
    *,
    constraint: str = "_overall",
    direction_name: str = "iid_mm",
    layer: int = 12,
    proj_std: float = 2.5,
    y0_mean: float = -0.4,
    y0_std: float = 0.7,
    y1_mean: float = 0.6,
    y1_std: float = 0.8,
):
    stats = {
        constraint: {
            f"L{layer}": {
                direction_name: {
                    "y0": {"mean": y0_mean, "std": y0_std},
                    "y1": {"mean": y1_mean, "std": y1_std},
                }
            }
        }
    }
    scales = {f"{direction_name}_L{layer}": proj_std}
    return stats, scales


def test_basic_math():
    stats, scales = _make_inputs(
        proj_std=2.5,
        y0_mean=-0.4,
        y0_std=0.7,
        y1_mean=0.6,
        y1_std=0.8,
    )
    clues = iid_mm_steering_clues(stats, scales, "_overall", "iid_mm", 12)

    assert clues["proj_std"] == pytest.approx(2.5)
    assert clues["y0_mean"] == pytest.approx(-0.4)
    assert clues["y0_std"] == pytest.approx(0.7)
    assert clues["y1_mean"] == pytest.approx(0.6)
    assert clues["y1_std"] == pytest.approx(0.8)
    assert clues["separation"] == pytest.approx(0.6 - (-0.4))
    assert clues["alpha_per_std"] == pytest.approx(2.5)
    assert clues["alpha_per_std"] == pytest.approx(clues["proj_std"])


def test_projection_targets():
    stats, scales = _make_inputs(y1_mean=0.6, y1_std=0.8)
    clues = iid_mm_steering_clues(stats, scales, "_overall", "iid_mm", 12)

    assert clues["target_y1_minus_half_class_std"] == pytest.approx(0.6 - 0.5 * 0.8)
    assert clues["target_y1"] == pytest.approx(0.6)
    assert clues["target_y1_plus_half_class_std"] == pytest.approx(0.6 + 0.5 * 0.8)
    assert clues["target_y1_plus_one_class_std"] == pytest.approx(0.6 + 1.0 * 0.8)


def test_per_conflict_direction_name():
    cid = "json_only_vs_plain"
    stats, scales = _make_inputs(
        constraint=cid,
        direction_name=f"iid_mm_{cid}",
        layer=8,
        proj_std=1.25,
        y1_mean=0.1,
        y1_std=0.5,
    )
    clues = iid_mm_steering_clues(stats, scales, cid, f"iid_mm_{cid}", 8)
    assert clues["proj_std"] == pytest.approx(1.25)
    assert clues["alpha_per_std"] == pytest.approx(1.25)
    assert clues["target_y1_plus_one_class_std"] == pytest.approx(0.1 + 0.5)


def test_negative_separation():
    stats, scales = _make_inputs(y0_mean=0.5, y1_mean=-0.2, y1_std=0.4)
    clues = iid_mm_steering_clues(stats, scales, "_overall", "iid_mm", 12)
    assert clues["separation"] == pytest.approx(-0.7)
    # alpha_per_std is always proj_std, sign-agnostic
    assert clues["alpha_per_std"] > 0


def test_keyerror_when_scale_missing():
    stats, scales = _make_inputs(direction_name="iid_mm", layer=12)
    # Ask for a direction not present in scales
    with pytest.raises(KeyError) as exc_info:
        iid_mm_steering_clues(stats, scales, "_overall", "iid_mm_mean", 12)
    assert "iid_mm_mean_L12" in str(exc_info.value)


def test_keyerror_when_stats_constraint_missing():
    stats, scales = _make_inputs()
    # Add the scale entry so the missing-stats path is reached
    scales["iid_mm_L12"] = 2.5
    with pytest.raises(KeyError):
        iid_mm_steering_clues(stats, scales, "missing_cid", "iid_mm", 12)


def test_keyerror_when_stats_layer_missing():
    stats, scales = _make_inputs(layer=12)
    scales["iid_mm_L7"] = 2.5  # scale present at L7 but stats only at L12
    with pytest.raises(KeyError):
        iid_mm_steering_clues(stats, scales, "_overall", "iid_mm", 7)


def test_keyerror_when_stats_direction_missing():
    stats, scales = _make_inputs(direction_name="iid_mm", layer=12)
    scales["iid_mm_other_L12"] = 2.5
    with pytest.raises(KeyError):
        iid_mm_steering_clues(stats, scales, "_overall", "iid_mm_other", 12)
