"""Unit tests for support_code/behavior_summaries.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# behavior_summaries.py lives in support_code/, which is not a package.
# Add it to sys.path so we can import it directly.
sys.path.insert(0, str(Path(__file__).parents[2] / "support_code"))

import behavior_summaries  # noqa: E402


BEHAVIOR = "Jumping"


def _make_bin_data(
    latency_first_values: list,
    latency_last_values: list,
    avg_bout_durations: list | None = None,
    stats_sample_counts: list | None = None,
    mouse_id: str = "mouse_A",
) -> pd.DataFrame:
    """Build a per-bin DataFrame matching the shape expected by aggregate_data_by_bin_size."""
    n = len(latency_first_values)
    if avg_bout_durations is None:
        avg_bout_durations = [1.5] * n
    if stats_sample_counts is None:
        stats_sample_counts = [2] * n
    return pd.DataFrame(
        {
            "MouseID": [mouse_id] * n,
            f"{BEHAVIOR}_latency_to_first_prediction": latency_first_values,
            f"{BEHAVIOR}_latency_to_last_prediction": latency_last_values,
            f"{BEHAVIOR}_time_behavior": [100.0] * n,
            f"{BEHAVIOR}_time_not_behavior": [200.0] * n,
            f"{BEHAVIOR}_behavior_dist": [50.0] * n,
            f"{BEHAVIOR}_behavior_dist_threshold": [10.0] * n,
            f"{BEHAVIOR}_behavior_dist_seg": [5.0] * n,
            f"{BEHAVIOR}_bout_behavior": [2] * n,
            f"{BEHAVIOR}_avg_bout_duration": avg_bout_durations,
            f"{BEHAVIOR}__stats_sample_count": stats_sample_counts,
            f"{BEHAVIOR}_bout_duration_std": [0.1] * n,
            f"{BEHAVIOR}_bout_duration_var": [0.01] * n,
        }
    )


class TestLatencyFirstPrediction:
    """Tests for bin_first_XX.latency_first_prediction (incremental semantics)."""

    def test_single_bin_returns_value(self):
        """bin_size=1, prev_bin_size=0: returns first bin's value."""
        data = _make_bin_data(
            latency_first_values=[2506.0],
            latency_last_values=[4900.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR, prev_bin_size=0
        )
        col = f"bin_first_5.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(2506.0)

    def test_single_bin_nan_returns_nan(self):
        data = _make_bin_data(
            latency_first_values=[float("nan")],
            latency_last_values=[float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR, prev_bin_size=0
        )
        col = f"bin_first_5.{BEHAVIOR}_latency_first_prediction"
        assert math.isnan(result[col].iloc[0])

    def test_consecutive_bins_returns_incremental_value(self):
        """bin_size=2, prev_bin_size=1: should return bin 1's value (5-10min), not bin 0's."""
        data = _make_bin_data(
            latency_first_values=[2506.0, 9412.0],
            latency_last_values=[4900.0, 11000.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_first_10.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(9412.0)

    def test_non_consecutive_bins_returns_first_in_range(self):
        """bin_size=3, prev_bin_size=1: incremental window is bins 1-2 (5-15min).
        Should return first non-NaN in that range."""
        data = _make_bin_data(
            latency_first_values=[2506.0, 9412.0, 18082.0],
            latency_last_values=[4900.0, 11000.0, 19000.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_first_15.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(9412.0)

    def test_non_consecutive_skips_nan_in_range(self):
        """bin_size=3, prev_bin_size=1: bins 1-2, bin 1 is NaN → returns bin 2's value."""
        data = _make_bin_data(
            latency_first_values=[2506.0, float("nan"), 18082.0],
            latency_last_values=[4900.0, float("nan"), 19000.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_first_15.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(18082.0)

    def test_incremental_all_nan_returns_nan(self):
        """bin_size=3, prev_bin_size=1: bins 1-2 both NaN → NaN."""
        data = _make_bin_data(
            latency_first_values=[2506.0, float("nan"), float("nan")],
            latency_last_values=[4900.0, float("nan"), float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_first_15.{BEHAVIOR}_latency_first_prediction"
        assert math.isnan(result[col].iloc[0])

    def test_prev_bin_zero_returns_first_non_nan(self):
        """bin_size=3, prev_bin_size=0: full window 0-15min."""
        data = _make_bin_data(
            latency_first_values=[float("nan"), 9412.0, 18082.0],
            latency_last_values=[float("nan"), 11000.0, 19000.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=0
        )
        col = f"bin_first_15.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(9412.0)


class TestLatencyLastPrediction:
    """Tests for bin_last_XX.latency_last_prediction (incremental semantics)."""

    def test_single_bin_returns_value(self):
        data = _make_bin_data(
            latency_first_values=[2506.0],
            latency_last_values=[4900.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR, prev_bin_size=0
        )
        col = f"bin_last_5.{BEHAVIOR}_latency_last_prediction"
        assert result[col].iloc[0] == pytest.approx(4900.0)

    def test_consecutive_bins_returns_incremental_value(self):
        """bin_size=2, prev_bin_size=1: returns bin 1's last prediction."""
        data = _make_bin_data(
            latency_first_values=[2506.0, 9412.0],
            latency_last_values=[4900.0, 14863.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_last_10.{BEHAVIOR}_latency_last_prediction"
        assert result[col].iloc[0] == pytest.approx(14863.0)

    def test_non_consecutive_returns_last_non_nan(self):
        """bin_size=3, prev_bin_size=1: bins 1-2, returns last non-NaN."""
        data = _make_bin_data(
            latency_first_values=[2506.0, 9412.0, float("nan")],
            latency_last_values=[4900.0, 11000.0, float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_last_15.{BEHAVIOR}_latency_last_prediction"
        assert result[col].iloc[0] == pytest.approx(11000.0)

    def test_incremental_all_nan_returns_nan(self):
        data = _make_bin_data(
            latency_first_values=[2506.0, float("nan"), float("nan")],
            latency_last_values=[4900.0, float("nan"), float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_last_15.{BEHAVIOR}_latency_last_prediction"
        assert math.isnan(result[col].iloc[0])


class TestAvgBoutLength:
    """Tests for avg_bout_length (cumulative weighted average over all bins 0..bin_size)."""

    def test_single_bin_returns_value(self):
        data = _make_bin_data(
            latency_first_values=[100.0],
            latency_last_values=[200.0],
            avg_bout_durations=[18.8],
            stats_sample_counts=[5],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR
        )
        col = f"bin_avg_5.{BEHAVIOR}_avg_bout_length"
        assert result[col].iloc[0] == pytest.approx(18.8)

    def test_cumulative_weighted_average(self):
        """avg_bout_length is cumulative: weighted avg across all bins 0..bin_size."""
        data = _make_bin_data(
            latency_first_values=[100.0, 200.0, 300.0],
            latency_last_values=[150.0, 250.0, 350.0],
            avg_bout_durations=[10.0, 20.0, 30.0],
            stats_sample_counts=[5, 3, 4],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR, prev_bin_size=1
        )
        col = f"bin_avg_15.{BEHAVIOR}_avg_bout_length"
        # Weighted avg = (10*5 + 20*3 + 30*4) / (5+3+4) = 230/12 ≈ 19.1667
        expected = np.average([10.0, 20.0, 30.0], weights=[5, 3, 4])
        assert result[col].iloc[0] == pytest.approx(expected)

    def test_returns_nan_when_all_bins_no_behavior(self):
        data = _make_bin_data(
            latency_first_values=[float("nan"), float("nan")],
            latency_last_values=[float("nan"), float("nan")],
            avg_bout_durations=[0.0, 0.0],
            stats_sample_counts=[0, 0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR
        )
        col = f"bin_avg_10.{BEHAVIOR}_avg_bout_length"
        assert math.isnan(result[col].iloc[0])

    def test_skips_bins_with_no_behavior_in_weighted_avg(self):
        """Bins with sample_count=0 have zero weight and don't affect the average."""
        data = _make_bin_data(
            latency_first_values=[100.0, float("nan"), 300.0],
            latency_last_values=[150.0, float("nan"), 350.0],
            avg_bout_durations=[10.0, 0.0, 30.0],
            stats_sample_counts=[5, 0, 4],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR
        )
        col = f"bin_avg_15.{BEHAVIOR}_avg_bout_length"
        # Weighted avg = (10*5 + 30*4) / (5+4) = 170/9 ≈ 18.889
        expected = np.average([10.0, 30.0], weights=[5, 4])
        assert result[col].iloc[0] == pytest.approx(expected)


class TestMultiMouseAlignment:
    def test_each_mouse_gets_own_latency(self):
        """Each mouse gets its own incremental latency values."""
        mouse_a = _make_bin_data(
            latency_first_values=[2506.0, 9412.0],
            latency_last_values=[4900.0, 11000.0],
            mouse_id="mouse_A",
        )
        mouse_b = _make_bin_data(
            latency_first_values=[3000.0, float("nan")],
            latency_last_values=[5000.0, float("nan")],
            mouse_id="mouse_B",
        )
        data = pd.concat([mouse_a, mouse_b], ignore_index=True)
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR, prev_bin_size=1
        )
        result = result.set_index("MouseID")

        first_col = f"bin_first_10.{BEHAVIOR}_latency_first_prediction"
        last_col = f"bin_last_10.{BEHAVIOR}_latency_last_prediction"

        assert result.loc["mouse_A", first_col] == pytest.approx(9412.0)
        assert math.isnan(result.loc["mouse_B", first_col])

        assert result.loc["mouse_A", last_col] == pytest.approx(11000.0)
        assert math.isnan(result.loc["mouse_B", last_col])

    def test_each_mouse_gets_own_avg_bout_length(self):
        """Each mouse gets its own cumulative weighted avg_bout_length."""
        mouse_a = _make_bin_data(
            latency_first_values=[100.0, 200.0],
            latency_last_values=[150.0, 250.0],
            avg_bout_durations=[10.0, 20.0],
            stats_sample_counts=[3, 5],
            mouse_id="mouse_A",
        )
        mouse_b = _make_bin_data(
            latency_first_values=[300.0, 400.0],
            latency_last_values=[350.0, 450.0],
            avg_bout_durations=[7.0, 0.0],
            stats_sample_counts=[2, 0],
            mouse_id="mouse_B",
        )
        data = pd.concat([mouse_a, mouse_b], ignore_index=True)
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR
        )
        result = result.set_index("MouseID")

        col = f"bin_avg_10.{BEHAVIOR}_avg_bout_length"
        # Mouse A: (10*3 + 20*5) / (3+5) = 130/8 = 16.25
        assert result.loc["mouse_A", col] == pytest.approx(16.25)
        # Mouse B: only bin 0 has behavior → 7.0
        assert result.loc["mouse_B", col] == pytest.approx(7.0)
