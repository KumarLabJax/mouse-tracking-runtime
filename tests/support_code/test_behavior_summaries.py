"""Unit tests for support_code/behavior_summaries.py."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# behavior_summaries.py lives in support_code/, which is not a package.
# Add it to sys.path so we can import it directly.
sys.path.insert(0, str(Path(__file__).parents[2] / "support_code"))

import behavior_summaries

BEHAVIOR = "Jumping"


def _make_filtered_data(
    latency_first_values: list,
    latency_last_values: list,
    mouse_id: str = "mouse_A",
) -> pd.DataFrame:
    """Build a minimal per-bin DataFrame matching the shape expected by aggregate_data_by_bin_size."""
    n = len(latency_first_values)
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
            f"{BEHAVIOR}_avg_bout_duration": [1.5] * n,
            f"{BEHAVIOR}__stats_sample_count": [2] * n,
            f"{BEHAVIOR}_bout_duration_std": [0.1] * n,
            f"{BEHAVIOR}_bout_duration_var": [0.01] * n,
        }
    )


class TestLatencyFirstPrediction:
    """Tests for latency_first_prediction aggregation."""

    def test_returns_first_bin_value_when_present(self):
        """latency_first should be the first bin's value, not a cumulative sum."""
        data = _make_filtered_data(
            latency_first_values=[2506.0, 9412.0, 18082.0, float("nan")],
            latency_last_values=[4900.0, 11000.0, 19000.0, float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=4, behavior=BEHAVIOR
        )
        col = f"bin_first_20.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(2506.0)

    def test_returns_nan_when_first_bin_has_no_behavior(self):
        """latency_first should be NaN when the first bin has no behavior, not a later bin's value."""
        data = _make_filtered_data(
            latency_first_values=[float("nan"), 5000.0, 12000.0, float("nan")],
            latency_last_values=[float("nan"), 8000.0, 15000.0, float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=4, behavior=BEHAVIOR
        )
        col = f"bin_first_20.{BEHAVIOR}_latency_first_prediction"
        assert math.isnan(result[col].iloc[0])

    def test_single_bin_returns_that_bins_value(self):
        """Single bin should return that bin's latency_first value."""
        data = _make_filtered_data(
            latency_first_values=[2506.0],
            latency_last_values=[4900.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR
        )
        col = f"bin_first_5.{BEHAVIOR}_latency_first_prediction"
        assert result[col].iloc[0] == pytest.approx(2506.0)


class TestLatencyLastPrediction:
    """Tests for latency_last_prediction aggregation."""

    def test_returns_last_bin_value_when_present(self):
        """latency_last should be the last bin's value, not a cumulative sum."""
        data = _make_filtered_data(
            latency_first_values=[2506.0, 9412.0, 18082.0, 38222.0],
            latency_last_values=[4900.0, 11000.0, 19000.0, 45000.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=4, behavior=BEHAVIOR
        )
        col = f"bin_last_20.{BEHAVIOR}_latency_last_prediction"
        assert result[col].iloc[0] == pytest.approx(45000.0)

    def test_returns_nan_when_last_bin_has_no_behavior(self):
        """latency_last should be NaN when the last bin has no behavior, not a previous bin's value."""
        data = _make_filtered_data(
            latency_first_values=[float("nan"), 5000.0, 12000.0, float("nan")],
            latency_last_values=[float("nan"), 8000.0, 15000.0, float("nan")],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=4, behavior=BEHAVIOR
        )
        col = f"bin_last_20.{BEHAVIOR}_latency_last_prediction"
        assert math.isnan(result[col].iloc[0])

    def test_single_bin_returns_that_bins_value(self):
        """Single bin should return that bin's latency_last value."""
        data = _make_filtered_data(
            latency_first_values=[2506.0],
            latency_last_values=[4900.0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR
        )
        col = f"bin_last_5.{BEHAVIOR}_latency_last_prediction"
        assert result[col].iloc[0] == pytest.approx(4900.0)


def _make_per_bin_data(
    avg_bout_durations: list,
    stats_sample_counts: list,
    mouse_id: str = "mouse_A",
) -> pd.DataFrame:
    """Build a per-bin DataFrame with varying avg_bout_duration and sample counts."""
    n = len(avg_bout_durations)
    return pd.DataFrame(
        {
            "MouseID": [mouse_id] * n,
            f"{BEHAVIOR}_latency_to_first_prediction": [100.0] * n,
            f"{BEHAVIOR}_latency_to_last_prediction": [200.0] * n,
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


class TestAvgBoutLength:
    """Tests for avg_bout_length aggregation."""

    def test_single_bin_returns_that_bins_value(self):
        """Single bin should return that bin's avg_bout_duration."""
        data = _make_per_bin_data(
            avg_bout_durations=[18.8],
            stats_sample_counts=[5],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=1, behavior=BEHAVIOR
        )
        col = f"bin_avg_5.{BEHAVIOR}_avg_bout_length"
        assert result[col].iloc[0] == pytest.approx(18.8)

    def test_multi_bin_returns_last_bin_value_not_sum(self):
        """avg_bout_length should be the last bin's value, not a cumulative sum."""
        data = _make_per_bin_data(
            avg_bout_durations=[10.0, 20.0, 30.0],
            stats_sample_counts=[5, 3, 4],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=3, behavior=BEHAVIOR
        )
        col = f"bin_avg_15.{BEHAVIOR}_avg_bout_length"
        # Should be 30.0 (last bin), NOT 60.0 (sum of 10+20+30)
        assert result[col].iloc[0] == pytest.approx(30.0)

    def test_returns_nan_when_last_bin_has_no_behavior(self):
        """Should return NaN when the last bin has no behavior."""
        data = _make_per_bin_data(
            avg_bout_durations=[18.0, 0.0],
            stats_sample_counts=[4, 0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR
        )
        col = f"bin_avg_10.{BEHAVIOR}_avg_bout_length"
        assert math.isnan(result[col].iloc[0])

    def test_returns_nan_when_all_bins_have_no_behavior(self):
        """Should return NaN when all bins have no behavior."""
        data = _make_per_bin_data(
            avg_bout_durations=[0.0, 0.0],
            stats_sample_counts=[0, 0],
        )
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR
        )
        col = f"bin_avg_10.{BEHAVIOR}_avg_bout_length"
        assert math.isnan(result[col].iloc[0])


class TestMultiMouseAlignment:
    """Tests for multi-mouse alignment in aggregation."""

    def test_each_mouse_gets_its_own_first_latency(self):
        """With multiple mice, each should receive their own first-bin latency value."""
        mouse_a = _make_filtered_data(
            latency_first_values=[2506.0, 9412.0],
            latency_last_values=[4900.0, 11000.0],
            mouse_id="mouse_A",
        )
        mouse_b = _make_filtered_data(
            latency_first_values=[float("nan"), 5000.0],
            latency_last_values=[float("nan"), 8000.0],
            mouse_id="mouse_B",
        )
        data = pd.concat([mouse_a, mouse_b], ignore_index=True)
        result = behavior_summaries.aggregate_data_by_bin_size(
            data, bin_size=2, behavior=BEHAVIOR
        )
        result = result.set_index("MouseID")

        first_col = f"bin_first_10.{BEHAVIOR}_latency_first_prediction"
        last_col = f"bin_last_10.{BEHAVIOR}_latency_last_prediction"

        assert result.loc["mouse_A", first_col] == pytest.approx(2506.0)
        assert math.isnan(result.loc["mouse_B", first_col])

        assert result.loc["mouse_A", last_col] == pytest.approx(11000.0)
        assert result.loc["mouse_B", last_col] == pytest.approx(8000.0)

    def test_each_mouse_gets_its_own_avg_bout_length(self):
        """Each mouse should get its own last-bin avg_bout_length, not a shared scalar."""
        mouse_a = _make_per_bin_data(
            avg_bout_durations=[10.0, 20.0],
            stats_sample_counts=[3, 5],
            mouse_id="mouse_A",
        )
        mouse_b = _make_per_bin_data(
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
        assert result.loc["mouse_A", col] == pytest.approx(20.0)
        assert math.isnan(result.loc["mouse_B", col])
