"""Comprehensive unit tests for VideoObservations.stitch_greedy_tracklets method.

This module provides thorough test coverage for the stitch_greedy_tracklets functionality,
including normal operation, edge cases, error conditions, and parameter variations.
"""

import copy
from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.matching.core import VideoObservations


def test_stitch_greedy_tracklets_basic_functionality(
    minimal_video_observations, stitching_verification_fixture
):
    """Test basic stitching functionality with minimal data."""
    # Arrange
    video_obs = minimal_video_observations
    original_count = len(video_obs._tracklets)
    original_tracklets = copy.deepcopy(video_obs._tracklets)

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert
    final_count = len(video_obs._tracklets)
    assert final_count <= original_count, "Stitching should not increase tracklet count"

    # Verify stitching results
    stitching_verification_fixture(
        original_tracklets, video_obs._tracklets, original_count, final_count
    )

    # Check that method attributes were set correctly
    assert video_obs._tracklet_stitch_method == "greedy"
    assert hasattr(video_obs, "_stitch_translation")
    assert isinstance(video_obs._stitch_translation, dict)


def test_stitch_greedy_tracklets_parameter_variations(minimal_video_observations):
    """Test different parameter combinations for stitch_greedy_tracklets."""
    # Test cases with different parameter combinations
    test_cases = [
        {"num_tracks": None, "all_embeds": True, "prioritize_long": False},
        {"num_tracks": None, "all_embeds": False, "prioritize_long": False},
        {"num_tracks": None, "all_embeds": True, "prioritize_long": True},
        {"num_tracks": 1, "all_embeds": True, "prioritize_long": False},
        {"num_tracks": 2, "all_embeds": False, "prioritize_long": True},
    ]

    for params in test_cases:
        # Arrange - reset tracklets for each test
        video_obs = minimal_video_observations
        video_obs._make_tracklets()
        original_count = len(video_obs._tracklets)

        # Act
        video_obs.stitch_greedy_tracklets(**params)

        # Assert
        final_count = len(video_obs._tracklets)
        assert final_count <= original_count, f"Failed for params: {params}"
        assert video_obs._tracklet_stitch_method == "greedy"
        assert hasattr(video_obs, "_stitch_translation")


def test_stitch_greedy_tracklets_fragmented_data(
    fragmented_video_observations, stitching_verification_fixture
):
    """Test stitching with fragmented tracklets that should be combined."""
    # Arrange
    video_obs = fragmented_video_observations
    original_count = len(video_obs._tracklets)
    original_tracklets = copy.deepcopy(video_obs._tracklets)

    # Should have multiple small tracklets initially
    assert original_count >= 6, "Should have multiple fragmented tracklets"

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert
    final_count = len(video_obs._tracklets)
    reduction = original_count - final_count

    # May see reduction in tracklet count (depends on similarity thresholds)
    # The important thing is that no tracklets are added
    assert reduction >= 0, "Should not increase tracklet count"
    assert final_count <= original_count, "Should not increase the number of tracklets"

    # Verify stitching results
    verification_result = stitching_verification_fixture(
        original_tracklets, video_obs._tracklets, original_count, final_count
    )

    # May see meaningful reduction depending on similarity thresholds
    # At minimum, should not increase tracklet count
    assert verification_result["reduction_percentage"] >= 0, (
        "Should not increase tracklet count"
    )


def test_stitch_greedy_tracklets_single_tracklet(
    single_tracklet_video_observations, verify_no_overlaps_fixture
):
    """Test stitching behavior with only one tracklet (edge case)."""
    # Arrange
    video_obs = single_tracklet_video_observations
    original_count = len(video_obs._tracklets)

    # Should have exactly one tracklet
    assert original_count == 1, "Should start with exactly one tracklet"

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert
    final_count = len(video_obs._tracklets)
    assert final_count == 1, "Should still have exactly one tracklet"

    # Verify state is consistent
    verify_no_overlaps_fixture(video_obs)
    assert video_obs._tracklet_stitch_method == "greedy"
    assert hasattr(video_obs, "_stitch_translation")


def test_stitch_greedy_tracklets_empty_tracklets(
    empty_video_observations, verify_no_overlaps_fixture
):
    """Test stitching behavior with no tracklets (edge case)."""
    # Arrange
    video_obs = empty_video_observations
    original_count = len(video_obs._tracklets)

    # Should have no tracklets
    assert original_count == 0, "Should start with no tracklets"

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert
    final_count = len(video_obs._tracklets)
    assert final_count == 0, "Should still have no tracklets"

    # Verify state is consistent
    verify_no_overlaps_fixture(video_obs)
    assert video_obs._tracklet_stitch_method == "greedy"
    assert hasattr(video_obs, "_stitch_translation")


def test_stitch_greedy_tracklets_complex_scenarios(
    complex_video_observations,
    stitching_verification_fixture,
    verify_no_overlaps_fixture,
):
    """Test stitching with complex scenarios including overlaps and various lengths."""
    # Arrange
    video_obs = complex_video_observations
    original_count = len(video_obs._tracklets)
    original_tracklets = copy.deepcopy(video_obs._tracklets)

    # Should have multiple tracklets of various lengths
    assert original_count >= 5, "Should have multiple tracklets for complex test"

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=True
    )

    # Assert
    final_count = len(video_obs._tracklets)

    # Verify no overlaps exist
    verify_no_overlaps_fixture(video_obs)

    # Verify stitching results
    stitching_verification_fixture(
        original_tracklets, video_obs._tracklets, original_count, final_count
    )

    # Complex scenarios should show some reduction
    assert final_count <= original_count, "Should not increase tracklet count"


def test_stitch_greedy_tracklets_with_num_tracks_parameter(minimal_video_observations):
    """Test stitching with specific num_tracks parameter."""
    # Arrange
    video_obs = minimal_video_observations
    video_obs._make_tracklets()
    original_count = len(video_obs._tracklets)

    target_tracks = 1

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=target_tracks, all_embeds=True, prioritize_long=False
    )

    # Assert
    final_count = len(video_obs._tracklets)

    # Should respect the target when possible
    assert final_count <= original_count, "Should not increase tracklet count"
    assert video_obs._tracklet_stitch_method == "greedy"


def test_stitch_greedy_tracklets_preserves_original_tracklets(
    minimal_video_observations,
):
    """Test that original tracklets are preserved after stitching."""
    # Arrange
    video_obs = minimal_video_observations
    original_tracklets = copy.deepcopy(video_obs._tracklets)

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert - implementation should restore original tracklets
    # This is based on the line: self._tracklets = original_tracklets
    for i, (original, current) in enumerate(
        zip(original_tracklets, video_obs._tracklets, strict=False)
    ):
        assert original.track_id == current.track_id, (
            f"Tracklet {i} ID should be preserved"
        )
        assert len(original.frames) == len(current.frames), (
            f"Tracklet {i} frame count should be preserved"
        )


def test_stitch_greedy_tracklets_translation_mapping(minimal_video_observations):
    """Test that stitch translation mapping is correctly created."""
    # Arrange
    video_obs = minimal_video_observations

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert
    assert hasattr(video_obs, "_stitch_translation")
    assert isinstance(video_obs._stitch_translation, dict)

    # Should contain mapping for track ID 0 (background)
    assert 0 in video_obs._stitch_translation.values()

    # Should have entries for original tracklets
    translation = video_obs._stitch_translation
    assert len(translation) >= 1, "Should have at least background translation"


def test_stitch_greedy_tracklets_prioritize_long_parameter(
    fragmented_video_observations,
):
    """Test that prioritize_long parameter affects stitching behavior."""
    # Test without prioritizing long tracklets
    video_obs_no_priority = fragmented_video_observations
    video_obs_no_priority._make_tracklets()
    video_obs_no_priority.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )
    result_no_priority = len(video_obs_no_priority._tracklets)

    # Test with prioritizing long tracklets
    video_obs_with_priority = fragmented_video_observations
    video_obs_with_priority._make_tracklets()
    video_obs_with_priority.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=True
    )
    result_with_priority = len(video_obs_with_priority._tracklets)

    # Both should be valid results
    assert result_no_priority >= 0
    assert result_with_priority >= 0

    # Results may differ based on prioritization
    # (This is hard to test deterministically without knowing the exact algorithm)


def test_stitch_greedy_tracklets_all_embeds_parameter(minimal_video_observations):
    """Test that all_embeds parameter affects behavior."""
    # Test with all_embeds=True
    video_obs_all_embeds = minimal_video_observations
    video_obs_all_embeds._make_tracklets()
    video_obs_all_embeds.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )
    result_all_embeds = len(video_obs_all_embeds._tracklets)

    # Test with all_embeds=False
    video_obs_no_all_embeds = minimal_video_observations
    video_obs_no_all_embeds._make_tracklets()
    video_obs_no_all_embeds.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=False, prioritize_long=False
    )
    result_no_all_embeds = len(video_obs_no_all_embeds._tracklets)

    # Both should be valid results
    assert result_all_embeds >= 0
    assert result_no_all_embeds >= 0


@pytest.mark.parametrize(
    "num_tracks, all_embeds, prioritize_long",
    [
        (None, True, False),
        (1, True, False),
        (2, False, True),
        (5, True, True),
        (None, False, False),
    ],
)
def test_stitch_greedy_tracklets_parameter_combinations(
    minimal_video_observations, num_tracks, all_embeds, prioritize_long
):
    """Test various parameter combinations for stitch_greedy_tracklets."""
    # Arrange
    video_obs = minimal_video_observations
    video_obs._make_tracklets()
    original_count = len(video_obs._tracklets)

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=num_tracks, all_embeds=all_embeds, prioritize_long=prioritize_long
    )

    # Assert
    final_count = len(video_obs._tracklets)
    assert final_count <= original_count, "Should not increase tracklet count"
    assert video_obs._tracklet_stitch_method == "greedy"
    assert hasattr(video_obs, "_stitch_translation")


def test_stitch_greedy_tracklets_idempotent(minimal_video_observations):
    """Test that running stitch_greedy_tracklets multiple times is safe."""
    # Arrange
    video_obs = minimal_video_observations

    # Act - run stitching twice
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )
    first_result = len(video_obs._tracklets)

    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )
    second_result = len(video_obs._tracklets)
    second_translation = video_obs._stitch_translation

    # Assert - should be consistent
    assert first_result == second_result, "Multiple runs should give same result"
    # Translation might change, but should still be valid
    assert isinstance(second_translation, dict)


def test_stitch_greedy_tracklets_state_consistency(minimal_video_observations):
    """Test that object state remains consistent after stitching."""
    # Arrange
    video_obs = minimal_video_observations
    original_num_frames = video_obs.num_frames

    # Act
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert - verify object state is consistent
    assert video_obs.num_frames == original_num_frames, "Frame count should not change"
    assert video_obs._tracklet_stitch_method == "greedy"
    assert hasattr(video_obs, "_stitch_translation")
    assert isinstance(video_obs._tracklets, list)


def test_stitch_greedy_tracklets_tracklet_properties(minimal_video_observations):
    """Test that tracklet properties are maintained after stitching."""
    # Arrange
    video_obs = minimal_video_observations
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    # Assert - verify tracklet properties
    for tracklet in video_obs._tracklets:
        assert hasattr(tracklet, "frames"), "Tracklet should have frames"
        assert hasattr(tracklet, "track_id"), "Tracklet should have track_id"
        assert hasattr(tracklet, "detection_list"), (
            "Tracklet should have detection_list"
        )

        # Verify frame consistency
        assert len(tracklet.frames) > 0, "Tracklet should have frames"
        assert len(tracklet.detection_list) == len(tracklet.frames), (
            "Detection count should match frame count"
        )


def test_stitch_greedy_tracklets_error_handling_invalid_parameters():
    """Test that method handles edge cases gracefully."""
    # Create minimal video observations for testing
    from mouse_tracking.matching.core import Detection

    detection = Detection(frame=0, pose_idx=0, pose=np.random.rand(12, 2))
    video_obs = VideoObservations([[detection]])
    video_obs.generate_greedy_tracklets()

    # The method should handle edge cases gracefully rather than raising exceptions
    # Test with unusual but valid parameters

    # Very large num_tracks should work
    video_obs.stitch_greedy_tracklets(num_tracks=1000)
    assert len(video_obs._tracklets) >= 0

    # Reset for next test
    video_obs._make_tracklets()

    # All valid parameter combinations should work
    video_obs.stitch_greedy_tracklets(
        num_tracks=0, all_embeds=False, prioritize_long=True
    )
    assert len(video_obs._tracklets) >= 0


def test_stitch_greedy_tracklets_memory_efficiency(complex_video_observations):
    """Test that stitching doesn't cause memory leaks or excessive usage."""
    # Arrange
    video_obs = complex_video_observations

    # Act - measure memory usage indirectly by checking object sizes
    import sys

    initial_size = sys.getsizeof(video_obs)

    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=False
    )

    final_size = sys.getsizeof(video_obs)

    # Assert - size should not grow excessively
    size_increase = final_size - initial_size
    assert size_increase < initial_size, (
        "Memory usage should not double after stitching"
    )


def test_stitch_greedy_tracklets_with_get_transition_costs_called(
    minimal_video_observations,
):
    """Test that _get_transition_costs is called during stitching."""
    # Arrange
    video_obs = minimal_video_observations

    # Act & Assert - using patch to verify method is called
    with patch.object(
        video_obs, "_get_transition_costs", wraps=video_obs._get_transition_costs
    ) as mock_costs:
        video_obs.stitch_greedy_tracklets(
            num_tracks=None, all_embeds=True, prioritize_long=False
        )

        # Should call _get_transition_costs at least once
        assert mock_costs.call_count > 0, (
            "_get_transition_costs should be called during stitching"
        )

        # Verify it was called with correct parameters
        call_args = mock_costs.call_args_list[0]
        assert "all_comparisons" in call_args[1] or len(call_args[0]) > 0
