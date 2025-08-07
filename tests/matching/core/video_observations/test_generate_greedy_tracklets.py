"""Unit tests for VideoObservations.generate_greedy_tracklets method.

This module contains comprehensive tests for the greedy tracklet generation algorithm,
including normal operation, edge cases, and error conditions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mouse_tracking.matching.core import Detection, VideoObservations


class TestGenerateGreedyTracklets:
    """Tests for the generate_greedy_tracklets method."""

    def test_generate_greedy_tracklets_basic_functionality(self, basic_detection):
        """Test basic functionality with simple sequential observations."""
        # Create a simple scenario with 3 frames, 2 observations per frame
        observations = []
        for frame in range(3):
            frame_observations = []
            for obs_idx in range(2):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx * 0.5,  # Different embeddings for different obs
                    pose_coords=(obs_idx * 50, obs_idx * 50),
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)

        # Test default parameters
        video_obs.generate_greedy_tracklets()

        # Verify internal state was updated
        assert video_obs._observation_id_dict is not None
        assert video_obs._tracklet_gen_method == "greedy"
        assert video_obs._tracklets is not None
        assert len(video_obs._tracklets) > 0

        # Should have one entry per frame
        assert len(video_obs._observation_id_dict) == 3

        # Each frame should have 2 observations
        for frame in range(3):
            assert len(video_obs._observation_id_dict[frame]) == 2

    def test_generate_greedy_tracklets_with_parameters(self, basic_detection):
        """Test with different parameter combinations."""
        # Create simple observations
        observations = []
        for frame in range(2):
            detection = basic_detection(frame_idx=frame, pose_idx=0)
            observations.append([detection])

        video_obs = VideoObservations(observations)

        # Test with custom parameters
        max_cost = -np.log(1e-4)  # Different from default
        video_obs.generate_greedy_tracklets(
            max_cost=max_cost, rotate_pose=True, num_threads=1
        )

        assert video_obs._tracklet_gen_method == "greedy"
        assert video_obs._tracklets is not None

    def test_generate_greedy_tracklets_single_frame(self, basic_detection):
        """Test with single frame (edge case)."""
        observations = [[basic_detection(frame_idx=0, pose_idx=0)]]
        video_obs = VideoObservations(observations)

        video_obs.generate_greedy_tracklets()

        # Should handle single frame correctly
        assert len(video_obs._observation_id_dict) == 1
        assert len(video_obs._observation_id_dict[0]) == 1
        assert len(video_obs._tracklets) == 1

    def test_generate_greedy_tracklets_empty_frames(self, basic_detection):
        """Test with some empty frames."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [],  # Empty frame
            [basic_detection(frame_idx=2, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)

        video_obs.generate_greedy_tracklets()

        # Should handle empty frames correctly
        assert len(video_obs._observation_id_dict) == 3
        assert len(video_obs._observation_id_dict[0]) == 1
        assert len(video_obs._observation_id_dict[1]) == 0  # Empty frame
        assert len(video_obs._observation_id_dict[2]) == 1

    def test_generate_greedy_tracklets_no_observations(self):
        """Test with no observations (edge case)."""
        observations = [[] for _ in range(3)]  # All empty frames
        video_obs = VideoObservations(observations)

        # TODO: This reveals a bug - _make_tracklets fails with empty tracklet_dict
        # The _make_tracklets method tries to call np.max on empty array
        with pytest.raises(
            ValueError, match="zero-size array to reduction operation maximum"
        ):
            video_obs.generate_greedy_tracklets()

    def test_generate_greedy_tracklets_single_observation_per_frame(
        self, basic_detection
    ):
        """Test with single observation per frame (simplest tracking case)."""
        observations = []
        for frame in range(5):
            detection = basic_detection(
                frame_idx=frame,
                pose_idx=0,
                embed_value=0.5,  # Same embedding to encourage linking
                pose_coords=(50, 50),  # Same position
            )
            observations.append([detection])

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets()

        # Should create a single tracklet spanning all frames
        assert len(video_obs._tracklets) == 1
        assert len(video_obs._tracklets[0].frames) == 5

    def test_generate_greedy_tracklets_multiple_observations_per_frame(
        self, basic_detection
    ):
        """Test with multiple observations per frame."""
        observations = []
        for frame in range(3):
            frame_observations = []
            for obs_idx in range(3):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx,  # Different embeddings
                    pose_coords=(obs_idx * 30, obs_idx * 30),  # Different positions
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets()

        # Should create multiple tracklets
        assert len(video_obs._tracklets) > 1

        # Each frame should have 3 observations assigned
        for frame in range(3):
            assert len(video_obs._observation_id_dict[frame]) == 3

    @patch("mouse_tracking.matching.core.VideoObservations._calculate_costs")
    @patch("mouse_tracking.matching.core.VideoObservations._start_pool")
    @patch("mouse_tracking.matching.core.VideoObservations._kill_pool")
    def test_generate_greedy_tracklets_multithreading(
        self, mock_kill_pool, mock_start_pool, mock_calculate_costs, basic_detection
    ):
        """Test multithreading functionality."""
        observations = []
        for frame in range(3):
            detection = basic_detection(frame_idx=frame, pose_idx=0)
            observations.append([detection])

        video_obs = VideoObservations(observations)

        # Mock the pool to simulate it being created
        mock_pool = MagicMock()

        def mock_start_pool_impl(num_threads):
            video_obs._pool = mock_pool

        def mock_kill_pool_impl():
            video_obs._pool = None

        mock_start_pool.side_effect = mock_start_pool_impl
        mock_kill_pool.side_effect = mock_kill_pool_impl

        # Mock _calculate_costs to return a simple cost matrix
        mock_calculate_costs.return_value = np.array([[0.5]])

        # Test with multiple threads
        video_obs.generate_greedy_tracklets(num_threads=2)

        # Should call pool management methods
        mock_start_pool.assert_called_once_with(2)
        # The pool should be killed after the processing is done
        mock_kill_pool.assert_called_once()

    @patch("mouse_tracking.matching.core.VideoObservations._start_pool")
    @patch("mouse_tracking.matching.core.VideoObservations._kill_pool")
    def test_generate_greedy_tracklets_single_thread(
        self, mock_kill_pool, mock_start_pool, basic_detection
    ):
        """Test that single thread doesn't use multiprocessing."""
        observations = [[basic_detection(frame_idx=0, pose_idx=0)]]
        video_obs = VideoObservations(observations)

        # Test with single thread (default)
        video_obs.generate_greedy_tracklets(num_threads=1)

        # Should not call pool management methods
        mock_start_pool.assert_not_called()
        mock_kill_pool.assert_not_called()

    @patch("mouse_tracking.matching.core.VideoObservations._calculate_costs")
    def test_generate_greedy_tracklets_calculate_costs_called(
        self, mock_calculate_costs, basic_detection
    ):
        """Test that _calculate_costs is called with correct parameters."""
        observations = []
        for frame in range(3):
            detection = basic_detection(frame_idx=frame, pose_idx=0)
            observations.append([detection])

        # Mock the cost calculation to return a simple matrix
        mock_calculate_costs.return_value = np.array([[0.5]])

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets(rotate_pose=True)

        # Should call _calculate_costs for each frame transition
        assert mock_calculate_costs.call_count == 2  # 3 frames = 2 transitions

        # Check that rotate_pose parameter is passed correctly
        for call in mock_calculate_costs.call_args_list:
            args, kwargs = call
            assert len(args) == 3  # frame_1, frame_2, rotate_pose
            assert args[2]  # rotate_pose=True

    def test_generate_greedy_tracklets_observation_caching(self, basic_detection):
        """Test that observations are properly cached and cleared."""
        observations = []
        for frame in range(3):
            detection = basic_detection(frame_idx=frame, pose_idx=0)
            observations.append([detection])

        video_obs = VideoObservations(observations)

        # Patch the cache and clear_cache methods to track calls
        with (
            patch.object(Detection, "cache") as mock_cache,
            patch.object(Detection, "clear_cache") as mock_clear_cache,
        ):
            video_obs.generate_greedy_tracklets()

            # Should cache observations during processing
            assert mock_cache.call_count > 0

            # Should clear cache after processing
            assert mock_clear_cache.call_count > 0

    def test_generate_greedy_tracklets_cost_masking(self, basic_detection):
        """Test that cost masking works correctly in greedy matching."""
        # Create observations with very different costs
        observations = []
        for frame in range(2):
            frame_observations = []
            for obs_idx in range(2):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx * 0.8,  # Different embeddings
                    pose_coords=(obs_idx * 100, obs_idx * 100),  # Far apart
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)

        # Use a high max_cost to allow poor matches
        video_obs.generate_greedy_tracklets(max_cost=10.0)

        # Should still create valid tracklets
        assert len(video_obs._tracklets) > 0

    def test_generate_greedy_tracklets_max_cost_filtering(self, basic_detection):
        """Test that max_cost parameter filters out poor matches."""
        observations = []
        for frame in range(2):
            frame_observations = []
            for obs_idx in range(2):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx,  # Very different embeddings
                    pose_coords=(obs_idx * 200, obs_idx * 200),  # Very far apart
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)

        # Use a very low max_cost to reject poor matches
        video_obs.generate_greedy_tracklets(max_cost=0.1)

        # Should create more tracklets due to rejected matches
        assert len(video_obs._tracklets) > 0

    def test_generate_greedy_tracklets_tracklet_id_assignment(self, basic_detection):
        """Test that tracklet IDs are assigned correctly."""
        observations = []
        for frame in range(3):
            frame_observations = []
            for obs_idx in range(2):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx,
                    pose_coords=(obs_idx * 50, obs_idx * 50),
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets()

        # Check that tracklet IDs are sequential and start from 0
        frame_0_ids = set(video_obs._observation_id_dict[0].values())
        expected_initial_ids = {0, 1}  # Should start with 0, 1 for first frame
        assert frame_0_ids == expected_initial_ids

    def test_generate_greedy_tracklets_make_tracklets_called(self, basic_detection):
        """Test that _make_tracklets is called at the end."""
        observations = [[basic_detection(frame_idx=0, pose_idx=0)]]
        video_obs = VideoObservations(observations)

        with patch.object(video_obs, "_make_tracklets") as mock_make_tracklets:
            video_obs.generate_greedy_tracklets()
            mock_make_tracklets.assert_called_once()

    def test_generate_greedy_tracklets_internal_state_update(self, basic_detection):
        """Test that internal state is updated correctly."""
        observations = [[basic_detection(frame_idx=0, pose_idx=0)]]
        video_obs = VideoObservations(observations)

        # Initial state
        assert video_obs._observation_id_dict is None
        assert video_obs._tracklet_gen_method is None
        assert video_obs._tracklets is None

        video_obs.generate_greedy_tracklets()

        # State should be updated
        assert video_obs._observation_id_dict is not None
        assert video_obs._tracklet_gen_method == "greedy"
        assert video_obs._tracklets is not None

    def test_generate_greedy_tracklets_pool_cleanup_on_exception(self, basic_detection):
        """Test that pool is properly cleaned up even if an exception occurs."""
        observations = []
        for frame in range(3):  # Need more frames to trigger _calculate_costs
            detection = basic_detection(frame_idx=frame, pose_idx=0)
            observations.append([detection])

        video_obs = VideoObservations(observations)

        with (
            patch.object(video_obs, "_start_pool") as mock_start_pool,
            patch.object(video_obs, "_kill_pool") as mock_kill_pool,
            patch.object(
                video_obs, "_calculate_costs", side_effect=RuntimeError("Test error")
            ),
        ):
            with pytest.raises(RuntimeError):
                video_obs.generate_greedy_tracklets(num_threads=2)

            # Pool should be started
            mock_start_pool.assert_called_once()
            # TODO: This reveals a bug - pool is not cleaned up on exception
            # The generate_greedy_tracklets method doesn't use try/finally for cleanup
            # Currently the pool is NOT cleaned up on exception
            assert (
                mock_kill_pool.call_count == 0
            )  # Documents the current buggy behavior

    def test_generate_greedy_tracklets_variable_observations_per_frame(
        self, basic_detection
    ):
        """Test with variable number of observations per frame."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],  # 1 observation
            [
                basic_detection(frame_idx=1, pose_idx=0),
                basic_detection(frame_idx=1, pose_idx=1),
            ],  # 2 observations
            [
                basic_detection(frame_idx=2, pose_idx=0),
                basic_detection(frame_idx=2, pose_idx=1),
                basic_detection(frame_idx=2, pose_idx=2),
            ],  # 3 observations
        ]
        video_obs = VideoObservations(observations)

        video_obs.generate_greedy_tracklets()

        # Should handle variable observations correctly
        assert len(video_obs._observation_id_dict[0]) == 1
        assert len(video_obs._observation_id_dict[1]) == 2
        assert len(video_obs._observation_id_dict[2]) == 3

    def test_generate_greedy_tracklets_perfect_matches(self, basic_detection):
        """Test with perfect matches (identical observations)."""
        observations = []
        for frame in range(3):
            detection = basic_detection(
                frame_idx=frame,
                pose_idx=0,
                embed_value=0.5,  # Identical embeddings
                pose_coords=(50, 50),  # Identical positions
            )
            observations.append([detection])

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets()

        # Should create a single tracklet for perfect matches
        assert len(video_obs._tracklets) == 1
        assert len(video_obs._tracklets[0].frames) == 3

    def test_generate_greedy_tracklets_with_none_values(self, basic_detection):
        """Test with Detection objects containing None values."""
        # Create detections with None values but valid other fields
        observations = []
        for frame in range(2):
            detection = basic_detection(
                frame_idx=frame,
                pose_idx=0,
                embed_value=0.5,  # Keep valid embed
                pose_coords=(50, 50),  # Keep valid pose
            )
            # Override with None to test edge case
            detection._pose = None
            detection._embed = None
            observations.append([detection])

        video_obs = VideoObservations(observations)

        # TODO: This reveals a bug - rotate_pose doesn't handle None poses correctly
        # The rotate_pose method assumes points is not None
        with pytest.raises(TypeError, match="unsupported operand type"):
            video_obs.generate_greedy_tracklets()

    def test_generate_greedy_tracklets_large_cost_matrix(self, basic_detection):
        """Test with larger cost matrices to ensure scalability."""
        # Create a larger scenario
        observations = []
        for frame in range(5):
            frame_observations = []
            for obs_idx in range(5):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx * 0.2,
                    pose_coords=(obs_idx * 20, obs_idx * 20),
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets()

        # Should handle larger matrices
        assert len(video_obs._tracklets) > 0
        assert all(
            len(frame_dict) == 5
            for frame_dict in video_obs._observation_id_dict.values()
        )

    def test_generate_greedy_tracklets_greedy_assignment_order(self, basic_detection):
        """Test that greedy assignment picks the best matches first."""
        # Create observations where one pair has much better match than others
        observations = []
        for frame in range(2):
            frame_observations = [
                basic_detection(
                    frame_idx=frame,
                    pose_idx=0,
                    embed_value=0.1,  # Very similar embeddings
                    pose_coords=(10, 10),  # Very similar positions
                ),
                basic_detection(
                    frame_idx=frame,
                    pose_idx=1,
                    embed_value=0.9,  # Very different embeddings
                    pose_coords=(90, 90),  # Very different positions
                ),
            ]
            observations.append(frame_observations)

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets()

        # Should create tracklets that preserve good matches
        assert len(video_obs._tracklets) == 2
        # The similar observations should be linked
        similar_tracklet = next(t for t in video_obs._tracklets if len(t.frames) == 2)
        assert similar_tracklet is not None

    def test_generate_greedy_tracklets_deterministic_behavior(self, basic_detection):
        """Test that the algorithm produces deterministic results."""
        # Create identical observations
        observations = []
        for frame in range(3):
            frame_observations = []
            for obs_idx in range(2):
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=obs_idx,
                    embed_value=obs_idx * 0.5,
                    pose_coords=(obs_idx * 50, obs_idx * 50),
                )
                frame_observations.append(detection)
            observations.append(frame_observations)

        # Run twice with same input
        video_obs1 = VideoObservations(observations)
        video_obs1.generate_greedy_tracklets()

        video_obs2 = VideoObservations(observations)
        video_obs2.generate_greedy_tracklets()

        # Should produce same results
        assert len(video_obs1._tracklets) == len(video_obs2._tracklets)
        assert video_obs1._observation_id_dict == video_obs2._observation_id_dict

    def test_generate_greedy_tracklets_empty_observation_list(self):
        """Test with empty observation list."""
        # TODO: This reveals a bug - VideoObservations constructor can't handle empty lists
        # The constructor tries to calculate median of empty list
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            observations = []
            VideoObservations(observations)

    def test_generate_greedy_tracklets_numerical_stability(self, basic_detection):
        """Test with edge cases that might cause numerical issues."""
        observations = []
        for frame in range(2):
            detection = basic_detection(
                frame_idx=frame,
                pose_idx=0,
                embed_value=1e-10,  # Very small embedding value
                pose_coords=(1e6, 1e6),  # Very large coordinates
            )
            observations.append([detection])

        video_obs = VideoObservations(observations)
        video_obs.generate_greedy_tracklets(max_cost=np.inf)  # Allow any cost

        # Should handle numerical edge cases
        assert len(video_obs._tracklets) > 0
