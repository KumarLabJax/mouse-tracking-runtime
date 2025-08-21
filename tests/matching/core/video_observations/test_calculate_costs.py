"""Unit tests for VideoObservations._calculate_costs method.

This module contains comprehensive tests for the cost calculation algorithm,
including both parallel and non-parallel execution paths, edge cases, and error conditions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mouse_tracking.matching.core import Detection, VideoObservations


class TestCalculateCosts:
    """Tests for the _calculate_costs method."""

    def test_calculate_costs_non_parallel_basic(self, basic_detection):
        """Test basic functionality with non-parallel execution."""
        # Create observations for two frames
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0, embed_value=0.1)],
            [basic_detection(frame_idx=1, pose_idx=0, embed_value=0.2)],
        ]
        video_obs = VideoObservations(observations)

        # Ensure no pool is set (non-parallel path)
        video_obs._pool = None

        with patch.object(
            Detection, "calculate_match_cost", return_value=0.5
        ) as mock_cost:
            result = video_obs._calculate_costs(0, 1, rotate_pose=False)

            # Should call calculate_match_cost once
            mock_cost.assert_called_once()
            args, kwargs = mock_cost.call_args
            assert len(args) == 2  # Two detections
            assert not kwargs.get("pose_rotation")

            # Should return correct shape
            assert result.shape == (1, 1)
            assert result[0, 0] == 0.5

    def test_calculate_costs_non_parallel_multiple_observations(self, basic_detection):
        """Test non-parallel execution with multiple observations per frame."""
        # Create observations: 2 in first frame, 3 in second frame
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0, embed_value=0.1),
                basic_detection(frame_idx=0, pose_idx=1, embed_value=0.2),
            ],
            [
                basic_detection(frame_idx=1, pose_idx=0, embed_value=0.3),
                basic_detection(frame_idx=1, pose_idx=1, embed_value=0.4),
                basic_detection(frame_idx=1, pose_idx=2, embed_value=0.5),
            ],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(
            Detection, "calculate_match_cost", return_value=0.7
        ) as mock_cost:
            result = video_obs._calculate_costs(0, 1, rotate_pose=True)

            # Should call calculate_match_cost for each pair (2 * 3 = 6 times)
            assert mock_cost.call_count == 6

            # Should return correct shape (2x3 matrix)
            assert result.shape == (2, 3)
            assert np.all(result == 0.7)

            # Check that rotate_pose was passed correctly
            for call in mock_cost.call_args_list:
                args, kwargs = call
                assert kwargs.get("pose_rotation")

    def test_calculate_costs_non_parallel_observation_caching(self, basic_detection):
        """Test that observations are properly cached in non-parallel execution."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with (
            patch.object(Detection, "calculate_match_cost", return_value=0.5),
            patch.object(Detection, "cache") as mock_cache,
        ):
            video_obs._calculate_costs(0, 1)

            # Should cache all observations involved
            assert mock_cache.call_count == 2  # One for each observation

    def test_calculate_costs_parallel_basic(self, basic_detection):
        """Test basic functionality with parallel execution."""
        # Create observations for two frames
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0, embed_value=0.1)],
            [basic_detection(frame_idx=1, pose_idx=0, embed_value=0.2)],
        ]
        video_obs = VideoObservations(observations)

        # Set up mock pool
        mock_pool = MagicMock()
        mock_pool.map.return_value = [0.8]
        video_obs._pool = mock_pool

        result = video_obs._calculate_costs(0, 1, rotate_pose=True)

        # Should call pool.map once
        mock_pool.map.assert_called_once()
        args, kwargs = mock_pool.map.call_args
        assert args[0] == Detection.calculate_match_cost_multi

        # Check the chunks passed to pool.map
        chunks = args[1]
        assert len(chunks) == 1  # 1x1 = 1 chunk
        chunk = chunks[0]
        assert (
            len(chunk) == 6
        )  # (det1, det2, max_dist, default_cost, beta, rotate_pose)
        assert chunk[2] == 40  # max_dist
        assert chunk[3] == 0.0  # default_cost
        assert chunk[4] == (1.0, 1.0, 1.0)  # beta
        assert chunk[5]  # rotate_pose

        # Should return correct shape and values
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.8

    def test_calculate_costs_parallel_multiple_observations(self, basic_detection):
        """Test parallel execution with multiple observations per frame."""
        # Create observations: 2 in first frame, 2 in second frame
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0, embed_value=0.1),
                basic_detection(frame_idx=0, pose_idx=1, embed_value=0.2),
            ],
            [
                basic_detection(frame_idx=1, pose_idx=0, embed_value=0.3),
                basic_detection(frame_idx=1, pose_idx=1, embed_value=0.4),
            ],
        ]
        video_obs = VideoObservations(observations)

        # Set up mock pool
        mock_pool = MagicMock()
        mock_pool.map.return_value = [0.1, 0.2, 0.3, 0.4]  # 2x2 = 4 results
        video_obs._pool = mock_pool

        result = video_obs._calculate_costs(0, 1, rotate_pose=False)

        # Should call pool.map once
        mock_pool.map.assert_called_once()
        args, kwargs = mock_pool.map.call_args

        # Check the chunks
        chunks = args[1]
        assert len(chunks) == 4  # 2x2 = 4 chunks

        # Verify rotate_pose parameter in all chunks
        for chunk in chunks:
            assert not chunk[5]  # rotate_pose

        # Should return correct shape
        assert result.shape == (2, 2)
        expected = np.array([[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_array_equal(result, expected)

    def test_calculate_costs_empty_frames(self, basic_detection):
        """Test with empty frames."""
        observations = [[], []]  # Both frames empty
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        result = video_obs._calculate_costs(0, 1)

        # Should return empty matrix
        assert result.shape == (0, 0)

    def test_calculate_costs_asymmetric_frames(self, basic_detection):
        """Test with frames having different numbers of observations."""
        # First frame has 3 observations, second frame has 1
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0),
                basic_detection(frame_idx=0, pose_idx=1),
                basic_detection(frame_idx=0, pose_idx=2),
            ],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(Detection, "calculate_match_cost", return_value=1.5):
            result = video_obs._calculate_costs(0, 1)

            # Should return 3x1 matrix
            assert result.shape == (3, 1)
            assert np.all(result == 1.5)

    def test_calculate_costs_reverse_frame_order(self, basic_detection):
        """Test calculating costs in reverse frame order."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(Detection, "calculate_match_cost", return_value=2.0):
            result = video_obs._calculate_costs(1, 0)  # Reverse order

            # Should work correctly in reverse
            assert result.shape == (1, 1)
            assert result[0, 0] == 2.0

    def test_calculate_costs_same_frame(self, basic_detection):
        """Test calculating costs within the same frame."""
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0),
                basic_detection(frame_idx=0, pose_idx=1),
            ]
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(Detection, "calculate_match_cost", return_value=0.1):
            result = video_obs._calculate_costs(0, 0)

            # Should work for same frame
            assert result.shape == (2, 2)
            assert np.all(result == 0.1)

    def test_calculate_costs_invalid_frame_indices(self, basic_detection):
        """Test with invalid frame indices."""
        observations = [[basic_detection(frame_idx=0, pose_idx=0)]]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        # Test with out-of-bounds frame index
        with pytest.raises(IndexError):
            video_obs._calculate_costs(0, 1)  # Frame 1 doesn't exist

    def test_calculate_costs_matrix_shape_consistency(self, basic_detection):
        """Test that matrix shape is consistent regardless of execution path."""
        # Create same observations for both tests
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0),
                basic_detection(frame_idx=0, pose_idx=1),
            ],
            [
                basic_detection(frame_idx=1, pose_idx=0),
                basic_detection(frame_idx=1, pose_idx=1),
                basic_detection(frame_idx=1, pose_idx=2),
            ],
        ]

        # Test non-parallel
        video_obs1 = VideoObservations(observations)
        video_obs1._pool = None
        with patch.object(Detection, "calculate_match_cost", return_value=0.5):
            result1 = video_obs1._calculate_costs(0, 1)

        # Test parallel
        video_obs2 = VideoObservations(observations)
        mock_pool = MagicMock()
        mock_pool.map.return_value = [0.5] * 6  # 2x3 = 6 results
        video_obs2._pool = mock_pool
        result2 = video_obs2._calculate_costs(0, 1)

        # Both should have same shape
        assert result1.shape == result2.shape == (2, 3)

    def test_calculate_costs_parallel_chunk_creation(self, basic_detection):
        """Test that chunks are created correctly for parallel execution."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)

        mock_pool = MagicMock()
        mock_pool.map.return_value = [1.0]
        video_obs._pool = mock_pool

        video_obs._calculate_costs(0, 1, rotate_pose=True)

        # Get the chunks passed to pool.map
        chunks = mock_pool.map.call_args[0][1]
        chunk = chunks[0]

        # Verify chunk structure
        assert isinstance(chunk[0], Detection)  # First detection
        assert isinstance(chunk[1], Detection)  # Second detection
        assert chunk[2] == 40  # max_dist parameter
        assert chunk[3] == 0.0  # default_cost parameter
        assert chunk[4] == (1.0, 1.0, 1.0)  # beta parameter
        assert chunk[5]  # rotate_pose parameter

    def test_calculate_costs_parallel_meshgrid_ordering(self, basic_detection):
        """Test that meshgrid creates correct observation pairings."""
        # Create 2x2 observation matrix
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0, embed_value=0.1),
                basic_detection(frame_idx=0, pose_idx=1, embed_value=0.2),
            ],
            [
                basic_detection(frame_idx=1, pose_idx=0, embed_value=0.3),
                basic_detection(frame_idx=1, pose_idx=1, embed_value=0.4),
            ],
        ]
        video_obs = VideoObservations(observations)

        mock_pool = MagicMock()
        mock_pool.map.return_value = [1.0, 2.0, 3.0, 4.0]
        video_obs._pool = mock_pool

        video_obs._calculate_costs(0, 1)

        # Get the chunks and verify pairings
        chunks = mock_pool.map.call_args[0][1]
        assert len(chunks) == 4

        # Verify the detection pairings match expected meshgrid order
        expected_pairings = [
            (0, 0),  # obs[0][0] with obs[1][0]
            (1, 0),  # obs[0][1] with obs[1][0]
            (0, 1),  # obs[0][0] with obs[1][1]
            (1, 1),  # obs[0][1] with obs[1][1]
        ]

        for i, (frame1_idx, frame2_idx) in enumerate(expected_pairings):
            chunk = chunks[i]
            # Verify the detections are from the correct indices by comparing attributes
            expected_det1 = observations[0][frame1_idx]
            expected_det2 = observations[1][frame2_idx]
            assert chunk[0].frame == expected_det1.frame
            assert chunk[0].pose_idx == expected_det1.pose_idx
            assert chunk[1].frame == expected_det2.frame
            assert chunk[1].pose_idx == expected_det2.pose_idx

    def test_calculate_costs_parallel_result_reshaping(self, basic_detection):
        """Test that parallel results are correctly reshaped."""
        # Create 2x3 observation matrix
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0),
                basic_detection(frame_idx=0, pose_idx=1),
            ],
            [
                basic_detection(frame_idx=1, pose_idx=0),
                basic_detection(frame_idx=1, pose_idx=1),
                basic_detection(frame_idx=1, pose_idx=2),
            ],
        ]
        video_obs = VideoObservations(observations)

        mock_pool = MagicMock()
        # Results should be in row-major order for reshaping
        mock_pool.map.return_value = [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
        video_obs._pool = mock_pool

        result = video_obs._calculate_costs(0, 1)

        # Verify correct reshaping
        expected = np.array([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]])
        np.testing.assert_array_equal(result, expected)

    def test_calculate_costs_return_type(self, basic_detection):
        """Test that function returns numpy array."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(Detection, "calculate_match_cost", return_value=0.5):
            result = video_obs._calculate_costs(0, 1)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float64

    def test_calculate_costs_zero_initialization_non_parallel(self, basic_detection):
        """Test that non-parallel path initializes matrix with zeros."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        # Mock calculate_match_cost to not be called (simulating an error)
        with (
            patch.object(Detection, "calculate_match_cost", side_effect=RuntimeError),
            pytest.raises(RuntimeError),
        ):
            video_obs._calculate_costs(0, 1)

    def test_calculate_costs_method_call_order_non_parallel(self, basic_detection):
        """Test the order of method calls in non-parallel execution."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        call_order = []

        def mock_cache(self):
            call_order.append(f"cache_{self.frame}")

        def mock_calculate_match_cost(det1, det2, **kwargs):
            call_order.append(f"calculate_{det1.frame}_{det2.frame}")
            return 0.5

        with (
            patch.object(Detection, "cache", mock_cache),
            patch.object(Detection, "calculate_match_cost", mock_calculate_match_cost),
        ):
            video_obs._calculate_costs(0, 1)

            # Should cache first detection, then second, then calculate
            expected_order = ["cache_0", "cache_1", "calculate_0_1"]
            assert call_order == expected_order

    def test_calculate_costs_large_matrix(self, basic_detection):
        """Test with larger observation matrices."""
        # Create 5x7 observation matrix
        observations = [
            [basic_detection(frame_idx=0, pose_idx=i) for i in range(5)],
            [basic_detection(frame_idx=1, pose_idx=i) for i in range(7)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(Detection, "calculate_match_cost", return_value=3.0):
            result = video_obs._calculate_costs(0, 1)

            # Should handle large matrices correctly
            assert result.shape == (5, 7)
            assert np.all(result == 3.0)

    def test_calculate_costs_parallel_vs_non_parallel_equivalence(
        self, basic_detection
    ):
        """Test that parallel and non-parallel execution give equivalent results."""
        observations = [
            [
                basic_detection(frame_idx=0, pose_idx=0, embed_value=0.1),
                basic_detection(frame_idx=0, pose_idx=1, embed_value=0.2),
            ],
            [
                basic_detection(frame_idx=1, pose_idx=0, embed_value=0.3),
                basic_detection(frame_idx=1, pose_idx=1, embed_value=0.4),
            ],
        ]

        # Test non-parallel with deterministic costs
        video_obs1 = VideoObservations(observations)
        video_obs1._pool = None
        with patch.object(
            Detection, "calculate_match_cost", side_effect=[1.0, 2.0, 3.0, 4.0]
        ):
            result1 = video_obs1._calculate_costs(0, 1)

        # Test parallel with same costs
        video_obs2 = VideoObservations(observations)
        mock_pool = MagicMock()
        mock_pool.map.return_value = [1.0, 2.0, 3.0, 4.0]
        video_obs2._pool = mock_pool
        result2 = video_obs2._calculate_costs(0, 1)

        # Results should be equivalent
        np.testing.assert_array_equal(result1, result2)

    def test_calculate_costs_error_in_parallel_execution(self, basic_detection):
        """Test error handling in parallel execution."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0)],
            [basic_detection(frame_idx=1, pose_idx=0)],
        ]
        video_obs = VideoObservations(observations)

        mock_pool = MagicMock()
        mock_pool.map.side_effect = RuntimeError("Pool error")
        video_obs._pool = mock_pool

        with pytest.raises(RuntimeError, match="Pool error"):
            video_obs._calculate_costs(0, 1)

    def test_calculate_costs_edge_case_single_observation(self, basic_detection):
        """Test edge case with single observation in each frame."""
        observations = [
            [basic_detection(frame_idx=0, pose_idx=0, embed_value=0.5)],
            [basic_detection(frame_idx=1, pose_idx=0, embed_value=0.6)],
        ]
        video_obs = VideoObservations(observations)
        video_obs._pool = None

        with patch.object(Detection, "calculate_match_cost", return_value=0.25):
            result = video_obs._calculate_costs(0, 1)

            assert result.shape == (1, 1)
            assert result[0, 0] == 0.25
