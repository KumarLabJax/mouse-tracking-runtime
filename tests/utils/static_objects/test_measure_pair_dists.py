"""Tests for measure_pair_dists function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import measure_pair_dists


class TestMeasurePairDists:
    """Test cases for measure_pair_dists function."""

    def test_measure_pair_dists_basic_functionality(self):
        """Test basic pairwise distance calculation functionality."""
        # Arrange
        keypoints = np.array([[0, 0], [3, 0], [0, 4]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        # For 3 points, should have 3 pairwise distances (3*2/2 = 3)
        assert len(result) == 3
        # Expected distances: (0,0)-(3,0)=3, (0,0)-(0,4)=4, (3,0)-(0,4)=5
        expected_distances = np.array([3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(
            np.sort(result), np.sort(expected_distances)
        )

    def test_measure_pair_dists_two_points(self):
        """Test pairwise distance calculation with two points."""
        # Arrange
        keypoints = np.array([[0, 0], [3, 4]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        # For 2 points, should have 1 pairwise distance
        assert len(result) == 1
        # Distance between (0,0) and (3,4) should be 5
        np.testing.assert_almost_equal(result[0], 5.0)

    def test_measure_pair_dists_single_point(self):
        """Test pairwise distance calculation with single point."""
        # Arrange
        keypoints = np.array([[5, 10]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        # For 1 point, should have 0 pairwise distances
        assert len(result) == 0

    def test_measure_pair_dists_empty_array(self):
        """Test pairwise distance calculation with empty array."""
        # Arrange
        keypoints = np.zeros((0, 2), dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        # For 0 points, should have 0 pairwise distances
        assert len(result) == 0

    def test_measure_pair_dists_four_points_square(self):
        """Test pairwise distance calculation with four points forming a square."""
        # Arrange - unit square corners
        keypoints = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        # For 4 points, should have 6 pairwise distances (4*3/2 = 6)
        assert len(result) == 6

        sorted_result = np.sort(result)
        # Expected: 4 edges of length 1, 2 diagonals of length sqrt(2)
        expected_edges = np.array([1.0, 1.0, 1.0, 1.0])
        expected_diagonals = np.array([np.sqrt(2), np.sqrt(2)])
        expected_all = np.sort(np.concatenate([expected_edges, expected_diagonals]))

        np.testing.assert_array_almost_equal(sorted_result, expected_all)

    @pytest.mark.parametrize(
        "n_points,expected_distances",
        [
            (2, 1),  # 2 points -> 1 distance
            (3, 3),  # 3 points -> 3 distances
            (4, 6),  # 4 points -> 6 distances
            (5, 10),  # 5 points -> 10 distances
            (6, 15),  # 6 points -> 15 distances
        ],
    )
    def test_measure_pair_dists_correct_number_of_distances(
        self, n_points, expected_distances
    ):
        """Test that the correct number of pairwise distances is returned for various point counts."""
        # Arrange - random points
        np.random.seed(42)  # For reproducibility
        keypoints = np.random.rand(n_points, 2).astype(np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert len(result) == expected_distances
        assert isinstance(result, np.ndarray)

    def test_measure_pair_dists_uses_cdist(self):
        """Test that the function uses scipy.spatial.distance.cdist."""
        # Arrange
        keypoints = np.array([[0, 0], [1, 0]], dtype=np.float32)

        with patch("mouse_tracking.utils.static_objects.cdist") as mock_cdist:
            # Mock cdist to return a simple distance matrix
            mock_cdist.return_value = np.array([[0.0, 1.0], [1.0, 0.0]])

            # Act
            result = measure_pair_dists(keypoints)

            # Assert
            mock_cdist.assert_called_once_with(keypoints, keypoints)
            # Should extract upper triangular values (excluding diagonal)
            np.testing.assert_array_equal(result, np.array([1.0]))

    def test_measure_pair_dists_upper_triangular_extraction(self):
        """Test that only upper triangular distances are extracted."""
        # Arrange
        keypoints = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        with patch("mouse_tracking.utils.static_objects.cdist") as mock_cdist:
            # Mock a symmetric distance matrix
            mock_cdist.return_value = np.array(
                [[0.0, 1.0, 1.0], [1.0, 0.0, np.sqrt(2)], [1.0, np.sqrt(2), 0.0]]
            )

            # Act
            result = measure_pair_dists(keypoints)

            # Assert
            # Should only return upper triangular values: [1.0, 1.0, sqrt(2)]
            expected = np.array([1.0, 1.0, np.sqrt(2)])
            np.testing.assert_array_almost_equal(np.sort(result), np.sort(expected))

    def test_measure_pair_dists_excludes_diagonal(self):
        """Test that diagonal elements (self-distances) are excluded."""
        # Arrange
        keypoints = np.array([[5, 10]], dtype=np.float32)

        with patch("mouse_tracking.utils.static_objects.cdist") as mock_cdist:
            # Mock distance matrix with diagonal element
            mock_cdist.return_value = np.array([[0.0]])

            # Act
            result = measure_pair_dists(keypoints)

            # Assert
            # Should exclude the diagonal (self-distance of 0)
            assert len(result) == 0

    def test_measure_pair_dists_float_precision(self):
        """Test that the function handles floating point precision correctly."""
        # Arrange - points that create known floating point results
        keypoints = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # 3 points -> 3 distances

        # Expected distances: sqrt(2), 2, sqrt(2)
        sorted_result = np.sort(result)
        expected = np.sort([np.sqrt(2), 2.0, np.sqrt(2)])
        np.testing.assert_array_almost_equal(sorted_result, expected, decimal=6)

    def test_measure_pair_dists_identical_points(self):
        """Test behavior with identical points."""
        # Arrange - two identical points
        keypoints = np.array([[1, 1], [1, 1]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        # Distance between identical points is 0, which gets filtered out by np.nonzero
        # So we expect an empty array
        assert len(result) == 0

    def test_measure_pair_dists_negative_coordinates(self):
        """Test function with negative coordinates."""
        # Arrange
        keypoints = np.array([[-1, -1], [1, -1], [0, 1]], dtype=np.float32)

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

        # Calculate expected distances manually
        # (-1,-1) to (1,-1): distance = 2
        # (-1,-1) to (0,1): distance = sqrt(1+4) = sqrt(5)
        # (1,-1) to (0,1): distance = sqrt(1+4) = sqrt(5)
        expected = np.sort([2.0, np.sqrt(5), np.sqrt(5)])
        np.testing.assert_array_almost_equal(np.sort(result), expected)

    def test_measure_pair_dists_large_coordinates(self):
        """Test function with large coordinate values."""
        # Arrange
        keypoints = np.array(
            [[1000, 2000], [1003, 2000], [1000, 2004]], dtype=np.float32
        )

        # Act
        result = measure_pair_dists(keypoints)

        # Assert
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

        # Expected distances: 3, 4, 5 (scaled version of 3-4-5 triangle)
        expected = np.sort([3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(np.sort(result), expected)

    def test_measure_pair_dists_return_type_and_shape(self):
        """Test that return type and shape are correct for various inputs."""
        # Arrange
        test_cases = [
            np.array([[0, 0]], dtype=np.float32),  # 1 point
            np.array([[0, 0], [1, 0]], dtype=np.float32),  # 2 points
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),  # 3 points
        ]
        expected_lengths = [0, 1, 3]

        for keypoints, expected_length in zip(
            test_cases, expected_lengths, strict=False
        ):
            # Act
            result = measure_pair_dists(keypoints)

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1  # Should be 1D array
            assert len(result) == expected_length
            assert result.dtype in [np.float32, np.float64]  # Should be floating point
