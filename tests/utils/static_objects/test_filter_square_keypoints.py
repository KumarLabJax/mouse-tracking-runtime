"""Tests for filter_square_keypoints function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import filter_square_keypoints


class TestFilterSquareKeypoints:
    """Test cases for filter_square_keypoints function."""

    def test_filter_square_keypoints_perfect_unit_square(self):
        """Test filtering with a perfect unit square."""
        # Arrange - single prediction with perfect unit square
        predictions = np.array(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]]  # Perfect unit square
            ],
            dtype=np.float32,
        )
        tolerance = 25.0

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array(
                [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
            )

            # Act
            result = filter_square_keypoints(predictions, tolerance)

            # Assert
            mock_filter_static.assert_called_once()
            # Check that the perfect square was passed to filter_static_keypoints
            passed_predictions = mock_filter_static.call_args[0][0]
            assert passed_predictions.shape == (1, 4, 2)
            np.testing.assert_array_equal(passed_predictions[0], predictions[0])
            assert isinstance(result, np.ndarray)

    def test_filter_square_keypoints_multiple_valid_squares(self):
        """Test filtering with multiple valid square predictions."""
        # Arrange - multiple valid square predictions
        predictions = np.array(
            [
                [[0, 0], [2, 0], [2, 2], [0, 2]],  # 2x2 square
                [[1, 1], [3, 1], [3, 3], [1, 3]],  # Another 2x2 square, offset
                [[0, 0], [1, 0], [1, 1], [0, 1]],  # 1x1 square
            ],
            dtype=np.float32,
        )
        tolerance = 25.0

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions, tolerance)

            # Assert
            mock_filter_static.assert_called_once()
            # All three squares should be passed to filter_static_keypoints
            passed_predictions = mock_filter_static.call_args[0][0]
            assert passed_predictions.shape == (3, 4, 2)

    def test_filter_square_keypoints_mixed_valid_invalid(self):
        """Test filtering with mix of valid and invalid predictions."""
        # Arrange - mix of square and non-square predictions
        predictions = np.array(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]],  # Valid square
                [
                    [0, 0],
                    [10, 0],
                    [5, 5],
                    [0, 5],
                ],  # Invalid - very distorted quadrilateral
                [[0, 0], [1, 0], [1, 1], [0, 1]],  # Valid square (duplicate)
            ],
            dtype=np.float32,
        )
        tolerance = 1.0  # Tight tolerance to filter out non-squares

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions, tolerance)

            # Assert
            mock_filter_static.assert_called_once()
            # Only the valid squares should be passed
            passed_predictions = mock_filter_static.call_args[0][0]
            assert passed_predictions.shape == (2, 4, 2)  # Only 2 valid squares

    def test_filter_square_keypoints_no_valid_squares_raises_error(self):
        """Test that ValueError is raised when no valid squares are found."""
        # Arrange - no valid square predictions (very distorted shapes)
        predictions = np.array(
            [
                [[0, 0], [10, 0], [5, 20], [0, 5]],  # Very distorted quadrilateral
                [[0, 0], [1, 0], [20, 30], [0, 1]],  # Very distorted quadrilateral
            ],
            dtype=np.float32,
        )
        tolerance = 0.1  # Very tight tolerance

        # Act & Assert
        with pytest.raises(ValueError, match="No predictions were square."):
            filter_square_keypoints(predictions, tolerance)

    def test_filter_square_keypoints_wrong_shape_raises_assertion(self):
        """Test that AssertionError is raised for wrong input shape."""
        # Arrange - wrong shape (2D instead of 3D)
        predictions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

        # Act & Assert
        with pytest.raises(AssertionError):
            filter_square_keypoints(predictions)

    def test_filter_square_keypoints_custom_tolerance(self):
        """Test filtering with custom tolerance values."""
        # Arrange - slightly imperfect square that should pass with higher tolerance
        predictions = np.array(
            [
                [[0, 0], [1.1, 0], [1, 1.1], [0, 0.9]]  # Slightly imperfect square
            ],
            dtype=np.float32,
        )

        # Should fail with tight tolerance
        with pytest.raises(ValueError):
            filter_square_keypoints(predictions, tolerance=0.01)

        # Should pass with loose tolerance
        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            filter_square_keypoints(predictions, tolerance=10.0)
            mock_filter_static.assert_called_once()

    def test_filter_square_keypoints_uses_measure_pair_dists(self):
        """Test that the function uses measure_pair_dists for distance calculation."""
        # Arrange
        predictions = np.array(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]]  # Perfect unit square
            ],
            dtype=np.float32,
        )

        with (
            patch(
                "mouse_tracking.utils.static_objects.measure_pair_dists"
            ) as mock_measure_dists,
            patch(
                "mouse_tracking.utils.static_objects.filter_static_keypoints"
            ) as mock_filter_static,
        ):
            # Mock measure_pair_dists to return expected distances for unit square
            # Unit square: 4 edges of length 1, 2 diagonals of length sqrt(2)
            mock_measure_dists.return_value = np.array(
                [1.0, 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2)]
            )
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions)

            # Assert
            mock_measure_dists.assert_called_once()
            # Should be called with the single square prediction
            np.testing.assert_array_equal(
                mock_measure_dists.call_args[0][0], predictions[0]
            )

    def test_filter_square_keypoints_distance_sorting_and_splitting(self):
        """Test that distances are properly sorted and split into edges and diagonals."""
        # Arrange
        predictions = np.array(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]]  # Perfect unit square
            ],
            dtype=np.float32,
        )

        with (
            patch(
                "mouse_tracking.utils.static_objects.measure_pair_dists"
            ) as mock_measure_dists,
            patch(
                "mouse_tracking.utils.static_objects.filter_static_keypoints"
            ) as mock_filter_static,
        ):
            # Mock unsorted distances (should be sorted internally)
            mock_measure_dists.return_value = np.array(
                [np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, 1.0]
            )
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions, tolerance=1.0)

            # Assert
            # Should pass because after sorting and processing, all edges should be equal
            mock_filter_static.assert_called_once()

    def test_filter_square_keypoints_diagonal_to_edge_conversion(self):
        """Test that diagonals are properly converted to equivalent edge lengths."""
        # Arrange - square where we can verify the diagonal conversion
        predictions = np.array(
            [
                [[0, 0], [2, 0], [2, 2], [0, 2]]  # 2x2 square
            ],
            dtype=np.float32,
        )

        with (
            patch(
                "mouse_tracking.utils.static_objects.measure_pair_dists"
            ) as mock_measure_dists,
            patch(
                "mouse_tracking.utils.static_objects.filter_static_keypoints"
            ) as mock_filter_static,
        ):
            # For 2x2 square: 4 edges of length 2, 2 diagonals of length 2*sqrt(2)
            mock_measure_dists.return_value = np.array(
                [2.0, 2.0, 2.0, 2.0, 2 * np.sqrt(2), 2 * np.sqrt(2)]
            )
            mock_filter_static.return_value = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])

            # Act
            filter_square_keypoints(predictions, tolerance=1.0)

            # Assert
            # Diagonals (2*sqrt(2)) converted to edges: sqrt((2*sqrt(2))²/2) = 2
            # So all "edges" should be length 2, which should pass tolerance test
            mock_filter_static.assert_called_once()

    @pytest.mark.parametrize("tolerance", [0.1, 1.0, 10.0, 50.0])
    def test_filter_square_keypoints_various_tolerances(self, tolerance):
        """Test filtering with various tolerance values."""
        # Arrange - perfect square should pass any reasonable tolerance
        predictions = np.array(
            [
                [[0, 0], [1, 0], [1, 1], [0, 1]]  # Perfect unit square
            ],
            dtype=np.float32,
        )

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions, tolerance=tolerance)

            # Assert
            mock_filter_static.assert_called_once()
            # Check that the tolerance was passed correctly (as second positional argument)
            assert mock_filter_static.call_args[0][1] == tolerance

    def test_filter_square_keypoints_empty_predictions(self):
        """Test behavior with empty predictions array."""
        # Arrange
        predictions = np.zeros((0, 4, 2), dtype=np.float32)

        # Act & Assert
        with pytest.raises(ValueError, match="No predictions were square."):
            filter_square_keypoints(predictions)

    def test_filter_square_keypoints_single_prediction_valid(self):
        """Test with single valid square prediction."""
        # Arrange
        predictions = np.array(
            [
                [[0, 0], [3, 0], [3, 3], [0, 3]]  # 3x3 square
            ],
            dtype=np.float32,
        )

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])

            # Act
            filter_square_keypoints(predictions)

            # Assert
            mock_filter_static.assert_called_once()
            passed_predictions = mock_filter_static.call_args[0][0]
            assert passed_predictions.shape == (1, 4, 2)

    def test_filter_square_keypoints_edge_error_calculation(self):
        """Test that edge error calculation works correctly."""
        # Arrange - prediction that should fail tight tolerance
        predictions = np.array(
            [
                [[0, 0], [1, 0], [1.5, 1], [0, 1]]  # Distorted square
            ],
            dtype=np.float32,
        )

        # Should fail with very tight tolerance
        with pytest.raises(ValueError):
            filter_square_keypoints(predictions, tolerance=0.01)

    def test_filter_square_keypoints_return_type(self):
        """Test that the function returns the correct type from filter_static_keypoints."""
        # Arrange
        predictions = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)

        expected_result = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = expected_result

            # Act
            result = filter_square_keypoints(predictions)

            # Assert
            np.testing.assert_array_equal(result, expected_result)
            assert result.shape == (4, 2)

    def test_filter_square_keypoints_passes_tolerance_to_filter_static(self):
        """Test that tolerance parameter is passed to filter_static_keypoints."""
        # Arrange
        predictions = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
        custom_tolerance = 15.5

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions, tolerance=custom_tolerance)

            # Assert
            mock_filter_static.assert_called_once()
            # Check that tolerance was passed correctly (as second positional argument)
            assert mock_filter_static.call_args[0][1] == custom_tolerance

    def test_filter_square_keypoints_large_number_predictions(self):
        """Test performance and correctness with larger number of predictions."""
        # Arrange - many predictions, mix of valid and invalid
        n_predictions = 10
        predictions = []

        for i in range(n_predictions):
            if i % 3 == 0:  # Every third is a valid square
                size = 1 + i * 0.5
                square = np.array([[0, 0], [size, 0], [size, size], [0, size]])
                predictions.append(square)
            else:  # Others are clearly not squares with very distorted shapes
                # Create clearly non-square quadrilaterals
                quad = np.array([[0, 0], [10 + i, 0], [5, 20 + i], [0, 3 + i]])
                predictions.append(quad)

        predictions = np.array(predictions, dtype=np.float32)

        with patch(
            "mouse_tracking.utils.static_objects.filter_static_keypoints"
        ) as mock_filter_static:
            mock_filter_static.return_value = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            # Act
            filter_square_keypoints(predictions, tolerance=1.0)  # Tighter tolerance

            # Assert
            mock_filter_static.assert_called_once()
            # Should have filtered to only the valid squares (every 3rd prediction)
            passed_predictions = mock_filter_static.call_args[0][0]
            expected_valid_count = len([i for i in range(n_predictions) if i % 3 == 0])
            assert passed_predictions.shape[0] == expected_valid_count
