"""
Unit tests for argmax_2d function from mouse_tracking.utils.arrays.

This module tests the argmax_2d function which finds peaks for all keypoints in pose data.
The function takes arrays of shape [batch, 12, img_width, img_height] and returns
the maximum values and their coordinates for each keypoint in each batch.
"""

import numpy as np
import pytest
from numpy.exceptions import AxisError

from mouse_tracking.utils.arrays import argmax_2d


class TestArgmax2D:
    """Test cases for the argmax_2d function."""

    @pytest.mark.parametrize(
        "batch_size,num_keypoints,img_width,img_height",
        [
            (1, 1, 5, 5),
            (1, 12, 10, 10),
            (2, 12, 8, 8),
            (3, 12, 15, 15),
            (1, 12, 64, 64),  # More realistic image size
            (4, 12, 32, 32),  # Multiple batches with realistic size
        ],
    )
    def test_argmax_2d_basic_functionality(
        self, batch_size, num_keypoints, img_width, img_height
    ):
        """Test basic functionality with various input shapes."""
        # Arrange
        arr = np.random.rand(batch_size, num_keypoints, img_width, img_height)

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert values.shape == (batch_size, num_keypoints), (
            f"Expected values shape {(batch_size, num_keypoints)}, got {values.shape}"
        )
        assert coordinates.shape == (batch_size, num_keypoints, 2), (
            f"Expected coordinates shape {(batch_size, num_keypoints, 2)}, got {coordinates.shape}"
        )

        # Verify that coordinates are within valid bounds
        assert np.all(coordinates[:, :, 0] >= 0), (
            "Row coordinates should be non-negative"
        )
        assert np.all(coordinates[:, :, 0] < img_width), (
            f"Row coordinates should be less than {img_width}"
        )
        assert np.all(coordinates[:, :, 1] >= 0), (
            "Column coordinates should be non-negative"
        )
        assert np.all(coordinates[:, :, 1] < img_height), (
            f"Column coordinates should be less than {img_height}"
        )

    @pytest.mark.parametrize(
        "max_row,max_col,expected_value",
        [
            (0, 0, 10.0),  # Top-left corner
            (2, 2, 15.0),  # Center
            (4, 4, 20.0),  # Bottom-right corner
            (1, 3, 25.0),  # Off-center
            (3, 1, 30.0),  # Different off-center
        ],
    )
    def test_argmax_2d_known_maxima(self, max_row, max_col, expected_value):
        """Test that argmax_2d correctly identifies known maximum positions."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 1, 1, 5, 5
        arr = np.ones((batch_size, num_keypoints, img_width, img_height))
        arr[0, 0, max_row, max_col] = expected_value

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert values[0, 0] == expected_value, (
            f"Expected value {expected_value}, got {values[0, 0]}"
        )
        assert coordinates[0, 0, 0] == max_row, (
            f"Expected row {max_row}, got {coordinates[0, 0, 0]}"
        )
        assert coordinates[0, 0, 1] == max_col, (
            f"Expected col {max_col}, got {coordinates[0, 0, 1]}"
        )

    def test_argmax_2d_multiple_keypoints_different_maxima(self):
        """Test with multiple keypoints having different maximum positions."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 1, 3, 5, 5
        arr = np.zeros((batch_size, num_keypoints, img_width, img_height))

        # Set different maxima for each keypoint
        expected_positions = [(0, 0), (2, 2), (4, 4)]
        expected_values = [10.0, 20.0, 30.0]

        for i, ((row, col), value) in enumerate(
            zip(expected_positions, expected_values, strict=False)
        ):
            arr[0, i, row, col] = value

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        for i, (expected_pos, expected_val) in enumerate(
            zip(expected_positions, expected_values, strict=False)
        ):
            assert values[0, i] == expected_val, (
                f"Keypoint {i}: expected value {expected_val}, got {values[0, i]}"
            )
            assert coordinates[0, i, 0] == expected_pos[0], (
                f"Keypoint {i}: expected row {expected_pos[0]}, got {coordinates[0, i, 0]}"
            )
            assert coordinates[0, i, 1] == expected_pos[1], (
                f"Keypoint {i}: expected col {expected_pos[1]}, got {coordinates[0, i, 1]}"
            )

    def test_argmax_2d_multiple_batches(self):
        """Test with multiple batches to ensure batch processing works correctly."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 2, 2, 3, 3
        arr = np.zeros((batch_size, num_keypoints, img_width, img_height))

        # Batch 0: maxima at (0,0) and (1,1)
        arr[0, 0, 0, 0] = 5.0
        arr[0, 1, 1, 1] = 6.0

        # Batch 1: maxima at (2,2) and (0,2)
        arr[1, 0, 2, 2] = 7.0
        arr[1, 1, 0, 2] = 8.0

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        # Batch 0 assertions
        assert values[0, 0] == 5.0, (
            f"Batch 0, keypoint 0: expected 5.0, got {values[0, 0]}"
        )
        assert coordinates[0, 0, 0] == 0 and coordinates[0, 0, 1] == 0
        assert values[0, 1] == 6.0, (
            f"Batch 0, keypoint 1: expected 6.0, got {values[0, 1]}"
        )
        assert coordinates[0, 1, 0] == 1 and coordinates[0, 1, 1] == 1

        # Batch 1 assertions
        assert values[1, 0] == 7.0, (
            f"Batch 1, keypoint 0: expected 7.0, got {values[1, 0]}"
        )
        assert coordinates[1, 0, 0] == 2 and coordinates[1, 0, 1] == 2
        assert values[1, 1] == 8.0, (
            f"Batch 1, keypoint 1: expected 8.0, got {values[1, 1]}"
        )
        assert coordinates[1, 1, 0] == 0 and coordinates[1, 1, 1] == 2

    @pytest.mark.parametrize("fill_value", [0.0, -1.0, 1.0, 100.0, -100.0])
    def test_argmax_2d_uniform_values(self, fill_value):
        """Test behavior when all values in an array are the same."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 1, 2, 3, 3
        arr = np.full((batch_size, num_keypoints, img_width, img_height), fill_value)

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert np.all(values == fill_value), f"All values should be {fill_value}"
        # When all values are the same, argmax should return (0, 0) consistently
        assert np.all(coordinates[:, :, 0] == 0), (
            "Row coordinates should be 0 for uniform arrays"
        )
        assert np.all(coordinates[:, :, 1] == 0), (
            "Column coordinates should be 0 for uniform arrays"
        )

    def test_argmax_2d_extreme_values(self):
        """Test with extreme floating point values."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 1, 3, 4, 4
        arr = np.ones((batch_size, num_keypoints, img_width, img_height))

        # Set extreme values
        arr[0, 0, 1, 1] = np.inf
        arr[0, 1, 2, 2] = -np.inf
        arr[0, 2, 3, 3] = np.finfo(np.float64).max

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert values[0, 0] == np.inf, "Should handle positive infinity"
        assert coordinates[0, 0, 0] == 1 and coordinates[0, 0, 1] == 1

        assert values[0, 1] == 1.0, "Should choose finite value over negative infinity"
        # For keypoint 1, max should be at one of the positions with value 1.0

        assert values[0, 2] == np.finfo(np.float64).max, (
            "Should handle maximum float value"
        )
        assert coordinates[0, 2, 0] == 3 and coordinates[0, 2, 1] == 3

    def test_argmax_2d_with_nan_values(self):
        """Test behavior with NaN values in the array."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 1, 2, 3, 3
        arr = np.ones((batch_size, num_keypoints, img_width, img_height))

        # Set some NaN values
        arr[0, 0, 0, 0] = np.nan
        arr[0, 1, 1, 1] = 5.0  # Clear maximum for second keypoint

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        # NaN behavior in argmax is to return NaN if present
        assert np.isnan(values[0, 0]) or values[0, 0] == 1.0, (
            "Should handle NaN appropriately"
        )
        assert values[0, 1] == 5.0, "Should find clear maximum despite other NaN values"
        assert coordinates[0, 1, 0] == 1 and coordinates[0, 1, 1] == 1

    def test_argmax_2d_invalid_1d_input(self):
        """Test that function raises AxisError for 1D input arrays."""
        # Arrange
        arr = np.random.rand(5)

        # Act & Assert
        with pytest.raises(AxisError, match="axis -2 is out of bounds"):
            argmax_2d(arr)

    @pytest.mark.parametrize(
        "shape,expected_values_shape,expected_coords_shape",
        [
            ((5, 5), (), (2,)),  # 2D array - works but produces scalar outputs
            ((5, 5, 5), (5,), (5, 2)),  # 3D array - works as batch of 1D keypoint data
            (
                (1, 2, 3, 4, 5),
                (1, 2, 3),
                (1, 2, 3, 2),
            ),  # 5D array - works by treating extra dims as batch/keypoint dims
        ],
    )
    def test_argmax_2d_unexpected_but_working_shapes(
        self, shape, expected_values_shape, expected_coords_shape
    ):
        """
        Test current behavior with non-4D input shapes that still work.

        These tests document the current behavior for backward compatibility,
        even though these shapes may not be the intended use case.
        """
        # Arrange
        arr = np.random.rand(*shape)

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert values.shape == expected_values_shape, (
            f"Expected values shape {expected_values_shape}, got {values.shape}"
        )
        assert coordinates.shape == expected_coords_shape, (
            f"Expected coordinates shape {expected_coords_shape}, got {coordinates.shape}"
        )

    def test_argmax_2d_minimum_size_input(self):
        """Test with minimum possible valid input size."""
        # Arrange
        arr = np.array([[[[5.0]]]])  # shape (1, 1, 1, 1)

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert values.shape == (1, 1)
        assert coordinates.shape == (1, 1, 2)
        assert values[0, 0] == 5.0
        assert coordinates[0, 0, 0] == 0 and coordinates[0, 0, 1] == 0

    def test_argmax_2d_standard_pose_dimensions(self):
        """Test with the standard dimensions mentioned in the docstring."""
        # Arrange - using the exact dimensions from docstring
        batch_size, num_keypoints = 1, 12
        img_width, img_height = 64, 64  # Realistic pose estimation dimensions
        arr = np.random.rand(batch_size, num_keypoints, img_width, img_height)

        # Set known maxima for first few keypoints
        for i in range(min(3, num_keypoints)):
            arr[0, i, i * 10 % img_width, i * 10 % img_height] = 10.0 + i

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert
        assert values.shape == (batch_size, num_keypoints)
        assert coordinates.shape == (batch_size, num_keypoints, 2)

        # Verify the known maxima we set
        for i in range(min(3, num_keypoints)):
            expected_value = 10.0 + i
            expected_row = i * 10 % img_width
            expected_col = i * 10 % img_height

            assert values[0, i] == expected_value
            assert coordinates[0, i, 0] == expected_row
            assert coordinates[0, i, 1] == expected_col

    def test_argmax_2d_data_types(self):
        """Test that function works with different numpy data types."""
        # Arrange
        batch_size, num_keypoints, img_width, img_height = 1, 2, 3, 3

        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            arr = np.ones(
                (batch_size, num_keypoints, img_width, img_height), dtype=dtype
            )
            arr[0, 0, 1, 1] = 5
            arr[0, 1, 2, 2] = 10

            # Act
            values, coordinates = argmax_2d(arr)

            # Assert
            assert values.shape == (batch_size, num_keypoints)
            assert coordinates.shape == (batch_size, num_keypoints, 2)
            assert values[0, 0] == 5
            assert values[0, 1] == 10
            assert coordinates[0, 0, 0] == 1 and coordinates[0, 0, 1] == 1
            assert coordinates[0, 1, 0] == 2 and coordinates[0, 1, 1] == 2

    def test_argmax_2d_backward_compatibility_regression(self):
        """
        Regression test to ensure backward compatibility.

        This test verifies that the function behaves consistently with its documented
        interface and expected behavior for typical use cases.
        """
        # Arrange - realistic scenario with multiple batches and keypoints
        np.random.seed(42)  # For reproducible results
        batch_size, num_keypoints, img_width, img_height = 2, 12, 32, 32
        arr = np.random.rand(batch_size, num_keypoints, img_width, img_height) * 0.5

        # Add clear peaks for verification
        peak_positions = [
            (5, 10),
            (15, 20),
            (8, 8),
            (25, 5),
            (10, 25),
            (20, 15),
            (3, 3),
            (28, 28),
            (12, 18),
            (22, 7),
            (7, 22),
            (16, 12),
        ]

        for batch in range(batch_size):
            for keypoint in range(num_keypoints):
                row, col = peak_positions[keypoint]
                arr[batch, keypoint, row, col] = 1.0

        # Act
        values, coordinates = argmax_2d(arr)

        # Assert - verify structure and key properties
        assert values.shape == (batch_size, num_keypoints)
        assert coordinates.shape == (batch_size, num_keypoints, 2)
        assert values.dtype in [np.float32, np.float64]
        assert coordinates.dtype in [np.int32, np.int64]

        # Verify that all detected peaks are at the expected positions
        for batch in range(batch_size):
            for keypoint in range(num_keypoints):
                expected_row, expected_col = peak_positions[keypoint]
                assert values[batch, keypoint] == 1.0, (
                    f"Batch {batch}, keypoint {keypoint}: expected peak value 1.0"
                )
                assert coordinates[batch, keypoint, 0] == expected_row, (
                    f"Batch {batch}, keypoint {keypoint}: wrong row"
                )
                assert coordinates[batch, keypoint, 1] == expected_col, (
                    f"Batch {batch}, keypoint {keypoint}: wrong column"
                )
