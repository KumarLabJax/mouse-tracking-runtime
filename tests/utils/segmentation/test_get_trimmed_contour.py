"""
Unit tests for the get_trimmed_contour function from mouse_tracking.utils.segmentation.

This module tests the get_trimmed_contour function which removes padding values
from contour arrays to extract valid coordinate data. The function filters out
rows that match the specified default padding value and ensures proper data
type conversion to int32 for OpenCV compatibility.

The tests cover:
- Padding removal from various positions (end, middle, mixed)
- Custom padding values and edge cases
- Empty contours and all-padding scenarios
- Data type conversion and shape preservation
- Integration with OpenCV contour processing workflows
"""

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import get_trimmed_contour


class TestGetTrimmedContour:
    """Test suite for get_trimmed_contour function."""

    def test_normal_contour_with_padding(self):
        """Test trimming a contour with padding at the end."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
                [-1, -1],  # padding
                [-1, -1],  # padding
            ]
        )
        expected = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.int32

    def test_contour_with_padding_in_middle(self):
        """Test trimming a contour with padding in the middle."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [-1, -1],  # padding
                [30, 40],
                [50, 60],
            ]
        )
        expected = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_contour_without_padding(self):
        """Test trimming a contour that has no padding."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ]
        )
        expected = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_contour_all_padding(self):
        """Test trimming a contour that is all padding values."""
        # Arrange
        padded_contour = np.array(
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ]
        )
        expected = np.array([], dtype=np.int32).reshape(0, 2)

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (0, 2)

    def test_empty_contour(self):
        """Test trimming an empty contour."""
        # Arrange
        padded_contour = np.array([]).reshape(0, 2)
        expected = np.array([], dtype=np.int32).reshape(0, 2)

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (0, 2)

    def test_single_point_contour(self):
        """Test trimming a contour with a single point."""
        # Arrange
        padded_contour = np.array([[10, 20]])
        expected = np.array([[10, 20]], dtype=np.int32)

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_custom_default_value(self):
        """Test trimming with a custom default padding value."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [999, 999],  # custom padding
                [999, 999],  # custom padding
            ]
        )
        expected = np.array(
            [
                [10, 20],
                [30, 40],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour, default_val=999)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_partial_padding_row(self):
        """Test that rows with partial padding are not removed."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [-1, 30],  # partial padding - should not be removed
                [50, 60],
                [-1, -1],  # full padding - should be removed
            ]
        )
        expected = np.array(
            [
                [10, 20],
                [-1, 30],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_float_input_conversion(self):
        """Test that float inputs are converted to int32."""
        # Arrange
        padded_contour = np.array(
            [
                [10.5, 20.7],
                [30.2, 40.9],
                [-1.0, -1.0],  # padding
            ],
            dtype=np.float64,
        )
        expected = np.array(
            [
                [10, 20],
                [30, 40],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.int32

    def test_negative_coordinates(self):
        """Test trimming contour with negative coordinates."""
        # Arrange
        padded_contour = np.array(
            [
                [-10, -20],
                [30, 40],
                [-1, -1],  # padding
            ]
        )
        expected = np.array(
            [
                [-10, -20],
                [30, 40],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_zero_padding_value(self):
        """Test trimming with zero as the padding value."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [0, 0],  # zero padding
                [0, 0],  # zero padding
            ]
        )
        expected = np.array(
            [
                [10, 20],
                [30, 40],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_trimmed_contour(padded_contour, default_val=0)

        # Assert
        np.testing.assert_array_equal(result, expected)

    def test_maintains_shape_format(self):
        """Test that the result maintains the expected shape format."""
        # Arrange
        padded_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [-1, -1],
            ]
        )

        # Act
        result = get_trimmed_contour(padded_contour)

        # Assert
        assert result.ndim == 2
        assert result.shape[1] == 2  # Always 2 columns for x,y coordinates
        assert result.shape[0] == 2  # 2 non-padding rows

    @pytest.mark.parametrize(
        "input_array,default_val,expected_shape",
        [
            (np.array([[1, 2], [3, 4]]), -1, (2, 2)),
            (np.array([[1, 2], [-1, -1]]), -1, (1, 2)),
            (np.array([[-1, -1], [-1, -1]]), -1, (0, 2)),
            (np.array([[0, 0], [1, 1]]), 0, (1, 2)),
        ],
    )
    def test_parametrized_shapes(self, input_array, default_val, expected_shape):
        """Test various input combinations and their expected output shapes."""
        # Act
        result = get_trimmed_contour(input_array, default_val)

        # Assert
        assert result.shape == expected_shape
        assert result.dtype == np.int32
