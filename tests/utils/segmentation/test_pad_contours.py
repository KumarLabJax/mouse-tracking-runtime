"""
Unit tests for the pad_contours function from mouse_tracking.utils.segmentation.

This module tests the pad_contours function which converts OpenCV contour data
into a padded matrix format suitable for batch processing and storage.
"""

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import pad_contours


class TestPadContours:
    """Test class for pad_contours function."""

    def test_single_contour_basic(self):
        """Test with single contour in OpenCV format."""
        # Arrange - OpenCV contour format is [n_points, 1, 2]
        contour = np.array([[[10, 20]], [[30, 40]], [[50, 60]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (1, 3, 2)
        assert result.dtype == np.int32

        # Check that contour data is properly squeezed and stored
        expected = np.array([[[10, 20], [30, 40], [50, 60]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_contours_same_length(self):
        """Test with multiple contours of the same length."""
        # Arrange
        contour1 = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contour2 = np.array([[[50, 60]], [[70, 80]]], dtype=np.int32)
        contours = [contour1, contour2]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (2, 2, 2)
        assert result.dtype == np.int32

        expected = np.array(
            [[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32
        )
        np.testing.assert_array_equal(result, expected)

    def test_multiple_contours_different_lengths(self):
        """Test with multiple contours of different lengths - core functionality."""
        # Arrange
        contour1 = np.array(
            [[[10, 20]], [[30, 40]], [[50, 60]]], dtype=np.int32
        )  # 3 points
        contour2 = np.array([[[70, 80]]], dtype=np.int32)  # 1 point
        contour3 = np.array(
            [[[90, 100]], [[110, 120]], [[130, 140]], [[150, 160]]], dtype=np.int32
        )  # 4 points
        contours = [contour1, contour2, contour3]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (3, 4, 2)  # 3 contours, max 4 points each
        assert result.dtype == np.int32

        expected = np.array(
            [
                [[10, 20], [30, 40], [50, 60], [-1, -1]],  # First contour + padding
                [[70, 80], [-1, -1], [-1, -1], [-1, -1]],  # Second contour + padding
                [
                    [90, 100],
                    [110, 120],
                    [130, 140],
                    [150, 160],
                ],  # Third contour (longest)
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_custom_default_value(self):
        """Test with custom default padding value."""
        # Arrange
        contour1 = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contour2 = np.array([[[50, 60]]], dtype=np.int32)
        contours = [contour1, contour2]
        default_val = -999

        # Act
        result = pad_contours(contours, default_val)

        # Assert
        assert result.shape == (2, 2, 2)

        expected = np.array(
            [[[10, 20], [30, 40]], [[50, 60], [-999, -999]]], dtype=np.int32
        )
        np.testing.assert_array_equal(result, expected)

    def test_zero_default_value(self):
        """Test with zero as default padding value."""
        # Arrange
        contour1 = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contour2 = np.array([[[50, 60]]], dtype=np.int32)
        contours = [contour1, contour2]
        default_val = 0

        # Act
        result = pad_contours(contours, default_val)

        # Assert
        expected = np.array([[[10, 20], [30, 40]], [[50, 60], [0, 0]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_positive_default_value(self):
        """Test with positive default padding value."""
        # Arrange
        contour = np.array([[[10, 20]]], dtype=np.int32)
        contours = [contour]
        default_val = 42

        # Act
        result = pad_contours(contours, default_val)

        # Assert
        expected = np.array([[[10, 20]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_empty_contours_list(self):
        """Test with empty contours list - should raise ValueError."""
        # Arrange
        contours = []

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="zero-size array to reduction operation maximum which has no identity",
        ):
            pad_contours(contours)

    def test_contour_with_zero_points(self):
        """Test with contour containing zero points."""
        # Arrange
        contour1 = np.array([[[10, 20]]], dtype=np.int32)
        contour2 = np.zeros((0, 1, 2), dtype=np.int32)  # Empty contour
        contours = [contour1, contour2]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (2, 1, 2)

        expected = np.array(
            [
                [[10, 20]],
                [[-1, -1]],  # Empty contour gets padded
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_contour_squeeze_functionality(self):
        """Test that np.squeeze is properly applied to contour data."""
        # Arrange - contour with extra dimensions that should be squeezed
        contour = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert - should have shape (1, 2, 2) not (1, 2, 1, 2)
        assert result.shape == (1, 2, 2)
        expected = np.array([[[10, 20], [30, 40]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_contour_different_shapes(self):
        """Test with contours of different shapes (but valid OpenCV format)."""
        # Arrange
        contour1 = np.array([[[10, 20]], [[30, 40]], [[50, 60]]], dtype=np.int32)
        contour2 = np.array(
            [[[70, 80]], [[90, 100]], [[110, 120]], [[130, 140]], [[150, 160]]],
            dtype=np.int32,
        )
        contours = [contour1, contour2]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (2, 5, 2)

        expected = np.array(
            [
                [[10, 20], [30, 40], [50, 60], [-1, -1], [-1, -1]],
                [[70, 80], [90, 100], [110, 120], [130, 140], [150, 160]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_large_contours(self):
        """Test with large contours to verify memory efficiency."""
        # Arrange
        large_contour = np.random.randint(0, 1000, (500, 1, 2), dtype=np.int32)
        small_contour = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contours = [large_contour, small_contour]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (2, 500, 2)
        assert result.dtype == np.int32

        # Check that large contour is preserved
        np.testing.assert_array_equal(result[0], large_contour.squeeze())

        # Check that small contour is padded correctly
        expected_small = np.full((500, 2), -1, dtype=np.int32)
        expected_small[0] = [10, 20]
        expected_small[1] = [30, 40]
        np.testing.assert_array_equal(result[1], expected_small)

    def test_different_data_types(self):
        """Test with different input data types (should be converted to int32)."""
        # Arrange
        contour1 = np.array([[[10, 20]], [[30, 40]]], dtype=np.float32)
        contour2 = np.array([[[50, 60]]], dtype=np.int16)
        contours = [contour1, contour2]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.dtype == np.int32
        assert result.shape == (2, 2, 2)

        expected = np.array(
            [[[10, 20], [30, 40]], [[50, 60], [-1, -1]]], dtype=np.int32
        )
        np.testing.assert_array_equal(result, expected)

    def test_negative_coordinates(self):
        """Test with negative coordinate values."""
        # Arrange
        contour = np.array([[[-10, -20]], [[30, -40]], [[-50, 60]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert
        expected = np.array([[[-10, -20], [30, -40], [-50, 60]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_very_large_coordinates(self):
        """Test with very large coordinate values."""
        # Arrange
        max_val = np.iinfo(np.int32).max
        contour = np.array([[[max_val, max_val]], [[0, 0]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert
        expected = np.array([[[max_val, max_val], [0, 0]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("default_val", [-1, 0, 1, -100, 100, -999])
    def test_various_default_values(self, default_val):
        """Test with various default padding values."""
        # Arrange
        contour1 = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contour2 = np.array([[[50, 60]]], dtype=np.int32)
        contours = [contour1, contour2]

        # Act
        result = pad_contours(contours, default_val)

        # Assert
        assert result.shape == (2, 2, 2)

        expected = np.array(
            [[[10, 20], [30, 40]], [[50, 60], [default_val, default_val]]],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_single_point_contours(self):
        """Test with contours containing single points."""
        # Arrange
        contour1 = np.array([[[100, 200]]], dtype=np.int32)
        contour2 = np.array([[[300, 400]]], dtype=np.int32)
        contours = [contour1, contour2]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (2, 1, 2)

        expected = np.array([[[100, 200]], [[300, 400]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_contour_sizes(self):
        """Test comprehensive mix of contour sizes."""
        # Arrange
        contour1 = np.array([[[1, 2]]], dtype=np.int32)  # 1 point
        contour2 = np.array([[[3, 4]], [[5, 6]]], dtype=np.int32)  # 2 points
        contour3 = np.array(
            [[[7, 8]], [[9, 10]], [[11, 12]]], dtype=np.int32
        )  # 3 points
        contour4 = np.array(
            [[[13, 14]], [[15, 16]], [[17, 18]], [[19, 20]]], dtype=np.int32
        )  # 4 points
        contours = [contour1, contour2, contour3, contour4]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (4, 4, 2)

        expected = np.array(
            [
                [[1, 2], [-1, -1], [-1, -1], [-1, -1]],
                [[3, 4], [5, 6], [-1, -1], [-1, -1]],
                [[7, 8], [9, 10], [11, 12], [-1, -1]],
                [[13, 14], [15, 16], [17, 18], [19, 20]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_return_type_and_shape(self):
        """Test that return type and shape are correct."""
        # Arrange
        contour = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert len(result.shape) == 3
        assert result.shape[0] == len(contours)  # Number of contours
        assert result.shape[2] == 2  # Always 2 for (x, y) coordinates

    def test_memory_layout_c_contiguous(self):
        """Test that resulting array has efficient memory layout."""
        # Arrange
        contour = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.flags.c_contiguous or result.flags.f_contiguous

    def test_no_modification_of_input(self):
        """Test that input contours are not modified."""
        # Arrange
        original_contour = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        contour_copy = original_contour.copy()
        contours = [original_contour]

        # Act
        result = pad_contours(contours)

        # Assert
        np.testing.assert_array_equal(original_contour, contour_copy)
        assert result is not original_contour  # Different object

    def test_edge_case_all_zero_coordinates(self):
        """Test with all zero coordinates."""
        # Arrange
        contour = np.array([[[0, 0]], [[0, 0]], [[0, 0]]], dtype=np.int32)
        contours = [contour]

        # Act
        result = pad_contours(contours)

        # Assert
        expected = np.array([[[0, 0], [0, 0], [0, 0]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_max_contour_length_calculation(self):
        """Test that max contour length is calculated correctly."""
        # Arrange
        short_contour = np.array([[[1, 2]]], dtype=np.int32)
        long_contour = np.array(
            [[[3, 4]], [[5, 6]], [[7, 8]], [[9, 10]], [[11, 12]]], dtype=np.int32
        )
        medium_contour = np.array([[[13, 14]], [[15, 16]], [[17, 18]]], dtype=np.int32)
        contours = [short_contour, long_contour, medium_contour]

        # Act
        result = pad_contours(contours)

        # Assert
        # Max length should be 5 (from long_contour)
        assert result.shape[1] == 5

    def test_squeeze_removes_singleton_dimensions(self):
        """Test that squeeze properly removes singleton dimensions from OpenCV format."""
        # Arrange - simulate OpenCV contour format [n_points, 1, 2]
        contour_data = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        assert contour_data.shape == (2, 1, 2)  # Verify OpenCV format
        contours = [contour_data]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (1, 2, 2)  # Should be [1, 2, 2], not [1, 2, 1, 2]
        expected = np.array([[[10, 20], [30, 40]]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_integration_with_realistic_opencv_contours(self):
        """Integration test with realistic OpenCV contour data."""
        # Arrange - create realistic contour data like OpenCV would produce
        # These represent rectangular and triangular shapes
        rect_contour = np.array(
            [[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]], dtype=np.int32
        )
        triangle_contour = np.array([[[0, 0]], [[10, 0]], [[5, 10]]], dtype=np.int32)
        contours = [rect_contour, triangle_contour]

        # Act
        result = pad_contours(contours)

        # Assert
        assert result.shape == (2, 4, 2)

        expected = np.array(
            [
                [[10, 10], [50, 10], [50, 50], [10, 50]],
                [[0, 0], [10, 0], [5, 10], [-1, -1]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(result, expected)
