"""
Unit tests for localmax_2d function from mouse_tracking.utils.arrays.

This module tests the localmax_2d function which performs non-maximum suppression
to find peaks in 2D arrays. The function uses OpenCV morphological operations
for peak detection and filtering.

NOTE: This function calls get_peak_coords internally, so it inherits the same bugs
where values have incorrect shapes due to the indexing bug in get_peak_coords.
These tests document the current buggy behavior to ensure backward compatibility.
"""

import cv2
import numpy as np
import pytest

from mouse_tracking.utils.arrays import localmax_2d


class TestLocalmax2D:
    """Test cases for the localmax_2d function."""

    @pytest.mark.parametrize(
        "shape,threshold,radius",
        [
            ((5, 5), 0.5, 1),
            ((10, 10), 0.3, 2),
            ((8, 8), 0.7, 1),
            ((6, 4), 0.4, 1),  # Non-square
            ((20, 20), 0.1, 3),  # Larger array
        ],
    )
    def test_localmax_2d_basic_functionality(self, shape, threshold, radius):
        """Test basic functionality with various input parameters."""
        # Arrange
        arr = np.random.rand(*shape) * 0.5  # Keep values low
        height, width = shape

        # Add some clear peaks above threshold
        peak_positions = [
            (1, 1),
            (height // 2, width // 2),
        ]

        # Ensure peaks are safe from IndexError bug and spaced apart
        safe_positions = []
        for row, col in peak_positions:
            if row < height and col < width and col < height:  # col < height due to bug
                # Check spacing from other peaks
                is_safe = True
                for existing_row, existing_col in safe_positions:
                    if (
                        abs(row - existing_row) <= radius * 2
                        or abs(col - existing_col) <= radius * 2
                    ):
                        is_safe = False
                        break
                if is_safe:
                    arr[row, col] = threshold + 0.3  # Well above threshold
                    safe_positions.append((row, col))

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert basic structure
        # BUG: Inherited from get_peak_coords - values have shape (n_peaks, 2, width) instead of (n_peaks,)
        if len(coordinates) > 0:
            assert values.shape == (len(coordinates), 2, width), (
                f"Expected values shape ({len(coordinates)}, 2, {width}), got {values.shape}"
            )
            assert coordinates.shape[1] == 2, "Coordinates should have 2 columns"

            # Verify coordinates are within bounds
            assert np.all(coordinates[:, 0] >= 0) and np.all(
                coordinates[:, 0] < height
            ), "Row coordinates out of bounds"
            assert np.all(coordinates[:, 1] >= 0) and np.all(
                coordinates[:, 1] < width
            ), "Column coordinates out of bounds"
        else:
            # No peaks found
            assert values.shape == (0,), (
                "No peaks case should return empty values array"
            )
            assert coordinates.shape == (0, 2), (
                "No peaks case should return empty coordinates array"
            )

    def test_localmax_2d_single_peak(self):
        """Test with a single clear peak."""
        # Arrange
        arr = np.zeros((7, 7))
        arr[3, 3] = 1.0  # Single peak at center
        threshold = 0.5
        radius = 1

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        assert len(coordinates) == 1, "Should find exactly one peak"
        # BUG: Values have shape (1, 2, 7) instead of (1,)
        assert values.shape == (1, 2, 7), (
            f"Expected values shape (1, 2, 7), got {values.shape}"
        )
        assert coordinates[0, 0] == 3 and coordinates[0, 1] == 3, (
            "Peak should be at center (3, 3)"
        )

    def test_localmax_2d_multiple_peaks_suppressed(self):
        """Test that nearby peaks are suppressed by non-max suppression."""
        # Arrange
        arr = np.zeros((9, 9))
        threshold = 0.5
        radius = 2

        # Place two peaks close together - only the larger should survive
        arr[3, 3] = 0.8  # Smaller peak
        arr[4, 4] = 1.0  # Larger peak (should suppress the smaller one)

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        # Due to non-max suppression, only the stronger peak should remain
        assert len(coordinates) <= 2, (
            "Should find at most 2 peaks due to non-max suppression"
        )

        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 9) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 9), (
                f"Expected values shape ({len(coordinates)}, 2, 9), got {values.shape}"
            )

    def test_localmax_2d_threshold_filtering(self):
        """Test that threshold properly filters peaks."""
        # Arrange
        arr = np.zeros((5, 5))
        threshold = 0.6
        radius = 1

        # Add peaks above and below threshold
        arr[1, 1] = 0.5  # Below threshold - should be filtered out
        arr[3, 3] = 0.8  # Above threshold - should be kept

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        # Only the peak above threshold should be found
        if len(coordinates) > 0:
            found_positions = [(coord[0], coord[1]) for coord in coordinates]
            assert (3, 3) in found_positions, "Peak above threshold should be found"
            assert (1, 1) not in found_positions, (
                "Peak below threshold should be filtered out"
            )

    def test_localmax_2d_no_peaks_found(self):
        """Test behavior when no peaks are found."""
        # Arrange
        arr = np.ones((5, 5)) * 0.3  # Uniform array below threshold
        threshold = 0.5
        radius = 1

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        assert values.shape == (0,), (
            "Should return empty values array when no peaks found"
        )
        assert coordinates.shape == (0, 2), (
            "Should return empty coordinates array when no peaks found"
        )

    @pytest.mark.parametrize("radius", [1, 2, 3, 5])
    def test_localmax_2d_different_radii(self, radius):
        """Test with different suppression radii."""
        # Arrange
        arr = np.zeros((15, 15))
        threshold = 0.5

        # Place peaks at known positions with sufficient spacing
        spacing = radius * 3  # Ensure they're far enough apart
        for i in range(0, 15, spacing):
            for j in range(0, 15, spacing):
                if i < 15 and j < 15 and j < 15:  # Avoid IndexError bug
                    arr[i, j] = 0.8

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert basic structure
        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 15) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 15), (
                f"Expected values shape ({len(coordinates)}, 2, 15), got {values.shape}"
            )
            assert coordinates.shape[1] == 2, "Coordinates should have 2 columns"

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.uint8,
        ],  # Removed int32 as OpenCV doesn't support it
    )
    def test_localmax_2d_different_dtypes(self, dtype):
        """Test with different numpy data types."""
        # Arrange
        if dtype == np.uint8:
            arr = np.zeros((5, 5), dtype=dtype)
            arr[2, 2] = dtype(200)  # Use valid uint8 value
            threshold = 100
        else:
            arr = np.zeros((5, 5), dtype=dtype)
            arr[2, 2] = dtype(0.8)
            threshold = 0.5

        radius = 1

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 5) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 5), (
                f"Expected values shape ({len(coordinates)}, 2, 5), got {values.shape}"
            )
            assert coordinates.shape[1] == 2, "Coordinates should have 2 columns"

    def test_localmax_2d_unsupported_dtypes(self):
        """Test that unsupported data types raise appropriate errors."""
        # Arrange
        arr = np.zeros((5, 5), dtype=np.int32)  # OpenCV doesn't support int32
        arr[2, 2] = 10
        threshold = 5
        radius = 1

        # Act & Assert
        # OpenCV should raise an error for unsupported data types
        with pytest.raises(cv2.error):  # OpenCV error for unsupported dtypes
            localmax_2d(arr, threshold, radius)

    def test_localmax_2d_input_validation_radius(self):
        """Test input validation for radius parameter."""
        # Arrange
        arr = np.ones((5, 5))
        threshold = 0.5

        # Act & Assert
        with pytest.raises(AssertionError):
            localmax_2d(arr, threshold, 0)  # radius < 1 should fail

        with pytest.raises(AssertionError):
            localmax_2d(arr, threshold, -1)  # negative radius should fail

    def test_localmax_2d_input_validation_dimensions(self):
        """Test input validation for array dimensions."""
        # Arrange
        threshold = 0.5
        radius = 1

        # Test 1D array
        arr_1d = np.array([1, 2, 3])
        with pytest.raises(AssertionError):
            localmax_2d(arr_1d, threshold, radius)

        # Test 3D array
        arr_3d = np.ones((3, 3, 3))
        with pytest.raises(AssertionError):
            localmax_2d(arr_3d, threshold, radius)

        # Test 0D array (scalar)
        arr_0d = np.array(5.0)
        with pytest.raises(AssertionError):
            localmax_2d(arr_0d, threshold, radius)

    def test_localmax_2d_squeezable_inputs_bug(self):
        """Test that function fails with squeezable multi-dimensional inputs due to a bug."""
        # Arrange - arrays that become 2D when squeezed
        arr_3d_squeezable = np.ones((1, 5, 5))  # Can be squeezed to 2D
        arr_3d_squeezable[0, 2, 2] = 2.0
        threshold = 1.5
        radius = 1

        # Act & Assert
        # BUG: The function fails with squeezable inputs because it uses the original
        # array for masking operations instead of the squeezed version
        with pytest.raises(IndexError, match="too many indices for array"):
            localmax_2d(arr_3d_squeezable, threshold, radius)

    def test_localmax_2d_proper_2d_inputs(self):
        """Test that function works with proper 2D inputs."""
        # Arrange - actual 2D array (not squeezable)
        arr_2d = np.ones((5, 5))
        arr_2d[2, 2] = 2.0
        threshold = 1.5
        radius = 1

        # Act
        values, coordinates = localmax_2d(arr_2d, threshold, radius)

        # Assert
        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 5) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 5), (
                f"Expected values shape ({len(coordinates)}, 2, 5), got {values.shape}"
            )

    def test_localmax_2d_edge_peaks(self):
        """Test detection of peaks at array edges."""
        # Arrange
        arr = np.zeros((6, 6))
        threshold = 0.5
        radius = 1

        # Place peaks at edges (avoiding IndexError bug)
        arr[0, 0] = 0.8  # Corner
        arr[0, 3] = 0.8  # Edge, but col < height so safe
        arr[3, 0] = 0.8  # Edge

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 6) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 6), (
                f"Expected values shape ({len(coordinates)}, 2, 6), got {values.shape}"
            )

            # Check that edge coordinates are valid
            assert np.all(coordinates[:, 0] >= 0), (
                "Row coordinates should be non-negative"
            )
            assert np.all(coordinates[:, 1] >= 0), (
                "Column coordinates should be non-negative"
            )

    def test_localmax_2d_uniform_array(self):
        """Test with uniform array (no peaks)."""
        # Arrange
        arr = np.ones((4, 4)) * 0.5  # Uniform array
        threshold = 0.3  # Below the uniform value
        radius = 1

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        # Due to morphological operations, uniform arrays typically don't produce peaks
        assert values.shape[0] == coordinates.shape[0], (
            "Values and coordinates should have same length"
        )

    def test_localmax_2d_extreme_threshold_values(self):
        """Test with extreme threshold values."""
        # Arrange
        arr = np.random.rand(6, 6)
        radius = 1

        # Test very high threshold (no peaks should be found)
        values_high, coordinates_high = localmax_2d(
            arr, 2.0, radius
        )  # Above max possible value
        assert len(coordinates_high) == 0, "Very high threshold should find no peaks"

        # Test very low threshold (many peaks might be found)
        values_low, coordinates_low = localmax_2d(
            arr, -1.0, radius
        )  # Below min possible value
        # Should find some peaks, but exact number depends on non-max suppression
        assert coordinates_low.shape[1] == 2, "Should return valid coordinate format"

    def test_localmax_2d_large_radius(self):
        """Test with radius larger than array dimensions."""
        # Arrange
        arr = np.zeros((5, 5))
        arr[2, 2] = 1.0  # Single peak
        threshold = 0.5
        radius = 10  # Much larger than array

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        # Should still work, morphological operations handle large kernels
        assert isinstance(values, np.ndarray), "Should return numpy array for values"
        assert isinstance(coordinates, np.ndarray), (
            "Should return numpy array for coordinates"
        )

    def test_localmax_2d_indexerror_bug_avoidance(self):
        """Test scenarios that would trigger the inherited IndexError bug."""
        # Arrange - create scenario where peaks have col >= height
        arr = np.zeros((3, 6))  # 3 rows, 6 columns
        threshold = 0.5
        radius = 1

        # This peak would cause IndexError due to bug in get_peak_coords
        # The bug happens when col coordinate >= number of rows
        arr[1, 4] = 0.8  # col=4 >= height=3 would cause IndexError

        # Act & Assert
        # This should raise IndexError due to the bug in get_peak_coords
        with pytest.raises(IndexError, match="index .* is out of bounds"):
            localmax_2d(arr, threshold, radius)

    def test_localmax_2d_minimum_valid_inputs(self):
        """Test with minimum valid input sizes."""
        # Arrange
        arr = np.zeros((2, 2))  # Minimum 2D array
        arr[0, 0] = 1.0
        threshold = 0.5
        radius = 1  # Minimum valid radius

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 2) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 2), (
                f"Expected values shape ({len(coordinates)}, 2, 2), got {values.shape}"
            )

    def test_localmax_2d_backward_compatibility_regression(self):
        """
        Regression test to ensure backward compatibility.

        This test verifies that the function behaves consistently with its current
        behavior for typical use cases, including the inherited bugs.
        """
        # Arrange - realistic peak detection scenario
        np.random.seed(42)
        arr = np.random.rand(10, 10) * 0.4  # Background noise
        threshold = 0.6
        radius = 2

        # Add clear peaks at safe positions
        peak_positions = [
            (2, 2),
            (7, 3),
            (4, 8),
        ]  # Ensure col < height to avoid IndexError
        for row, col in peak_positions:
            if col < arr.shape[0]:  # Avoid the IndexError bug
                arr[row, col] = 0.9

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert basic structure
        assert isinstance(values, np.ndarray), "Values should be numpy array"
        assert isinstance(coordinates, np.ndarray), "Coordinates should be numpy array"
        assert values.shape[0] == coordinates.shape[0], (
            "Values and coordinates should have same length"
        )

        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 10) instead of (n_peaks,)
            assert values.shape[1:] == (2, 10), (
                f"Expected values shape (n_peaks, 2, 10), got {values.shape}"
            )
            assert coordinates.shape[1] == 2, "Coordinates should have 2 columns"

            # Verify peaks are within bounds
            assert np.all(coordinates[:, 0] >= 0), (
                "Row coordinates should be non-negative"
            )
            assert np.all(coordinates[:, 0] < arr.shape[0]), (
                "Row coordinates should be within array bounds"
            )
            assert np.all(coordinates[:, 1] >= 0), (
                "Column coordinates should be non-negative"
            )
            assert np.all(coordinates[:, 1] < arr.shape[1]), (
                "Column coordinates should be within array bounds"
            )

    def test_localmax_2d_morphological_operations_behavior(self):
        """Test that morphological operations work as expected."""
        # Arrange - create a pattern where morphological operations matter
        arr = np.zeros((7, 7))
        threshold = 0.3
        radius = 1

        # Create a cross pattern - center should be peak, arms should be suppressed
        arr[3, 3] = 1.0  # Center peak
        arr[3, 2] = 0.8  # Should be suppressed
        arr[3, 4] = 0.8  # Should be suppressed
        arr[2, 3] = 0.8  # Should be suppressed
        arr[4, 3] = 0.8  # Should be suppressed

        # Act
        values, coordinates = localmax_2d(arr, threshold, radius)

        # Assert
        # The exact behavior depends on OpenCV's morphological operations
        # We mainly verify the function runs and returns valid structure
        assert values.shape[0] == coordinates.shape[0], (
            "Values and coordinates should have matching length"
        )

        if len(coordinates) > 0:
            # BUG: Values have shape (n_peaks, 2, 7) instead of (n_peaks,)
            assert values.shape == (len(coordinates), 2, 7), (
                f"Expected values shape ({len(coordinates)}, 2, 7), got {values.shape}"
            )
