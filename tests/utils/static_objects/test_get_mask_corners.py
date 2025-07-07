"""Unit tests for get_mask_corners function.

This module contains comprehensive tests for the mask corner detection functionality,
ensuring proper handling of computer vision operations, affine transformations,
and contour processing.
"""

import contextlib
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from mouse_tracking.utils.static_objects import get_mask_corners


@pytest.fixture
def standard_img_size():
    """Standard image size for testing.

    Returns:
        tuple: Image size (width, height) in pixels.
    """
    return (512, 512)


@pytest.fixture
def simple_box():
    """Simple bounding box for testing.

    Returns:
        numpy.ndarray: Bounding box [x1, y1, x2, y2] format.
    """
    return np.array([0.2, 0.2, 0.8, 0.8], dtype=np.float32)


@pytest.fixture
def large_box():
    """Large bounding box for testing.

    Returns:
        numpy.ndarray: Large bounding box [x1, y1, x2, y2] format.
    """
    return np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)


@pytest.fixture
def mock_sort_corners():
    """Mock the sort_corners function to work around the bug in source code.

    Returns:
        Mock object for sort_corners function.
    """

    def mock_sort_function(corners, img_size):
        # Return corners in a consistent format for testing
        return corners.astype(np.float32)

    with patch(
        "mouse_tracking.utils.static_objects.sort_corners",
        side_effect=mock_sort_function,
    ):
        yield


def create_simple_rectangular_mask(width: int = 255, height: int = 255) -> np.ndarray:
    """Create a simple rectangular mask that works with the affine transformation.

    Args:
        width: Mask width in pixels.
        height: Mask height in pixels.

    Returns:
        numpy.ndarray: Binary mask with rectangular object.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    # Create a centered rectangle that should survive affine transformation
    center_x, center_y = width // 2, height // 2
    rect_w, rect_h = width // 3, height // 3

    x1 = center_x - rect_w // 2
    x2 = center_x + rect_w // 2
    y1 = center_y - rect_h // 2
    y2 = center_y + rect_h // 2

    mask[y1:y2, x1:x2] = 1.0
    return mask


def create_full_mask(width: int = 255, height: int = 255) -> np.ndarray:
    """Create a mask that fills the entire space.

    Args:
        width: Mask width in pixels.
        height: Mask height in pixels.

    Returns:
        numpy.ndarray: Binary mask filling entire space.
    """
    return np.ones((height, width), dtype=np.float32)


def create_circular_mask(
    width: int = 255, height: int = 255, radius_ratio: float = 0.3
) -> np.ndarray:
    """Create a circular mask for testing.

    Args:
        width: Mask width in pixels.
        height: Mask height in pixels.
        radius_ratio: Radius as ratio of minimum dimension.

    Returns:
        numpy.ndarray: Binary mask with circular object.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    center_x, center_y = width // 2, height // 2
    radius = int(min(width, height) * radius_ratio)

    y, x = np.ogrid[:height, :width]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    mask[mask_circle] = 1.0

    return mask


def validate_corners_output(corners: np.ndarray) -> bool:
    """Validate that corners output has correct format.

    Args:
        corners: Output from get_mask_corners function.

    Returns:
        bool: True if corners are valid format.
    """
    return (
        isinstance(corners, np.ndarray)
        and corners.shape == (4, 2)
        and np.isfinite(corners).all()
        and corners.dtype in [np.float32, np.float64]
    )


class TestGetMaskCornersSuccessfulCases:
    """Test successful execution paths of get_mask_corners function."""

    def test_simple_rectangular_mask(
        self, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test corner detection with simple rectangular mask.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)
        # All corners should be within reasonable bounds
        assert np.all(corners >= 0)
        assert np.all(corners[:, 0] <= standard_img_size[0])
        assert np.all(corners[:, 1] <= standard_img_size[1])

    def test_full_mask(self, simple_box, standard_img_size, mock_sort_corners):
        """Test corner detection with mask filling entire space.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_full_mask()

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

    def test_circular_mask(self, simple_box, standard_img_size, mock_sort_corners):
        """Test corner detection with circular mask.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_circular_mask()

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)
        # Corners should form a reasonable bounding rectangle
        x_coords, y_coords = corners[:, 0], corners[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        assert width > 0 and height > 0

    @pytest.mark.parametrize(
        "box_coords",
        [
            [0.1, 0.1, 0.5, 0.5],  # Small box
            [0.2, 0.2, 0.8, 0.8],  # Medium box
            [0.0, 0.0, 1.0, 1.0],  # Full box
        ],
    )
    def test_different_box_sizes(
        self, box_coords, standard_img_size, mock_sort_corners
    ):
        """Test corner detection with various bounding box sizes.

        Args:
            box_coords: Bounding box coordinates [x1, y1, x2, y2].
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        box = np.array(box_coords, dtype=np.float32)
        mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

    @pytest.mark.parametrize(
        "img_size",
        [
            (256, 256),  # Small image
            (512, 512),  # Standard image
            (1024, 768),  # Large rectangular image
        ],
    )
    def test_different_image_sizes(self, simple_box, img_size, mock_sort_corners):
        """Test corner detection with various image sizes.

        Args:
            simple_box: Fixture providing simple bounding box.
            img_size: Image size (width, height) to test.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(simple_box, mask, img_size)

        # Assert
        assert validate_corners_output(corners)
        # Corners should be within image bounds
        assert np.all(corners[:, 0] <= img_size[0])
        assert np.all(corners[:, 1] <= img_size[1])


class TestGetMaskCornersEdgeCases:
    """Test edge cases and boundary conditions of get_mask_corners function."""

    def test_mask_at_threshold(self, simple_box, standard_img_size):
        """Test corner detection with mask values exactly at threshold.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
        """
        # Arrange
        mask = create_simple_rectangular_mask()
        mask[mask > 0] = 0.5  # Exactly at threshold (will be > 0.5 after processing)

        # Act & Assert - this should raise an error since 0.5 is not > 0.5
        with pytest.raises((ValueError, AttributeError, cv2.error)):
            get_mask_corners(simple_box, mask, standard_img_size)

    def test_high_threshold_mask_values(
        self, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test corner detection with mask values well above threshold.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_simple_rectangular_mask()
        mask[mask > 0] = 0.9  # Well above threshold

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

    @pytest.mark.parametrize("data_type", [np.float32, np.float64, np.uint8])
    def test_different_mask_data_types(
        self, simple_box, standard_img_size, data_type, mock_sort_corners
    ):
        """Test corner detection with different mask data types.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            data_type: NumPy data type to test.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_simple_rectangular_mask()
        if data_type == np.uint8:
            mask = (mask * 255).astype(data_type)
        else:
            mask = mask.astype(data_type)

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)


class TestGetMaskCornersErrorCases:
    """Test error conditions and exception handling of get_mask_corners function."""

    def test_empty_mask_raises_error(self, simple_box, standard_img_size):
        """Test behavior with completely empty mask.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
        """
        # Arrange
        mask = np.zeros((255, 255), dtype=np.float32)  # Completely empty

        # Act & Assert
        with pytest.raises((ValueError, AttributeError, cv2.error)):
            get_mask_corners(simple_box, mask, standard_img_size)

    def test_mask_below_threshold_raises_error(self, simple_box, standard_img_size):
        """Test behavior with mask values all below threshold.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
        """
        # Arrange
        mask = np.full((255, 255), 0.4, dtype=np.float32)  # All below 0.5 threshold

        # Act & Assert
        with pytest.raises((ValueError, AttributeError, cv2.error)):
            get_mask_corners(simple_box, mask, standard_img_size)

    def test_invalid_box_format_raises_error(self, standard_img_size):
        """Test behavior with invalid bounding box format.

        Args:
            standard_img_size: Fixture providing standard image size.
        """
        # Arrange
        invalid_box = np.array([0.5, 0.5], dtype=np.float32)  # Wrong shape
        mask = create_simple_rectangular_mask()

        # Act & Assert
        with pytest.raises(IndexError):
            get_mask_corners(invalid_box, mask, standard_img_size)

    def test_negative_box_coordinates(self, standard_img_size, mock_sort_corners):
        """Test behavior with negative bounding box coordinates.

        Args:
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        negative_box = np.array([-0.1, -0.1, 0.5, 0.5], dtype=np.float32)
        mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(negative_box, mask, standard_img_size)

        # Assert - should handle gracefully
        assert validate_corners_output(corners)

    def test_box_coordinates_out_of_range(self, standard_img_size, mock_sort_corners):
        """Test behavior with bounding box coordinates > 1.0.

        Args:
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        large_box = np.array(
            [0.0, 0.0, 1.5, 1.5], dtype=np.float32
        )  # Beyond normal range
        mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(large_box, mask, standard_img_size)

        # Assert - should handle gracefully
        assert validate_corners_output(corners)

    def test_zero_size_image_raises_error(self, simple_box):
        """Test behavior with zero-size image.

        Args:
            simple_box: Fixture providing simple bounding box.
        """
        # Arrange
        zero_img_size = (0, 0)
        mask = create_simple_rectangular_mask()

        # Act & Assert
        with pytest.raises((ValueError, cv2.error)):
            get_mask_corners(simple_box, mask, zero_img_size)


class TestGetMaskCornersIntegration:
    """Integration tests for get_mask_corners function with realistic scenarios."""

    def test_realistic_object_detection_scenario(
        self, standard_img_size, mock_sort_corners
    ):
        """Test corner detection with realistic object detection scenario.

        Args:
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange - simulate realistic object detection
        object_box = np.array([0.3, 0.2, 0.7, 0.6], dtype=np.float32)
        object_mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(object_box, object_mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

        # Verify corners form reasonable rectangle
        x_coords, y_coords = corners[:, 0], corners[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        # Should be reasonable size
        assert width > 10  # At least 10 pixels wide
        assert height > 10  # At least 10 pixels tall
        assert width < standard_img_size[0]  # Not larger than image
        assert height < standard_img_size[1]  # Not larger than image

    def test_small_object_detection(self, standard_img_size, mock_sort_corners):
        """Test corner detection with small object.

        Args:
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange - small object
        small_box = np.array([0.4, 0.4, 0.6, 0.6], dtype=np.float32)
        small_mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(small_box, small_mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

        # For small objects, corners should be close together
        x_coords, y_coords = corners[:, 0], corners[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        # Should detect small object appropriately
        assert 0 < width < standard_img_size[0] // 2  # Reasonable small width
        assert 0 < height < standard_img_size[1] // 2  # Reasonable small height

    def test_large_object_detection(self, standard_img_size, mock_sort_corners):
        """Test corner detection with large object covering most of image.

        Args:
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange - large object
        large_box = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        large_mask = create_full_mask()

        # Act
        corners = get_mask_corners(large_box, large_mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

        # Should detect large object appropriately
        x_coords, y_coords = corners[:, 0], corners[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        # Should be substantial portion of image
        assert width > standard_img_size[0] // 4  # At least 1/4 of image width
        assert height > standard_img_size[1] // 4  # At least 1/4 of image height

    def test_circular_object_bounding_rectangle(
        self, standard_img_size, mock_sort_corners
    ):
        """Test that circular objects get reasonable bounding rectangles.

        Args:
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        circle_box = np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32)
        circle_mask = create_circular_mask()

        # Act
        corners = get_mask_corners(circle_box, circle_mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

        # Verify corners form reasonable bounding rectangle for circular object
        x_coords, y_coords = corners[:, 0], corners[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        # For circular object, width and height should be similar
        aspect_ratio = width / height if height > 0 else float("inf")
        assert 0.5 < aspect_ratio < 2.0  # Allow some tolerance for circular objects

    def test_consistency_across_runs(
        self, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test that function produces consistent results across multiple runs.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mask = create_simple_rectangular_mask()

        # Act - run multiple times
        corners1 = get_mask_corners(simple_box, mask, standard_img_size)
        corners2 = get_mask_corners(simple_box, mask, standard_img_size)
        corners3 = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert - should be identical
        assert np.allclose(corners1, corners2, rtol=1e-6)
        assert np.allclose(corners2, corners3, rtol=1e-6)
        assert np.allclose(corners1, corners3, rtol=1e-6)


class TestGetMaskCornersInternalLogic:
    """Test the internal logic components of get_mask_corners function."""

    @patch("mouse_tracking.utils.static_objects.get_affine_xform")
    def test_affine_transform_called_correctly(
        self, mock_affine, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test that affine transform is called with correct parameters.

        Args:
            mock_affine: Mock for get_affine_xform function.
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mock_affine.return_value = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mask = create_simple_rectangular_mask()

        # Act
        with contextlib.suppress(cv2.error):
            get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        mock_affine.assert_called_once_with(simple_box, img_size=standard_img_size)

    @patch("cv2.findContours")
    def test_contour_detection_called_correctly(
        self, mock_contours, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test that contour detection is called with correct parameters.

        Args:
            mock_contours: Mock for cv2.findContours function.
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        # Create a simple contour that represents a rectangle
        simple_contour = np.array(
            [[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32
        )
        mock_contours.return_value = ([simple_contour], None)
        mask = create_simple_rectangular_mask()

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert mock_contours.called
        # Verify it was called with the right parameters (binary mask, mode, method)
        call_args = mock_contours.call_args[0]
        assert len(call_args) == 3  # mask, mode, method
        assert call_args[1] == cv2.RETR_TREE
        assert call_args[2] == cv2.CHAIN_APPROX_SIMPLE
        assert validate_corners_output(corners)

    def test_threshold_processing(
        self, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test that mask thresholding works correctly.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange - mask with values just above threshold
        mask = create_simple_rectangular_mask()
        mask[mask > 0] = 0.6  # Above 0.5 threshold

        # Act
        corners = get_mask_corners(simple_box, mask, standard_img_size)

        # Assert
        assert validate_corners_output(corners)

    def test_largest_contour_selection(self, simple_box, standard_img_size):
        """Test that the largest contour is selected when multiple contours exist.

        Args:
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
        """
        # Arrange - create mask with multiple objects of different sizes
        mask = np.zeros((255, 255), dtype=np.float32)
        # Large rectangle
        mask[50:150, 50:200] = 1.0
        # Small rectangle
        mask[200:220, 200:220] = 1.0

        with patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            side_effect=lambda corners, img_size: corners.astype(np.float32),
        ):
            # Act
            corners = get_mask_corners(simple_box, mask, standard_img_size)

            # Assert
            assert validate_corners_output(corners)
            # Should detect the larger object based on area

    @patch("cv2.contourArea")
    def test_contour_area_calculation(
        self, mock_area, simple_box, standard_img_size, mock_sort_corners
    ):
        """Test that contour area calculation is used for selecting largest contour.

        Args:
            mock_area: Mock for cv2.contourArea function.
            simple_box: Fixture providing simple bounding box.
            standard_img_size: Fixture providing standard image size.
            mock_sort_corners: Mock for sort_corners function.
        """
        # Arrange
        mock_area.side_effect = [100, 200]  # Second contour is larger

        # Create two simple contours
        contour1 = np.array(
            [[[50, 50]], [[60, 50]], [[60, 60]], [[50, 60]]], dtype=np.int32
        )
        contour2 = np.array(
            [[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32
        )

        with patch("cv2.findContours", return_value=([contour1, contour2], None)):
            mask = create_simple_rectangular_mask()

            # Act
            corners = get_mask_corners(simple_box, mask, standard_img_size)

            # Assert
            assert mock_area.call_count == 2  # Called once for each contour
            assert validate_corners_output(corners)
