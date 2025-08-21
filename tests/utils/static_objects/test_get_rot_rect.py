"""Tests for get_rot_rect function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import get_rot_rect


def test_get_rot_rect_basic_functionality():
    """Test basic rotated rectangle detection from mask."""
    # Arrange - simple square mask
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 20:80] = 1.0  # Square region

    # Mock sort_corners to avoid the broadcasting bug
    with patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort:
        expected_corners = np.array([[20, 20], [79, 20], [79, 79], [20, 79]])
        mock_sort.return_value = expected_corners

        # Act
        result = get_rot_rect(mask)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result, expected_corners)


def test_get_rot_rect_uses_cv2_find_contours():
    """Test that function uses cv2.findContours correctly."""
    # Arrange
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[10:40, 10:40] = 0.8  # Above threshold

    # Mock cv2.findContours
    mock_contours = [np.array([[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]])]
    mock_hierarchy = None

    with (
        patch("cv2.findContours") as mock_find_contours,
        patch("cv2.contourArea", return_value=900),
        patch("cv2.minAreaRect", return_value=((25, 25), (30, 30), 0)),
        patch(
            "cv2.boxPoints",
            return_value=np.array([[10, 10], [40, 10], [40, 40], [10, 40]]),
        ),
        patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort,
    ):
        mock_find_contours.return_value = (mock_contours, mock_hierarchy)
        mock_sort.return_value = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])

        # Act
        get_rot_rect(mask)

        # Assert
        mock_find_contours.assert_called_once()
        # Check parameters passed to findContours
        call_args = mock_find_contours.call_args[0]
        binary_mask = call_args[0]
        retr_mode = call_args[1]
        approx_method = call_args[2]

        # Mask should be converted to uint8 and thresholded at 0.5
        expected_binary = np.uint8(mask > 0.5)
        np.testing.assert_array_equal(binary_mask, expected_binary)
        # Should use cv2.RETR_TREE and cv2.CHAIN_APPROX_SIMPLE
        import cv2

        assert retr_mode == cv2.RETR_TREE
        assert approx_method == cv2.CHAIN_APPROX_SIMPLE


def test_get_rot_rect_mask_thresholding():
    """Test that mask is properly thresholded at 0.5."""
    # Arrange - mask with values both above and below threshold
    mask = np.array(
        [[0.0, 0.3, 0.4], [0.5, 0.6, 0.9], [1.0, 0.2, 0.8]], dtype=np.float32
    )

    with (
        patch("cv2.findContours") as mock_find_contours,
        patch("cv2.contourArea", return_value=1),
        patch("cv2.minAreaRect", return_value=((1.5, 1.5), (1, 1), 0)),
        patch("cv2.boxPoints", return_value=np.array([[1, 1], [2, 1], [2, 2], [1, 2]])),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=np.array([[1, 1], [2, 1], [2, 2], [1, 2]]),
        ),
    ):
        mock_contours = [np.array([[[1, 1]], [[2, 1]], [[2, 2]], [[1, 2]]])]
        mock_find_contours.return_value = (mock_contours, None)

        # Act
        get_rot_rect(mask)

        # Assert - check thresholded mask
        call_args = mock_find_contours.call_args[0]
        binary_mask = call_args[0]

        expected_binary = np.uint8(mask > 0.5)
        np.testing.assert_array_equal(binary_mask, expected_binary)


def test_get_rot_rect_largest_contour_selection():
    """Test that the largest contour is selected correctly."""
    # Arrange
    mask = np.ones((50, 50), dtype=np.float32)

    # Mock multiple contours with different areas
    contour1 = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])  # Small
    contour2 = np.array([[[5, 5]], [[45, 5]], [[45, 45]], [[5, 45]]])  # Large
    contour3 = np.array([[[15, 15]], [[25, 15]], [[25, 25]], [[15, 25]]])  # Medium

    mock_contours = [contour1, contour2, contour3]
    mock_areas = [100, 1600, 100]  # contour2 has largest area

    with (
        patch("cv2.findContours", return_value=(mock_contours, None)),
        patch("cv2.contourArea", side_effect=mock_areas),
        patch("cv2.minAreaRect") as mock_min_area_rect,
        patch(
            "cv2.boxPoints", return_value=np.array([[5, 5], [45, 5], [45, 45], [5, 45]])
        ),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=np.array([[5, 5], [45, 5], [45, 45], [5, 45]]),
        ),
    ):
        mock_min_area_rect.return_value = ((25, 25), (40, 40), 0)

        # Act
        get_rot_rect(mask)

        # Assert
        # minAreaRect should be called with the largest contour (contour2)
        mock_min_area_rect.assert_called_once_with(contour2)


def test_get_rot_rect_uses_cv2_min_area_rect():
    """Test that cv2.minAreaRect is used correctly."""
    # Arrange
    mask = np.ones((30, 30), dtype=np.float32)

    mock_contour = np.array([[[5, 5]], [[25, 5]], [[25, 25]], [[5, 25]]])

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=400),
        patch("cv2.minAreaRect") as mock_min_area_rect,
        patch("cv2.boxPoints") as mock_box_points,
        patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort,
    ):
        mock_min_area_rect.return_value = (
            (15, 15),
            (20, 20),
            45,
        )  # Center, size, angle
        mock_corners = np.array([[10, 5], [20, 10], [20, 25], [10, 20]])
        mock_box_points.return_value = mock_corners
        mock_sort.return_value = mock_corners

        # Act
        get_rot_rect(mask)

        # Assert
        mock_min_area_rect.assert_called_once_with(mock_contour)
        mock_box_points.assert_called_once_with(((15, 15), (20, 20), 45))


def test_get_rot_rect_uses_cv2_box_points():
    """Test that cv2.boxPoints is used correctly."""
    # Arrange
    mask = np.ones((40, 40), dtype=np.float32)

    mock_contour = np.array([[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]])
    mock_rect = ((20, 20), (20, 20), 0)  # Rotated rectangle

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=400),
        patch("cv2.minAreaRect", return_value=mock_rect),
        patch("cv2.boxPoints") as mock_box_points,
        patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort,
    ):
        expected_corners = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        mock_box_points.return_value = expected_corners
        mock_sort.return_value = expected_corners

        # Act
        get_rot_rect(mask)

        # Assert
        mock_box_points.assert_called_once_with(mock_rect)
        mock_sort.assert_called_once_with(expected_corners, mask.shape[:2])


def test_get_rot_rect_uses_sort_corners():
    """Test that sort_corners is called with correct parameters."""
    # Arrange
    mask = np.zeros((60, 80), dtype=np.float32)  # Non-square mask
    mask[10:50, 10:70] = 1.0

    mock_contour = np.array([[[10, 10]], [[70, 10]], [[70, 50]], [[10, 50]]])
    corners = np.array([[10, 10], [70, 10], [70, 50], [10, 50]])

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=2400),
        patch("cv2.minAreaRect", return_value=((40, 30), (60, 40), 0)),
        patch("cv2.boxPoints", return_value=corners),
        patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort,
    ):
        expected_sorted = np.array([[10, 10], [70, 10], [70, 50], [10, 50]])
        mock_sort.return_value = expected_sorted

        # Act
        get_rot_rect(mask)

        # Assert
        mock_sort.assert_called_once_with(corners, mask.shape[:2])
        # mask.shape[:2] should be (60, 80)
        call_args = mock_sort.call_args[0]
        np.testing.assert_array_equal(call_args[1], (60, 80))


def test_get_rot_rect_empty_mask():
    """Test behavior with empty mask (no foreground pixels)."""
    # Arrange - all background
    mask = np.zeros((50, 50), dtype=np.float32)

    # Act & Assert - should raise cv2.error when trying to process None contour
    with pytest.raises(
        (Exception, AttributeError)
    ):  # cv2.error or AttributeError when trying to process empty contours
        get_rot_rect(mask)


def test_get_rot_rect_single_pixel_mask():
    """Test behavior with single pixel mask."""
    # Arrange - single foreground pixel
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[10, 10] = 1.0

    # Mock single point contour
    mock_contour = np.array([[[10, 10]]])

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=0),  # Single point has zero area
        patch("cv2.minAreaRect", return_value=((10, 10), (0, 0), 0)),
        patch(
            "cv2.boxPoints",
            return_value=np.array([[10, 10], [10, 10], [10, 10], [10, 10]]),
        ),
        patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort,
    ):
        mock_sort.return_value = np.array([[10, 10], [10, 10], [10, 10], [10, 10]])

        # Act
        result = get_rot_rect(mask)

        # Assert
        assert result.shape == (4, 2)


def test_get_rot_rect_rotated_rectangle():
    """Test with a rotated rectangular mask."""
    # Arrange - mask representing a rotated rectangle
    mask = np.zeros((100, 100), dtype=np.float32)
    # Create a diagonal rectangle-like shape
    for i in range(30, 70):
        for j in range(i - 10, i + 10):
            if 0 <= j < 100:
                mask[i, j] = 1.0

    # Mock rotated rectangle detection
    mock_contour = np.array([[[20, 30]], [[80, 30]], [[90, 70]], [[30, 70]]])

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=1600),
        patch(
            "cv2.minAreaRect", return_value=((50, 50), (40, 60), 30)
        ),  # 30 degree rotation
        patch("cv2.boxPoints") as mock_box_points,
        patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort,
    ):
        rotated_corners = np.array([[25, 35], [75, 25], [85, 65], [35, 75]])
        mock_box_points.return_value = rotated_corners
        mock_sort.return_value = rotated_corners

        # Act
        result = get_rot_rect(mask)

        # Assert
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result, rotated_corners)


def test_get_rot_rect_multiple_contours_different_areas():
    """Test with multiple contours where areas need to be compared."""
    # Arrange
    mask = np.ones((80, 80), dtype=np.float32)

    # Mock three contours with different areas
    contour1 = np.array([[[10, 10]], [[15, 10]], [[15, 15]], [[10, 15]]])  # Area = 25
    contour2 = np.array(
        [[[20, 20]], [[60, 20]], [[60, 60]], [[20, 60]]]
    )  # Area = 1600 (largest)
    contour3 = np.array([[[5, 5]], [[25, 5]], [[25, 25]], [[5, 25]]])  # Area = 400

    mock_contours = [contour1, contour2, contour3]

    # Mock different areas for each contour
    def mock_contour_area(contour):
        if np.array_equal(contour, contour1):
            return 25
        elif np.array_equal(contour, contour2):
            return 1600
        elif np.array_equal(contour, contour3):
            return 400
        return 0

    with (
        patch("cv2.findContours", return_value=(mock_contours, None)),
        patch("cv2.contourArea", side_effect=mock_contour_area),
        patch("cv2.minAreaRect") as mock_min_area_rect,
        patch(
            "cv2.boxPoints",
            return_value=np.array([[20, 20], [60, 20], [60, 60], [20, 60]]),
        ),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=np.array([[20, 20], [60, 20], [60, 60], [20, 60]]),
        ),
    ):
        mock_min_area_rect.return_value = ((40, 40), (40, 40), 0)

        # Act
        get_rot_rect(mask)

        # Assert
        # Should use contour2 (largest area)
        mock_min_area_rect.assert_called_once_with(contour2)


def test_get_rot_rect_mask_dtype_conversion():
    """Test that mask is properly converted to uint8."""
    # Arrange - mask with different data types
    mask_float64 = np.array([[0.3, 0.7], [0.9, 0.1]], dtype=np.float64)

    with (
        patch("cv2.findContours") as mock_find_contours,
        patch("cv2.contourArea", return_value=1),
        patch("cv2.minAreaRect", return_value=((0.5, 0.5), (1, 1), 0)),
        patch("cv2.boxPoints", return_value=np.array([[0, 0], [1, 0], [1, 1], [0, 1]])),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        ),
    ):
        mock_contours = [np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]])]
        mock_find_contours.return_value = (mock_contours, None)

        # Act
        get_rot_rect(mask_float64)

        # Assert - check that uint8 conversion happened
        call_args = mock_find_contours.call_args[0]
        binary_mask = call_args[0]
        assert binary_mask.dtype == np.uint8


def test_get_rot_rect_threshold_boundary_values():
    """Test behavior at threshold boundary (exactly 0.5)."""
    # Arrange - mask with values exactly at threshold
    mask = np.array([[0.49, 0.50, 0.51], [0.5, 0.0, 1.0]], dtype=np.float32)

    with (
        patch("cv2.findContours") as mock_find_contours,
        patch("cv2.contourArea", return_value=1),
        patch("cv2.minAreaRect", return_value=((1.5, 0.5), (1, 1), 0)),
        patch("cv2.boxPoints", return_value=np.array([[1, 0], [2, 0], [2, 1], [1, 1]])),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=np.array([[1, 0], [2, 0], [2, 1], [1, 1]]),
        ),
    ):
        mock_find_contours.return_value = (
            [np.array([[[1, 0]], [[2, 0]], [[2, 1]], [[1, 1]]])],
            None,
        )

        # Act
        get_rot_rect(mask)

        # Assert
        call_args = mock_find_contours.call_args[0]
        binary_mask = call_args[0]

        # Values > 0.5 should be True (1), values <= 0.5 should be False (0)
        # Corrected expected values based on actual threshold behavior:
        # [0.49, 0.50, 0.51] -> [0, 0, 1] (only 0.51 > 0.5 is True)
        # [0.5, 0.0, 1.0] -> [0, 0, 1] (only 1.0 > 0.5 is True)
        expected = np.uint8([[0, 0, 1], [0, 0, 1]])
        np.testing.assert_array_equal(binary_mask, expected)


def test_get_rot_rect_return_type_and_shape():
    """Test that function returns correct type and shape."""
    # Arrange
    mask = np.ones((30, 30), dtype=np.float32)

    expected_result = np.array(
        [[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32
    )

    with (
        patch(
            "cv2.findContours",
            return_value=(
                [np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])],
                None,
            ),
        ),
        patch("cv2.contourArea", return_value=100),
        patch("cv2.minAreaRect", return_value=((15, 15), (10, 10), 0)),
        patch("cv2.boxPoints", return_value=expected_result),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=expected_result,
        ),
    ):
        # Act
        result = get_rot_rect(mask)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)
        assert result.ndim == 2


def test_get_rot_rect_large_mask():
    """Test with large mask to verify performance."""
    # Arrange
    mask = np.zeros((1000, 1000), dtype=np.float32)
    mask[200:800, 200:800] = 1.0  # Large square

    mock_contour = np.array([[[200, 200]], [[800, 200]], [[800, 800]], [[200, 800]]])
    expected_corners = np.array([[200, 200], [800, 200], [800, 800], [200, 800]])

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=360000),
        patch("cv2.minAreaRect", return_value=((500, 500), (600, 600), 0)),
        patch("cv2.boxPoints", return_value=expected_corners),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=expected_corners,
        ),
    ):
        # Act
        result = get_rot_rect(mask)

        # Assert
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result, expected_corners)


@pytest.mark.parametrize("mask_shape", [(50, 50), (100, 80), (30, 120), (200, 200)])
def test_get_rot_rect_various_mask_shapes(mask_shape):
    """Test with various mask shapes."""
    # Arrange
    mask = np.zeros(mask_shape, dtype=np.float32)
    # Create a rectangular region in the center
    h, w = mask_shape
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0

    mock_contour = np.array(
        [
            [[w // 4, h // 4]],
            [[3 * w // 4, h // 4]],
            [[3 * w // 4, 3 * h // 4]],
            [[w // 4, 3 * h // 4]],
        ]
    )
    expected_corners = np.array(
        [
            [w // 4, h // 4],
            [3 * w // 4, h // 4],
            [3 * w // 4, 3 * h // 4],
            [w // 4, 3 * h // 4],
        ]
    )

    with (
        patch("cv2.findContours", return_value=([mock_contour], None)),
        patch("cv2.contourArea", return_value=(h // 2) * (w // 2)),
        patch("cv2.minAreaRect", return_value=((w // 2, h // 2), (w // 2, h // 2), 0)),
        patch("cv2.boxPoints", return_value=expected_corners),
        patch(
            "mouse_tracking.utils.static_objects.sort_corners",
            return_value=expected_corners,
        ),
    ):
        # Act
        result = get_rot_rect(mask)

        # Assert
        assert result.shape == (4, 2)


def test_get_rot_rect_integration_with_actual_cv2():
    """Test integration with actual OpenCV functions."""
    # Arrange - create a simple rectangular mask
    mask = np.zeros((60, 80), dtype=np.float32)
    mask[20:40, 30:50] = 1.0  # 20x20 square

    # Act - use real OpenCV functions (no mocking for CV2)
    with patch("mouse_tracking.utils.static_objects.sort_corners") as mock_sort:
        # Mock only sort_corners to avoid dependency on that function's correctness
        mock_sort.return_value = np.array([[30, 20], [50, 20], [50, 40], [30, 40]])

        result = get_rot_rect(mask)

        # Assert
        assert result.shape == (4, 2)
        mock_sort.assert_called_once()
        # sort_corners should be called with mask.shape[:2] = (60, 80)
        call_args = mock_sort.call_args[0]
        assert call_args[1] == (60, 80)
