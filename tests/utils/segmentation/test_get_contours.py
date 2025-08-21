"""
Unit tests for the get_contours function from mouse_tracking.utils.segmentation.

This module tests the get_contours function which processes binary masks to extract
OpenCV-compliant contours and hierarchy information, with filtering based on contour area.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import get_contours


class TestGetContours:
    """Test class for get_contours function."""

    def test_empty_mask_returns_empty_arrays(self):
        """Test that an empty mask returns correctly formatted empty arrays."""
        # Arrange
        mask = np.zeros((100, 100), dtype=np.uint8)

        # Act
        contours, hierarchy = get_contours(mask)

        # Assert
        assert isinstance(contours, list)
        assert isinstance(hierarchy, list)
        assert len(contours) == 1
        assert len(hierarchy) == 1

        # Check the format of empty arrays
        expected_empty_contour = np.zeros([0, 2], dtype=np.int32)
        expected_empty_hierarchy = np.zeros([0, 4], dtype=np.int32)

        np.testing.assert_array_equal(contours[0], expected_empty_contour)
        np.testing.assert_array_equal(hierarchy[0], expected_empty_hierarchy)

    def test_all_zero_mask_returns_empty_arrays(self):
        """Test that a mask with all zeros returns empty arrays."""
        # Arrange
        mask = np.zeros((50, 50), dtype=np.float32)

        # Act
        contours, hierarchy = get_contours(mask)

        # Assert
        assert len(contours) == 1
        assert len(hierarchy) == 1
        assert contours[0].shape == (0, 2)
        assert hierarchy[0].shape == (0, 4)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_contours_above_threshold_returned(self, mock_area, mock_find_contours):
        """Test that contours above area threshold are returned."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 50.0

        # Mock contours and hierarchy
        mock_contour1 = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_contour2 = np.array(
            [[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32
        )
        mock_contours = [mock_contour1, mock_contour2]
        mock_hierarchy = np.array([[[0, 1, -1, -1], [1, 0, -1, -1]]], dtype=np.int32)

        mock_find_contours.return_value = (mock_contours, mock_hierarchy)
        mock_area.side_effect = [100.0, 75.0]  # Both above threshold

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        assert mock_area.call_count == 2
        assert len(contours) == 2
        np.testing.assert_array_equal(contours[0], mock_contour1)
        np.testing.assert_array_equal(contours[1], mock_contour2)
        np.testing.assert_array_equal(hierarchy, mock_hierarchy)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_contours_below_threshold_filtered_out(self, mock_area, mock_find_contours):
        """Test that contours below area threshold are filtered out."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 50.0

        # Mock contours and hierarchy
        mock_contour1 = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_contour2 = np.array(
            [[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32
        )
        mock_contour3 = np.array(
            [[[50, 50]], [[60, 50]], [[60, 60]], [[50, 60]]], dtype=np.int32
        )
        mock_contours = [mock_contour1, mock_contour2, mock_contour3]
        mock_hierarchy = np.array(
            [[[0, 1, -1, -1], [1, 2, -1, -1], [2, 0, -1, -1]]], dtype=np.int32
        )

        mock_find_contours.return_value = (mock_contours, mock_hierarchy)
        mock_area.side_effect = [25.0, 75.0, 30.0]  # Only middle one above threshold

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        assert mock_area.call_count == 3
        assert len(contours) == 1
        np.testing.assert_array_equal(contours[0], mock_contour2)
        # Check that hierarchy is properly filtered
        expected_hierarchy = np.array([[[1, 2, -1, -1]]], dtype=np.int32).reshape(
            [1, -1, 4]
        )
        np.testing.assert_array_equal(hierarchy, expected_hierarchy)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_all_contours_below_threshold_returns_empty(
        self, mock_area, mock_find_contours
    ):
        """Test that when all contours are below threshold, empty arrays are returned."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 100.0

        # Mock contours and hierarchy
        mock_contour1 = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_contour2 = np.array(
            [[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32
        )
        mock_contours = [mock_contour1, mock_contour2]
        mock_hierarchy = np.array([[[0, 1, -1, -1], [1, 0, -1, -1]]], dtype=np.int32)

        mock_find_contours.return_value = (mock_contours, mock_hierarchy)
        mock_area.side_effect = [25.0, 50.0]  # Both below threshold

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        assert mock_area.call_count == 2
        assert len(contours) == 1
        assert len(hierarchy) == 1
        assert contours[0].shape == (0, 2)
        assert hierarchy[0].shape == (0, 4)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_zero_min_area_returns_all_contours(self, mock_area, mock_find_contours):
        """Test that zero minimum area returns all contours without filtering."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 0.0

        # Mock contours and hierarchy
        mock_contour1 = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_contour2 = np.array(
            [[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32
        )
        mock_contours = [mock_contour1, mock_contour2]
        mock_hierarchy = np.array([[[0, 1, -1, -1], [1, 0, -1, -1]]], dtype=np.int32)

        mock_find_contours.return_value = (mock_contours, mock_hierarchy)

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        mock_area.assert_not_called()  # Should not filter when min_area is 0
        assert len(contours) == 2
        np.testing.assert_array_equal(contours[0], mock_contour1)
        np.testing.assert_array_equal(contours[1], mock_contour2)
        np.testing.assert_array_equal(hierarchy, mock_hierarchy)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_negative_min_area_returns_all_contours(
        self, mock_area, mock_find_contours
    ):
        """Test that negative minimum area returns all contours without filtering."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = -10.0

        # Mock contours and hierarchy
        mock_contour1 = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_contours = [mock_contour1]
        mock_hierarchy = np.array([[[0, 0, -1, -1]]], dtype=np.int32)

        mock_find_contours.return_value = (mock_contours, mock_hierarchy)

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        mock_area.assert_not_called()  # Should not filter when min_area <= 0
        assert len(contours) == 1
        np.testing.assert_array_equal(contours[0], mock_contour1)
        np.testing.assert_array_equal(hierarchy, mock_hierarchy)

    @patch("cv2.findContours")
    def test_opencv_called_with_correct_parameters(self, mock_find_contours):
        """Test that OpenCV findContours is called with correct parameters."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.float32)
        mock_find_contours.return_value = ([], np.array([]))

        # Act
        get_contours(mask)

        # Assert
        mock_find_contours.assert_called_once()
        call_args = mock_find_contours.call_args[0]

        # Check that mask is converted to uint8
        np.testing.assert_array_equal(call_args[0], mask.astype(np.uint8))

        # Check OpenCV parameters
        import cv2

        assert call_args[1] == cv2.RETR_CCOMP
        assert call_args[2] == cv2.CHAIN_APPROX_SIMPLE

    @patch("cv2.findContours")
    def test_mask_conversion_to_uint8(self, mock_find_contours):
        """Test that mask is properly converted to uint8 before processing."""
        # Arrange
        mask = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 0.3]], dtype=np.float32)
        mock_find_contours.return_value = ([], np.array([]))

        # Act
        get_contours(mask)

        # Assert
        mock_find_contours.assert_called_once()
        call_args = mock_find_contours.call_args[0]

        # Check that mask is converted to uint8
        expected_mask = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(call_args[0], expected_mask)

    @pytest.mark.parametrize("mask_dtype", [np.uint8, np.float32, np.int32, np.bool_])
    def test_different_mask_data_types(self, mask_dtype):
        """Test that function handles different mask data types correctly."""
        # Arrange
        mask = np.array([[0, 1, 0], [1, 1, 1]], dtype=mask_dtype)

        with patch("cv2.findContours") as mock_find_contours:
            mock_find_contours.return_value = ([], np.array([]))

            # Act
            get_contours(mask)

            # Assert
            mock_find_contours.assert_called_once()
            call_args = mock_find_contours.call_args[0]

            # Should always convert to uint8
            assert call_args[0].dtype == np.uint8

    @pytest.mark.parametrize("min_area", [0.0, 1.0, 25.0, 50.0, 100.0, 500.0])
    def test_various_min_area_thresholds(self, min_area):
        """Test function with various minimum area thresholds."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)

        with (
            patch("cv2.findContours") as mock_find_contours,
            patch("cv2.contourArea") as mock_area,
        ):
            mock_contour = np.array(
                [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
            )
            mock_find_contours.return_value = (
                [mock_contour],
                np.array([[[0, 0, -1, -1]]]),
            )
            mock_area.return_value = 75.0

            # Act
            contours, hierarchy = get_contours(mask, min_area)

            # Assert
            mock_find_contours.assert_called_once()

            if min_area <= 0:
                mock_area.assert_not_called()
                assert len(contours) == 1
            elif min_area <= 75.0:
                mock_area.assert_called_once()
                assert len(contours) == 1
            else:
                mock_area.assert_called_once()
                assert len(contours) == 1
                assert contours[0].shape == (0, 2)

    @patch("cv2.findContours")
    def test_no_contours_found_returns_empty(self, mock_find_contours):
        """Test that when no contours are found, empty arrays are returned."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        mock_find_contours.return_value = ([], np.array([]))

        # Act
        contours, hierarchy = get_contours(mask)

        # Assert
        mock_find_contours.assert_called_once()
        assert len(contours) == 1
        assert len(hierarchy) == 1
        assert contours[0].shape == (0, 2)
        assert hierarchy[0].shape == (0, 4)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_hierarchy_filtering_matches_contour_filtering(
        self, mock_area, mock_find_contours
    ):
        """Test that hierarchy is filtered to match contour filtering."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 50.0

        # Mock 3 contours with different areas
        mock_contour1 = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_contour2 = np.array(
            [[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32
        )
        mock_contour3 = np.array(
            [[[50, 50]], [[60, 50]], [[60, 60]], [[50, 60]]], dtype=np.int32
        )
        mock_contours = [mock_contour1, mock_contour2, mock_contour3]

        # Mock hierarchy with 3 entries
        mock_hierarchy = np.array(
            [[[0, 1, -1, -1], [1, 2, -1, -1], [2, 0, -1, -1]]], dtype=np.int32
        )

        mock_find_contours.return_value = (mock_contours, mock_hierarchy)
        mock_area.side_effect = [
            25.0,
            75.0,
            100.0,
        ]  # First below, second and third above threshold

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        assert mock_area.call_count == 3
        assert len(contours) == 2

        # Check that contours 1 and 2 are returned (indices 1 and 2 from original)
        np.testing.assert_array_equal(contours[0], mock_contour2)
        np.testing.assert_array_equal(contours[1], mock_contour3)

        # Check that hierarchy is properly filtered (indices 1 and 2 from original)
        expected_hierarchy = mock_hierarchy[0, [1, 2], :].reshape([1, -1, 4])
        np.testing.assert_array_equal(hierarchy, expected_hierarchy)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_single_contour_above_threshold(self, mock_area, mock_find_contours):
        """Test with single contour above threshold."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 50.0

        mock_contour = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_hierarchy = np.array([[[0, 0, -1, -1]]], dtype=np.int32)

        mock_find_contours.return_value = ([mock_contour], mock_hierarchy)
        mock_area.return_value = 75.0

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        mock_area.assert_called_once()
        assert len(contours) == 1
        np.testing.assert_array_equal(contours[0], mock_contour)
        np.testing.assert_array_equal(hierarchy, mock_hierarchy)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_single_contour_below_threshold(self, mock_area, mock_find_contours):
        """Test with single contour below threshold."""
        # Arrange
        mask = np.ones((100, 100), dtype=np.uint8)
        min_area = 100.0

        mock_contour = np.array(
            [[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32
        )
        mock_hierarchy = np.array([[[0, 0, -1, -1]]], dtype=np.int32)

        mock_find_contours.return_value = ([mock_contour], mock_hierarchy)
        mock_area.return_value = 75.0

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        mock_find_contours.assert_called_once()
        mock_area.assert_called_once()
        assert len(contours) == 1
        assert len(hierarchy) == 1
        assert contours[0].shape == (0, 2)
        assert hierarchy[0].shape == (0, 4)

    def test_integration_with_actual_mask(self):
        """Integration test with actual mask data (without mocking OpenCV)."""
        # Arrange - create a simple binary mask with a rectangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # Create a 50x50 rectangle
        min_area = 100.0

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        # When contours are found, OpenCV returns a tuple; when empty, function returns a list
        assert isinstance(contours, list | tuple)
        # When contours are found, hierarchy is a numpy array; when empty, it's a list
        assert isinstance(hierarchy, list | np.ndarray)
        assert len(contours) >= 1

        # Should find at least one contour for the rectangle
        if len(contours) > 0 and contours[0].shape[0] > 0:
            # OpenCV contours have shape [n_points, 1, 2] where last dimension is [x, y]
            assert contours[0].shape[2] == 2  # Each contour point has x,y coordinates
            if isinstance(hierarchy, np.ndarray):
                assert hierarchy.shape[2] == 4  # Hierarchy has 4 components per contour
            else:
                assert hierarchy[0].shape[1] == 4  # Empty case format

    def test_edge_case_single_pixel_mask(self):
        """Test edge case with single pixel mask."""
        # Arrange
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255  # Single pixel
        min_area = 0.0

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        # When contours are found, OpenCV returns a tuple; when empty, function returns a list
        assert isinstance(contours, list | tuple)
        # When contours are found, hierarchy is a numpy array; when empty, it's a list
        assert isinstance(hierarchy, list | np.ndarray)
        # Single pixel might not form a valid contour in OpenCV
        assert len(contours) >= 1

    def test_edge_case_very_small_mask(self):
        """Test edge case with very small mask."""
        # Arrange
        mask = np.ones((2, 2), dtype=np.uint8)
        min_area = 0.0

        # Act
        contours, hierarchy = get_contours(mask, min_area)

        # Assert
        # When contours are found, OpenCV returns a tuple; when empty, function returns a list
        assert isinstance(contours, list | tuple)
        # When contours are found, hierarchy is a numpy array; when empty, it's a list
        assert isinstance(hierarchy, list | np.ndarray)
        assert len(contours) >= 1
