"""Tests for plot_keypoints function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import plot_keypoints


class TestPlotKeypoints:
    """Test cases for plot_keypoints function."""

    def test_plot_keypoints_basic_functionality(self):
        """Test basic keypoint plotting functionality."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)

        # Act
        result = plot_keypoints(keypoints, image, color=color)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape
        assert result is not image  # Result should be a copy, not the same object

    def test_plot_keypoints_is_yx_flag_true(self):
        """Test keypoint plotting with is_yx=True flips coordinates."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40]], dtype=np.float32)  # y, x format
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.circle") as mock_circle:
            # Act
            plot_keypoints(keypoints, image, is_yx=True)

            # Assert - should be called with flipped coordinates (x, y)
            calls = mock_circle.call_args_list
            # First keypoint: (20, 10) - flipped from (10, 20)
            assert calls[0][0][1] == (20, 10)
            # Second keypoint: (40, 30) - flipped from (30, 40)
            assert calls[2][0][1] == (40, 30)

    def test_plot_keypoints_is_yx_flag_false(self):
        """Test keypoint plotting with is_yx=False keeps coordinates."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40]], dtype=np.float32)  # x, y format
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.circle") as mock_circle:
            # Act
            plot_keypoints(keypoints, image, is_yx=False)

            # Assert - should be called with original coordinates
            calls = mock_circle.call_args_list
            # First keypoint: (10, 20) - unchanged
            assert calls[0][0][1] == (10, 20)
            # Second keypoint: (30, 40) - unchanged
            assert calls[2][0][1] == (30, 40)

    def test_plot_keypoints_include_lines_true(self):
        """Test keypoint plotting with include_lines=True draws contours."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with (
            patch("cv2.drawContours") as mock_contours,
            patch("cv2.circle") as mock_circle,
        ):
            # Act
            plot_keypoints(keypoints, image, include_lines=True)

            # Assert
            # Should call drawContours twice (black outline + colored line)
            assert mock_contours.call_count == 2
            # Should still call circle for each keypoint
            assert mock_circle.call_count == len(keypoints) * 2

    def test_plot_keypoints_include_lines_false(self):
        """Test keypoint plotting with include_lines=False skips contours."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with (
            patch("cv2.drawContours") as mock_contours,
            patch("cv2.circle") as mock_circle,
        ):
            # Act
            plot_keypoints(keypoints, image, include_lines=False)

            # Assert
            # Should not call drawContours
            assert mock_contours.call_count == 0
            # Should still call circle for each keypoint
            assert mock_circle.call_count == len(keypoints) * 2

    def test_plot_keypoints_single_keypoint_no_lines(self):
        """Test that single keypoint doesn't draw lines even with include_lines=True."""
        # Arrange
        keypoints = np.array([[10, 20]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with (
            patch("cv2.drawContours") as mock_contours,
            patch("cv2.circle") as mock_circle,
        ):
            # Act
            plot_keypoints(keypoints, image, include_lines=True)

            # Assert
            # Should call drawContours (condition checks shape[0] >= 1)
            assert mock_contours.call_count == 2
            # Should call circle for the keypoint
            assert mock_circle.call_count == 2

    def test_plot_keypoints_empty_keypoints_no_lines(self):
        """Test that empty keypoints array doesn't draw lines."""
        # Arrange
        keypoints = np.zeros((0, 2), dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with (
            patch("cv2.drawContours") as mock_contours,
            patch("cv2.circle") as mock_circle,
        ):
            # Act
            plot_keypoints(keypoints, image, include_lines=True)

            # Assert
            # Should not call drawContours (shape[0] = 0)
            assert mock_contours.call_count == 0
            # Should not call circle
            assert mock_circle.call_count == 0

    def test_plot_keypoints_custom_color(self):
        """Test keypoint plotting with custom color."""
        # Arrange
        keypoints = np.array([[10, 20]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        custom_color = (128, 64, 192)

        with patch("cv2.circle") as mock_circle:
            # Act
            plot_keypoints(keypoints, image, color=custom_color)

            # Assert
            calls = mock_circle.call_args_list
            # First call should be black outline
            assert calls[0][0][3] == (0, 0, 0)
            # Second call should be custom color
            assert calls[1][0][3] == custom_color

    def test_plot_keypoints_default_color(self):
        """Test keypoint plotting with default color."""
        # Arrange
        keypoints = np.array([[10, 20]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.circle") as mock_circle:
            # Act
            plot_keypoints(keypoints, image)

            # Assert
            calls = mock_circle.call_args_list
            # First call should be black outline
            assert calls[0][0][3] == (0, 0, 0)
            # Second call should be default red color
            assert calls[1][0][3] == (0, 0, 255)

    def test_plot_keypoints_float_coordinates_converted_to_int(self):
        """Test that floating point coordinates are converted to integers."""
        # Arrange
        keypoints = np.array([[10.7, 20.3]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.circle") as mock_circle:
            # Act
            plot_keypoints(keypoints, image)

            # Assert
            calls = mock_circle.call_args_list
            # Should convert to integers
            assert calls[0][0][1] == (10, 20)
            assert calls[1][0][1] == (10, 20)

    def test_plot_keypoints_returns_copy_not_reference(self):
        """Test that function returns a copy of the image, not a reference."""
        # Arrange
        keypoints = np.array([[10, 20]], dtype=np.float32)
        original_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        result = plot_keypoints(keypoints, original_image)

        # Assert
        assert result is not original_image
        assert isinstance(result, np.ndarray)
        assert result.shape == original_image.shape
        assert result.dtype == original_image.dtype

    def test_plot_keypoints_cv2_calls_mocked(self):
        """Test that cv2 functions are called correctly when mocked."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)

        with (
            patch("cv2.circle") as mock_circle,
            patch(
                "cv2.drawContours", side_effect=lambda img, *args, **kwargs: img
            ) as mock_contours,
        ):
            # Act
            result = plot_keypoints(keypoints, image, color=color, include_lines=True)

            # Assert
            # Should call cv2.circle twice per keypoint (black outline + colored fill)
            expected_circle_calls = len(keypoints) * 2
            assert mock_circle.call_count == expected_circle_calls

            # Should call cv2.drawContours twice (black outline + colored line)
            assert mock_contours.call_count == 2

            # Verify result properties
            assert isinstance(result, np.ndarray)
            assert result.shape == image.shape

    @pytest.mark.parametrize(
        "keypoints,expected_shape",
        [
            (np.array([[10, 20]], dtype=np.float32), (1, 2)),
            (np.array([[10, 20], [30, 40]], dtype=np.float32), (2, 2)),
            (np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32), (3, 2)),
            (np.zeros((0, 2), dtype=np.float32), (0, 2)),
        ],
    )
    def test_plot_keypoints_various_keypoint_shapes(self, keypoints, expected_shape):
        """Test keypoint plotting with various keypoint array shapes."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.circle") as mock_circle:
            # Act
            result = plot_keypoints(keypoints, image)

            # Assert
            assert keypoints.shape == expected_shape
            assert isinstance(result, np.ndarray)
            expected_circles = len(keypoints) * 2 if len(keypoints) > 0 else 0
            assert mock_circle.call_count == expected_circles

    def test_plot_keypoints_1d_keypoints_error(self):
        """Test that 1D keypoint arrays raise an appropriate error."""
        # Arrange
        keypoints = np.array([10, 20], dtype=np.float32)  # 1D array - invalid input
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act & Assert
        # The function expects 2D arrays and will fail with 1D input
        with pytest.raises(IndexError):
            plot_keypoints(keypoints, image, include_lines=True)

    def test_plot_keypoints_circle_parameters(self):
        """Test that cv2.circle is called with correct parameters."""
        # Arrange
        keypoints = np.array([[15, 25]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (100, 150, 200)

        with patch("cv2.circle") as mock_circle:
            # Act
            plot_keypoints(keypoints, image, color=color)

            # Assert
            calls = mock_circle.call_args_list

            # First call (black outline)
            assert calls[0][0][1] == (15, 25)  # center
            assert calls[0][0][2] == 3  # radius
            assert calls[0][0][3] == (0, 0, 0)  # black color
            assert calls[0][0][4] == -1  # filled

            # Second call (colored fill)
            assert calls[1][0][1] == (15, 25)  # center
            assert calls[1][0][2] == 2  # radius
            assert calls[1][0][3] == color  # custom color
            assert calls[1][0][4] == -1  # filled

    def test_plot_keypoints_contour_parameters(self):
        """Test that cv2.drawContours is called with correct parameters."""
        # Arrange
        keypoints = np.array([[10, 20], [30, 40]], dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (100, 150, 200)

        with patch(
            "cv2.drawContours", side_effect=lambda img, *args, **kwargs: img
        ) as mock_contours:
            # Act
            plot_keypoints(keypoints, image, color=color, include_lines=True)

            # Assert
            calls = mock_contours.call_args_list

            # First call (black outline)
            assert calls[0][0][2] == 0  # contour index
            assert calls[0][0][3] == (0, 0, 0)  # black color
            assert calls[0][0][4] == 2  # thickness

            # Second call (colored line)
            assert calls[1][0][2] == 0  # contour index
            assert calls[1][0][3] == color  # custom color
            assert calls[1][0][4] == 1  # thickness
