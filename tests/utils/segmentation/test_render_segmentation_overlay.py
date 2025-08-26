"""Unit tests for render_segmentation_overlay function.

This module contains comprehensive tests for the render_segmentation_overlay function from
the mouse_tracking.utils.segmentation module, including edge cases and error conditions.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import render_segmentation_overlay


class TestRenderSegmentationOverlay:
    """Test cases for render_segmentation_overlay function."""

    def test_render_segmentation_overlay_basic_functionality(self):
        """Test basic functionality with RGB image."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)  # Red color
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            assert not np.array_equal(result, image)  # Should be modified
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert np.array_equal(call_args[0][0], contour)
            assert call_args[1]["frame_size"] == (100, 100)
            # Check that color was applied to outline pixels
            assert np.all(result[expected_outline] == color)

    def test_render_segmentation_overlay_with_all_padding_contour(self):
        """Test behavior when contour is all padding values."""
        # Arrange
        contour = np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]])
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        color = (0, 255, 0)  # Green color

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert - should return original image unchanged
            assert result.shape == (50, 50, 3)
            assert result.dtype == np.uint8
            assert np.array_equal(result, image)
            mock_render_outline.assert_not_called()

    def test_render_segmentation_overlay_with_grayscale_image(self):
        """Test conversion from grayscale to RGB."""
        # Arrange
        contour = np.array([[[5, 10], [15, 20]], [[-1, -1], [-1, -1]]])
        image = np.zeros((50, 50, 1), dtype=np.uint8)
        color = (255, 255, 0)  # Yellow color
        expected_outline = np.zeros((50, 50), dtype=bool)
        expected_outline[10:20, 10:20] = True

        with (
            patch(
                "mouse_tracking.utils.segmentation.render_outline"
            ) as mock_render_outline,
            patch("mouse_tracking.utils.segmentation.cv2.cvtColor") as mock_cvt_color,
        ):
            mock_render_outline.return_value = expected_outline
            # Mock cv2.cvtColor to return RGB version
            rgb_image = np.zeros((50, 50, 3), dtype=np.uint8)
            mock_cvt_color.return_value = rgb_image

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (50, 50, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            mock_cvt_color.assert_called_once()
            # Check the call args manually to avoid numpy array comparison issues
            call_args = mock_cvt_color.call_args
            assert call_args[0][0].shape == (
                50,
                50,
                1,
            )  # first arg should be the grayscale image copy
            # Second argument should be the OpenCV constant for converting grayscale to RGB
            # We can't easily compare with cv2.COLOR_GRAY2RGB since it's imported, just check it's an integer
            assert isinstance(call_args[0][1], int)
            # Check that color was applied to outline pixels
            assert np.all(result[expected_outline] == color)

    def test_render_segmentation_overlay_with_rgb_image_no_conversion(self):
        """Test RGB image doesn't get converted."""
        # Arrange
        contour = np.array([[[5, 10], [15, 20]], [[-1, -1], [-1, -1]]])
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        color = (0, 0, 255)  # Blue color
        expected_outline = np.zeros((50, 50), dtype=bool)
        expected_outline[10:20, 10:20] = True

        with (
            patch(
                "mouse_tracking.utils.segmentation.render_outline"
            ) as mock_render_outline,
            patch("mouse_tracking.utils.segmentation.cv2.cvtColor") as mock_cvt_color,
        ):
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (50, 50, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            mock_cvt_color.assert_not_called()  # Should not be called for RGB images
            # Check that color was applied to outline pixels
            assert np.all(result[expected_outline] == color)

    @pytest.mark.parametrize(
        "color",
        [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 255),  # White
            (0, 0, 0),  # Black
            (128, 64, 192),  # Custom color
        ],
    )
    def test_render_segmentation_overlay_with_different_colors(self, color):
        """Test rendering with different color values."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            # Check that correct color was applied
            assert np.all(result[expected_outline] == color)

    def test_render_segmentation_overlay_with_default_color(self):
        """Test rendering with default color (red)."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image)  # No color specified

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            # Check that default color (0, 0, 255) was applied
            assert np.all(result[expected_outline] == (0, 0, 255))

    def test_render_segmentation_overlay_preserves_original_image(self):
        """Test that original image is not modified."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original_image = image.copy()
        color = (255, 0, 0)
        expected_outline = np.zeros((100, 100), dtype=bool)
        expected_outline[10:20, 10:20] = True

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert np.array_equal(image, original_image)  # Original should be unchanged
            assert not np.array_equal(result, image)  # Result should be different
            # Check that non-outline pixels are unchanged
            assert np.all(result[~expected_outline] == image[~expected_outline])

    def test_render_segmentation_overlay_with_partial_contour(self):
        """Test rendering with contour that has some padding."""
        # Arrange
        contour = np.array(
            [[[10, 20], [30, 40], [50, 60]], [[-1, -1], [-1, -1], [-1, -1]]]
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (128, 128, 128)  # Gray color
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert np.array_equal(call_args[0][0], contour)
            assert call_args[1]["frame_size"] == (100, 100)

    def test_render_segmentation_overlay_with_2d_contour(self):
        """Test rendering with 2D contour input."""
        # Arrange
        contour = np.array([[10, 20], [30, 40], [50, 60]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 128, 0)  # Orange color
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert np.array_equal(call_args[0][0], contour)

    def test_render_segmentation_overlay_with_empty_outline(self):
        """Test rendering when outline is empty (all False)."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        empty_outline = np.zeros((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = empty_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            # Should be same as original since no outline pixels to color
            assert np.array_equal(result, image)

    def test_render_segmentation_overlay_with_full_outline(self):
        """Test rendering when outline covers entire image."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        color = (0, 255, 255)  # Cyan color
        full_outline = np.ones((50, 50), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = full_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (50, 50, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            # All pixels should be the specified color
            assert np.all(result == color)

    def test_render_segmentation_overlay_render_outline_exception(self):
        """Test handling of exceptions from render_outline."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.side_effect = ValueError("Render outline error")

            # Act & Assert
            with pytest.raises(ValueError, match="Render outline error"):
                render_segmentation_overlay(contour, image, color)

    def test_render_segmentation_overlay_cv2_cvtcolor_exception(self):
        """Test handling of exceptions from cv2.cvtColor."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((50, 50, 1), dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones((50, 50), dtype=bool)

        with (
            patch(
                "mouse_tracking.utils.segmentation.render_outline"
            ) as mock_render_outline,
            patch("mouse_tracking.utils.segmentation.cv2.cvtColor") as mock_cvt_color,
        ):
            mock_render_outline.return_value = expected_outline
            mock_cvt_color.side_effect = Exception("OpenCV conversion error")

            # Act & Assert
            with pytest.raises(Exception, match="OpenCV conversion error"):
                render_segmentation_overlay(contour, image, color)

    @pytest.mark.parametrize(
        "image_shape",
        [
            (50, 50, 3),
            (100, 100, 3),
            (256, 256, 3),
            (480, 640, 3),
            (1080, 1920, 3),
        ],
    )
    def test_render_segmentation_overlay_different_image_sizes(self, image_shape):
        """Test rendering with different image sizes."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros(image_shape, dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones(image_shape[:2], dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == image_shape
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert call_args[1]["frame_size"] == image_shape[:2]

    def test_render_segmentation_overlay_with_zeros_contour(self):
        """Test rendering with contour containing zeros."""
        # Arrange
        contour = np.array([[[0, 0], [10, 10]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert np.array_equal(call_args[0][0], contour)

    def test_render_segmentation_overlay_with_negative_coordinates(self):
        """Test rendering with negative coordinates in contour."""
        # Arrange
        contour = np.array([[[-5, -10], [50, 60]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert np.array_equal(call_args[0][0], contour)

    @pytest.mark.parametrize(
        "input_dtype", [np.int32, np.int64, np.float32, np.float64]
    )
    def test_render_segmentation_overlay_with_different_contour_dtypes(
        self, input_dtype
    ):
        """Test rendering with different contour data types."""
        # Arrange
        contour = np.array(
            [[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]], dtype=input_dtype
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert call_args[0][0].dtype == input_dtype

    @pytest.mark.parametrize("image_dtype", [np.uint8, np.uint16, np.int32, np.float32])
    def test_render_segmentation_overlay_with_different_image_dtypes(self, image_dtype):
        """Test rendering with different image data types."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=image_dtype)
        color = (255, 0, 0)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == image_dtype  # Should preserve input image dtype
            mock_render_outline.assert_called_once()

    def test_render_segmentation_overlay_frame_size_extraction(self):
        """Test that frame_size is correctly extracted from image shape."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((123, 456, 3), dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones((123, 456), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (123, 456, 3)
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert call_args[1]["frame_size"] == (123, 456)

    def test_render_segmentation_overlay_color_type_annotation(self):
        """Test that color parameter accepts Tuple[int] type."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color: tuple[int, int, int] = (255, 128, 64)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            assert np.all(result[expected_outline] == color)

    def test_render_segmentation_overlay_outline_boolean_indexing(self):
        """Test that boolean indexing works correctly with outline."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        # Create a specific outline pattern
        expected_outline = np.zeros((100, 100), dtype=bool)
        expected_outline[25:75, 25:75] = True  # Square outline

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            # Check that only outline pixels have the color
            assert np.all(result[expected_outline] == color)
            # Check that non-outline pixels are unchanged (still zero)
            assert np.all(result[~expected_outline] == 0)

    def test_render_segmentation_overlay_mixed_padding_contour(self):
        """Test rendering with contour that has mixed padding and valid points."""
        # Arrange
        contour = np.array(
            [[[10, 20], [-1, -1], [30, 40]], [[-1, -1], [-1, -1], [-1, -1]]]
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (0, 255, 0)
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
            mock_render_outline.assert_called_once()
            call_args = mock_render_outline.call_args
            assert np.array_equal(call_args[0][0], contour)

    def test_render_segmentation_overlay_np_all_check_behavior(self):
        """Test that np.all(contour == -1) check works correctly."""
        # Arrange
        # Create contour with some -1 values but not all
        contour = np.array([[[10, 20], [-1, -1]], [[-1, -1], [-1, -1]]])
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        color = (255, 0, 0)
        expected_outline = np.ones((50, 50), dtype=bool)

        with patch(
            "mouse_tracking.utils.segmentation.render_outline"
        ) as mock_render_outline:
            mock_render_outline.return_value = expected_outline

            # Act
            result = render_segmentation_overlay(contour, image, color)

            # Assert
            # Should call render_outline because not ALL values are -1
            mock_render_outline.assert_called_once()
            assert result.shape == (50, 50, 3)
            assert np.all(result[expected_outline] == color)
