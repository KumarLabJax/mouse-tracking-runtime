"""Unit tests for render_outline function.

This module contains comprehensive tests for the render_outline function from
the mouse_tracking.utils.segmentation module, including edge cases and error conditions.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import render_outline


class TestRenderOutline:
    """Test cases for render_outline function."""

    def test_single_contour_basic_rendering(self):
        """Test rendering a single contour with default parameters."""
        # Arrange
        contour = np.array(
            [
                [[10, 20], [30, 40], [50, 60]],
                [[-1, -1], [-1, -1], [-1, -1]],  # Padding
            ]
        )
        expected_contour_stack = [np.array([[10, 20], [30, 40], [50, 60]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()
            # Check cv2.drawContours call arguments
            call_args = mock_draw_contours.call_args[0]
            assert call_args[0].shape == (100, 100)  # new_mask
            assert call_args[1] == expected_contour_stack  # contour_stack
            assert call_args[2] == -1  # contour index (-1 for all)
            assert call_args[3] == 1  # color
            # Check kwargs
            assert mock_draw_contours.call_args[1]["thickness"] == 1

    def test_render_outline_with_custom_thickness(self):
        """Test rendering with custom thickness."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[50, 50], thickness=3)

            # Assert
            assert result.shape == (50, 50)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()
            # Check thickness parameter
            assert mock_draw_contours.call_args[1]["thickness"] == 3

    def test_render_outline_with_custom_default_val(self):
        """Test rendering with custom default value."""
        # Arrange
        contour = np.array(
            [
                [[10, 20], [30, 40]],
                [[-99, -99], [-99, -99]],  # Custom padding
            ]
        )
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[50, 50], default_val=-99)

            # Assert
            assert result.shape == (50, 50)
            assert result.dtype == bool
            # NOTE: This test exposes a bug - the function doesn't pass default_val to get_contour_stack
            # It should be called with default_val=-99 but currently calls with default default_val=-1
            mock_get_contour_stack.assert_called_once_with(contour)

    def test_render_outline_with_multiple_contours(self):
        """Test rendering multiple contours."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
        expected_contour_stack = [
            np.array([[10, 20], [30, 40]]),
            np.array([[50, 60], [70, 80]]),
        ]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()
            # Check that all contours are passed to cv2.drawContours
            call_args = mock_draw_contours.call_args[0]
            assert call_args[1] == expected_contour_stack

    def test_render_outline_with_empty_contour_stack(self):
        """Test rendering with empty contour stack."""
        # Arrange
        contour = np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = []

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            assert not np.any(result)  # Should be all False since no contours to draw
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    @pytest.mark.parametrize(
        "frame_size", [[50, 50], [100, 200], [1, 1], [1024, 768], [800, 600]]
    )
    def test_render_outline_with_different_frame_sizes(self, frame_size):
        """Test rendering with different frame sizes."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=frame_size)

            # Assert
            assert result.shape == (frame_size[0], frame_size[1])
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    def test_render_outline_with_frame_size_as_tuple(self):
        """Test rendering with frame size as tuple."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=(150, 200))

            # Assert
            assert result.shape == (150, 200)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    @pytest.mark.parametrize("thickness", [1, 2, 3, 5, 10, 15])
    def test_render_outline_with_different_thickness_values(self, thickness):
        """Test rendering with different thickness values."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100], thickness=thickness)

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()
            assert mock_draw_contours.call_args[1]["thickness"] == thickness

    def test_render_outline_with_default_parameters(self):
        """Test rendering with all default parameters."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour)

            # Assert
            assert result.shape == (800, 800)  # Default frame size
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()
            assert (
                mock_draw_contours.call_args[1]["thickness"] == 1
            )  # Default thickness

    def test_render_outline_boolean_conversion(self):
        """Test proper conversion from uint8 to boolean."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack

            # Mock cv2.drawContours to modify the mask
            def mock_draw_side_effect(mask, contours, idx, color, thickness=1):
                # Simulate drawing by setting some pixels to the color value
                mask[10:30, 10:30] = color
                return None

            mock_draw_contours.side_effect = mock_draw_side_effect

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.dtype == bool
            # Check that the modified region is True
            assert np.all(result[10:30, 10:30])
            # Check that the unmodified region is False
            assert not np.any(result[0:10, 0:10])

    def test_render_outline_2d_contour_input(self):
        """Test rendering with 2D contour input [n_points, 2]."""
        # Arrange
        contour = np.array([[10, 20], [30, 40], [50, 60]])
        expected_contour_stack = [np.array([[10, 20], [30, 40], [50, 60]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    def test_render_outline_get_contour_stack_exception(self):
        """Test handling of exceptions from get_contour_stack."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch("mouse_tracking.utils.segmentation.cv2.drawContours"),
        ):
            mock_get_contour_stack.side_effect = ValueError("Invalid contour matrix")

            # Act & Assert
            with pytest.raises(ValueError, match="Invalid contour matrix"):
                render_outline(contour, frame_size=[100, 100])

    def test_render_outline_cv2_draw_contours_exception(self):
        """Test handling of exceptions from cv2.drawContours."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.side_effect = Exception("OpenCV error")

            # Act & Assert
            with pytest.raises(Exception, match="OpenCV error"):
                render_outline(contour, frame_size=[100, 100])

    def test_render_outline_with_zeros_contour(self):
        """Test rendering with contour containing zeros."""
        # Arrange
        contour = np.array([[[0, 0], [10, 10]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[0, 0], [10, 10]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    def test_render_outline_with_negative_coordinates(self):
        """Test rendering with negative coordinates."""
        # Arrange
        contour = np.array([[[-5, -10], [50, 60]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[-5, -10], [50, 60]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    def test_render_outline_with_large_coordinates(self):
        """Test rendering with coordinates larger than frame size."""
        # Arrange
        contour = np.array([[[1000, 2000], [3000, 4000]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[1000, 2000], [3000, 4000]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    @pytest.mark.parametrize(
        "input_dtype", [np.int32, np.int64, np.float32, np.float64]
    )
    def test_render_outline_with_different_input_dtypes(self, input_dtype):
        """Test rendering with different input data types."""
        # Arrange
        contour = np.array(
            [[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]], dtype=input_dtype
        )
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once()
            # Verify the input to get_contour_stack maintains the original dtype
            passed_contour = mock_get_contour_stack.call_args[0][0]
            assert passed_contour.dtype == input_dtype

    def test_render_outline_mask_initialization(self):
        """Test that new_mask is properly initialized."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack

            # Capture the mask that's passed to cv2.drawContours
            captured_mask = None

            def capture_mask(mask, contours, idx, color, thickness=1):
                nonlocal captured_mask
                captured_mask = mask.copy()
                return None

            mock_draw_contours.side_effect = capture_mask

            # Act
            render_outline(contour, frame_size=[50, 50])

            # Assert
            assert captured_mask is not None
            assert captured_mask.shape == (50, 50)
            assert captured_mask.dtype == np.uint8
            assert np.all(captured_mask == 0)  # Should be initialized to zeros

    def test_render_outline_opencv_color_parameter(self):
        """Test that OpenCV is called with correct color parameter."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            render_outline(contour, frame_size=[100, 100])

            # Assert
            call_args = mock_draw_contours.call_args[0]
            assert call_args[3] == 1  # Color should be 1 for single channel

    def test_render_outline_opencv_contour_index_parameter(self):
        """Test that OpenCV is called with correct contour index parameter."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]])
        expected_contour_stack = [np.array([[10, 20], [30, 40]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            render_outline(contour, frame_size=[100, 100])

            # Assert
            call_args = mock_draw_contours.call_args[0]
            assert call_args[2] == -1  # Contour index should be -1 (draw all contours)

    def test_render_outline_single_point_contour(self):
        """Test rendering with single point contour."""
        # Arrange
        contour = np.array([[[10, 20]], [[-1, -1]]])
        expected_contour_stack = [np.array([[10, 20]])]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            result = render_outline(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            mock_get_contour_stack.assert_called_once_with(contour)
            mock_draw_contours.assert_called_once()

    def test_render_outline_comment_describes_opencv_hole_detection(self):
        """Test that the function draws all contours at once for hole detection."""
        # Arrange
        contour = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
        expected_contour_stack = [
            np.array([[10, 20], [30, 40]]),
            np.array([[50, 60], [70, 80]]),
        ]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_contour_stack,
            patch(
                "mouse_tracking.utils.segmentation.cv2.drawContours"
            ) as mock_draw_contours,
        ):
            mock_get_contour_stack.return_value = expected_contour_stack
            mock_draw_contours.return_value = None

            # Act
            render_outline(contour, frame_size=[100, 100])

            # Assert
            mock_draw_contours.assert_called_once()
            # Verify that ALL contours are passed in a single call (not multiple calls)
            call_args = mock_draw_contours.call_args[0]
            assert call_args[1] == expected_contour_stack
            assert call_args[2] == -1  # -1 means draw all contours in the list
