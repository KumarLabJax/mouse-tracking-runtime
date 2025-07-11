"""
Unit tests for the render_blob function from mouse_tracking.utils.segmentation.

This module tests the render_blob function which renders contour data as filled blobs
on a boolean mask. The function uses OpenCV's drawContours with cv2.FILLED thickness
to render solid regions and returns a boolean mask of the rendered blobs for
segmentation visualization and processing.

The tests cover:
- 2D and 3D contour matrix rendering
- Frame size customization and default values
- Custom padding value handling
- Boolean mask conversion and type safety
- OpenCV integration and parameter validation
- Exception handling and edge cases
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import render_blob


class TestRenderBlob:
    """Test suite for render_blob function."""

    def test_2d_contour_normal_usage(self):
        """Test rendering a 2D contour matrix."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
                [-1, -1],  # padding
            ]
        )
        frame_size = [100, 100]
        mock_contour_stack = [np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Simulate cv2.drawContours filling the mask
            def fill_mask(mask, contours, contour_idx, color, thickness):
                mask[20:60, 10:50] = 1  # Fill a rectangular area
                return mask

            mock_draw.side_effect = fill_mask

            # Act
            result = render_blob(contour, frame_size=frame_size)

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 100)
            assert result.dtype == bool

            # Verify get_contour_stack was called correctly
            mock_get_stack.assert_called_once_with(contour, default_val=-1)

            # Verify cv2.drawContours was called correctly
            mock_draw.assert_called_once()
            call_args = mock_draw.call_args[0]
            assert call_args[1] == mock_contour_stack  # contours
            assert call_args[2] == -1  # contour_idx (-1 means all)
            assert call_args[3] == 1  # color
            assert mock_draw.call_args[1]["thickness"] == -1  # cv2.FILLED

    def test_3d_contour_normal_usage(self):
        """Test rendering a 3D contour matrix."""
        # Arrange
        contour = np.array(
            [
                [  # First contour
                    [10, 20],
                    [30, 40],
                    [-1, -1],  # padding
                ],
                [  # Second contour
                    [50, 60],
                    [70, 80],
                    [90, 100],
                ],
            ]
        )
        frame_size = [200, 200]
        mock_contour_stack = [
            np.array([[10, 20], [30, 40]], dtype=np.int32),
            np.array([[50, 60], [70, 80], [90, 100]], dtype=np.int32),
        ]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Simulate cv2.drawContours filling the mask
            def fill_mask(mask, contours, contour_idx, color, thickness):
                mask[20:100, 10:90] = 1  # Fill a larger area
                return mask

            mock_draw.side_effect = fill_mask

            # Act
            result = render_blob(contour, frame_size=frame_size)

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (200, 200)
            assert result.dtype == bool

            # Verify get_contour_stack was called correctly
            mock_get_stack.assert_called_once_with(contour, default_val=-1)

            # Verify cv2.drawContours was called correctly
            mock_draw.assert_called_once()
            call_args = mock_draw.call_args[0]
            assert call_args[1] == mock_contour_stack

    def test_default_frame_size(self):
        """Test using default frame size."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours"),
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour)

            # Assert
            assert result.shape == (800, 800)  # Default frame size
            mock_get_stack.assert_called_once_with(contour, default_val=-1)

    def test_custom_default_value(self):
        """Test using custom default padding value."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
                [999, 999],  # custom padding
            ]
        )
        custom_default = 999
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours"),
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, default_val=custom_default)

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.dtype == bool

            # Verify get_contour_stack was called with custom default
            mock_get_stack.assert_called_once_with(contour, default_val=custom_default)

    def test_empty_contour_stack(self):
        """Test rendering when get_contour_stack returns empty list."""
        # Arrange
        contour = np.array(
            [
                [-1, -1],
                [-1, -1],
            ]
        )
        mock_contour_stack = []

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[50, 50])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (50, 50)
            assert result.dtype == bool
            assert not result.any()  # Should be all False

            # Verify cv2.drawContours was called with empty contour list
            mock_draw.assert_called_once()
            call_args = mock_draw.call_args[0]
            assert call_args[1] == []

    def test_rectangular_frame_size(self):
        """Test with rectangular (non-square) frame size."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        frame_size = [300, 200]
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours"),
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=frame_size)

            # Assert
            assert result.shape == (300, 200)
            mock_get_stack.assert_called_once_with(contour, default_val=-1)

    def test_single_point_contour(self):
        """Test rendering a contour with a single point."""
        # Arrange
        contour = np.array([[10, 20]])
        mock_contour_stack = [np.array([[10, 20]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[100, 100])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 100)
            assert result.dtype == bool

            # Verify cv2.drawContours was called with single point contour
            mock_draw.assert_called_once()
            call_args = mock_draw.call_args[0]
            assert len(call_args[1]) == 1
            np.testing.assert_array_equal(call_args[1][0], [[10, 20]])

    def test_multiple_contours_with_holes(self):
        """Test rendering multiple contours with potential holes."""
        # Arrange
        contour = np.array(
            [
                [  # Outer contour
                    [10, 10],
                    [90, 10],
                    [90, 90],
                    [10, 90],
                ],
                [  # Inner contour (hole)
                    [30, 30],
                    [70, 30],
                    [70, 70],
                    [30, 70],
                ],
            ]
        )
        mock_contour_stack = [
            np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32),
            np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32),
        ]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[100, 100])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 100)
            assert result.dtype == bool

            # Verify cv2.drawContours was called with all contours at once
            mock_draw.assert_called_once()
            call_args = mock_draw.call_args[0]
            assert call_args[2] == -1  # -1 means draw all contours
            assert len(call_args[1]) == 2  # Two contours

    def test_cv2_drawcontours_parameters(self):
        """Test that cv2.drawContours is called with correct parameters."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            render_blob(contour, frame_size=[100, 100])

            # Assert
            mock_draw.assert_called_once()
            args, kwargs = mock_draw.call_args

            # Check positional arguments
            assert args[0].shape == (100, 100)  # mask
            assert args[0].dtype == np.uint8
            assert args[1] == mock_contour_stack  # contours
            assert args[2] == -1  # contour_idx
            assert args[3] == 1  # color

            # Check keyword arguments
            assert "thickness" in kwargs
            assert kwargs["thickness"] == -1  # cv2.FILLED

    def test_mask_initialization(self):
        """Test that the mask is properly initialized."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        frame_size = [50, 60]
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Capture the mask that was passed to cv2.drawContours
            def capture_mask(mask, contours, contour_idx, color, thickness):
                # Check that initial mask is zeros
                assert mask.shape == (50, 60)
                assert mask.dtype == np.uint8
                assert not mask.any()  # Should be all zeros initially
                return mask

            mock_draw.side_effect = capture_mask

            # Act
            render_blob(contour, frame_size=frame_size)

            # Assert
            mock_draw.assert_called_once()

    def test_boolean_conversion(self):
        """Test that the result is properly converted to boolean."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Simulate cv2.drawContours setting values to 1
            def fill_mask(mask, contours, contour_idx, color, thickness):
                mask[20:40, 10:30] = 1
                return mask

            mock_draw.side_effect = fill_mask

            # Act
            result = render_blob(contour, frame_size=[100, 100])

            # Assert
            assert result.dtype == bool
            assert result[20:40, 10:30].all()  # Should be True where filled
            assert not result[0:20, 0:10].any()  # Should be False elsewhere

    def test_get_contour_stack_exception_handling(self):
        """Test behavior when get_contour_stack raises an exception."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )

        with patch(
            "mouse_tracking.utils.segmentation.get_contour_stack"
        ) as mock_get_stack:
            mock_get_stack.side_effect = ValueError("get_contour_stack failed")

            # Act & Assert
            with pytest.raises(ValueError, match="get_contour_stack failed"):
                render_blob(contour, frame_size=[100, 100])

    def test_cv2_drawcontours_exception_handling(self):
        """Test behavior when cv2.drawContours raises an exception."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack
            mock_draw.side_effect = Exception("cv2.drawContours failed")

            # Act & Assert
            with pytest.raises(Exception, match="cv2.drawContours failed"):
                render_blob(contour, frame_size=[100, 100])

    def test_frame_size_tuple_vs_list(self):
        """Test that frame_size works with both tuple and list."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act - Test with tuple
            result_tuple = render_blob(contour, frame_size=(100, 100))

            # Reset mock
            mock_get_stack.reset_mock()
            mock_draw.reset_mock()

            # Act - Test with list
            result_list = render_blob(contour, frame_size=[100, 100])

            # Assert
            assert result_tuple.shape == result_list.shape
            assert result_tuple.dtype == result_list.dtype

    @pytest.mark.parametrize(
        "frame_height,frame_width",
        [
            (50, 50),
            (100, 200),
            (300, 150),
            (800, 600),
            (1, 1),
        ],
    )
    def test_parametrized_frame_sizes(self, frame_height, frame_width):
        """Test various frame sizes."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours"),
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[frame_height, frame_width])

            # Assert
            assert result.shape == (frame_height, frame_width)
            assert result.dtype == bool

    def test_large_contour_matrix(self):
        """Test with a large contour matrix."""
        # Arrange
        n_contours = 5
        n_points = 100
        contour = np.random.randint(0, 800, size=(n_contours, n_points, 2))
        mock_contour_stack = [
            np.random.randint(0, 800, size=(n_points, 2), dtype=np.int32)
            for _ in range(n_contours)
        ]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[800, 800])

            # Assert
            assert result.shape == (800, 800)
            assert result.dtype == bool
            mock_get_stack.assert_called_once_with(contour, default_val=-1)
            mock_draw.assert_called_once()

    def test_zero_frame_size_edge_case(self):
        """Test with zero frame size (edge case)."""
        # Arrange
        contour = np.array(
            [
                [10, 20],
                [30, 40],
            ]
        )
        mock_contour_stack = [np.array([[10, 20], [30, 40]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[0, 0])

            # Assert
            assert result.shape == (0, 0)
            assert result.dtype == bool
            mock_draw.assert_called_once()

    def test_contour_coordinates_outside_frame(self):
        """Test rendering contour with coordinates outside the frame."""
        # Arrange
        contour = np.array(
            [
                [1000, 2000],  # Outside frame
                [3000, 4000],  # Outside frame
            ]
        )
        mock_contour_stack = [np.array([[1000, 2000], [3000, 4000]], dtype=np.int32)]

        with (
            patch(
                "mouse_tracking.utils.segmentation.get_contour_stack"
            ) as mock_get_stack,
            patch("cv2.drawContours") as mock_draw,
        ):
            mock_get_stack.return_value = mock_contour_stack

            # Act
            result = render_blob(contour, frame_size=[100, 100])

            # Assert
            assert result.shape == (100, 100)
            assert result.dtype == bool
            # cv2.drawContours should handle coordinates outside frame gracefully
            mock_draw.assert_called_once()
            call_args = mock_draw.call_args[0]
            np.testing.assert_array_equal(call_args[1][0], [[1000, 2000], [3000, 4000]])
