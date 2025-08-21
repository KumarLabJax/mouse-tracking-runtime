"""Unit tests for get_frame_outlines function.

This module contains comprehensive tests for the get_frame_outlines function from
the mouse_tracking.utils.segmentation module, including edge cases and error conditions.
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import get_frame_outlines


class TestGetFrameOutlines:
    """Test cases for get_frame_outlines function."""

    def test_single_animal_basic_contour(self):
        """Test processing single animal with basic contour."""
        # Arrange
        contour_mat = np.array(
            [
                [
                    [[10, 20], [30, 40], [50, 60]],
                    [[-1, -1], [-1, -1], [-1, -1]],  # Padding
                ]
            ]
        )
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[100, 100])

            # Assert
            assert result.shape == (1, 100, 100)
            assert result.dtype == bool
            assert np.array_equal(result[0], expected_outline)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == [100, 100]
            assert call_args[1]["thickness"] == 1

    def test_multiple_animals_with_different_outlines(self):
        """Test processing multiple animals with different outline patterns."""
        # Arrange
        # Create arrays with consistent shapes
        animal1_contour = np.array(
            [[[10, 20], [30, 40], [50, 60]], [[-1, -1], [-1, -1], [-1, -1]]]
        )
        animal2_contour = np.array(
            [[[100, 200], [300, 400], [-1, -1]], [[-1, -1], [-1, -1], [-1, -1]]]
        )
        contour_mat = np.array([animal1_contour, animal2_contour])

        outline1 = np.zeros((800, 800), dtype=np.uint8)
        outline1[10:20, 10:20] = 1
        outline2 = np.zeros((800, 800), dtype=np.uint8)
        outline2[30:40, 30:40] = 1

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.side_effect = [outline1, outline2]

            # Act
            result = get_frame_outlines(contour_mat)

            # Assert
            assert result.shape == (2, 800, 800)
            assert result.dtype == bool
            assert mock_render.call_count == 2
            # Manually check each call
            call_args_list = mock_render.call_args_list
            # First call
            assert np.array_equal(call_args_list[0][0][0], contour_mat[0])
            assert call_args_list[0][1]["frame_size"] == [800, 800]
            assert call_args_list[0][1]["thickness"] == 1
            # Second call
            assert np.array_equal(call_args_list[1][0][0], contour_mat[1])
            assert call_args_list[1][1]["frame_size"] == [800, 800]
            assert call_args_list[1][1]["thickness"] == 1

    def test_empty_contour_matrix(self):
        """Test processing empty contour matrix."""
        # Arrange
        contour_mat = np.empty((0, 0, 0, 2))

        # Act
        result = get_frame_outlines(contour_mat)

        # Assert
        assert result.shape == (0, 800, 800)
        assert result.dtype == float  # Default numpy array dtype

    def test_empty_contour_matrix_custom_frame_size(self):
        """Test processing empty contour matrix with custom frame size."""
        # Arrange
        contour_mat = np.empty((0, 0, 0, 2))

        # Act
        result = get_frame_outlines(contour_mat, frame_size=[200, 300])

        # Assert
        assert result.shape == (0, 200, 300)

    @pytest.mark.parametrize(
        "frame_size", [[100, 100], [200, 150], [512, 384], [1024, 768]]
    )
    def test_different_frame_sizes(self, frame_size):
        """Test processing with different frame sizes."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        expected_outline = np.ones(frame_size, dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=frame_size)

            # Assert
            assert result.shape == (1, frame_size[0], frame_size[1])
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == frame_size
            assert call_args[1]["thickness"] == 1

    @pytest.mark.parametrize("thickness", [1, 2, 3, 5, 10])
    def test_different_thickness_values(self, thickness):
        """Test processing with different thickness values."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(
                contour_mat, frame_size=[100, 100], thickness=thickness
            )

            # Assert
            assert result.shape == (1, 100, 100)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == [100, 100]
            assert call_args[1]["thickness"] == thickness

    def test_frame_size_as_tuple(self):
        """Test processing with frame size as tuple instead of list."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        expected_outline = np.ones((150, 200), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=(150, 200))

            # Assert
            assert result.shape == (1, 150, 200)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == (150, 200)
            assert call_args[1]["thickness"] == 1

    def test_boolean_conversion_from_uint8(self):
        """Test proper conversion from uint8 to boolean."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        # Create uint8 array with values 0, 1, 255
        outline_uint8 = np.array(
            [[0, 1, 255], [0, 1, 255], [0, 1, 255]], dtype=np.uint8
        )
        expected_bool = outline_uint8.astype(bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = outline_uint8

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[3, 3])

            # Assert
            assert result.dtype == bool
            assert np.array_equal(result[0], expected_bool)

    def test_boolean_conversion_from_float(self):
        """Test proper conversion from float to boolean."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        # Create float array with values 0.0, 0.5, 1.0
        outline_float = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32)
        expected_bool = outline_float.astype(bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = outline_float

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[2, 3])

            # Assert
            assert result.dtype == bool
            assert np.array_equal(result[0], expected_bool)

    def test_large_number_of_animals(self):
        """Test processing with many animals."""
        # Arrange
        n_animals = 10
        contour_mat = np.array(
            [
                [[[i * 10, i * 20], [i * 30, i * 40]], [[-1, -1], [-1, -1]]]
                for i in range(n_animals)
            ]
        )
        expected_outline = np.ones((50, 50), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[50, 50])

            # Assert
            assert result.shape == (n_animals, 50, 50)
            assert result.dtype == bool
            assert mock_render.call_count == n_animals

    def test_render_outline_exception_handling(self):
        """Test handling of exceptions from render_outline."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.side_effect = ValueError("Mock error")

            # Act & Assert
            with pytest.raises(ValueError, match="Mock error"):
                get_frame_outlines(contour_mat)

    def test_mixed_valid_and_invalid_contours(self):
        """Test processing when some animals have valid contours and others don't."""
        # Arrange
        contour_mat = np.array(
            [
                [[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]],
                [
                    [[-1, -1], [-1, -1]],  # All padding
                    [[-1, -1], [-1, -1]],
                ],
            ]
        )

        outline1 = np.ones((50, 50), dtype=np.uint8)
        outline2 = np.zeros((50, 50), dtype=np.uint8)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.side_effect = [outline1, outline2]

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[50, 50])

            # Assert
            assert result.shape == (2, 50, 50)
            assert result.dtype == bool
            assert np.array_equal(result[0], outline1.astype(bool))
            assert np.array_equal(result[1], outline2.astype(bool))

    def test_default_parameter_values(self):
        """Test that default parameter values are used correctly."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        expected_outline = np.ones((800, 800), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat)

            # Assert
            assert result.shape == (1, 800, 800)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == [800, 800]
            assert call_args[1]["thickness"] == 1

    def test_numpy_arange_usage(self):
        """Test that numpy.arange is used correctly for animal indexing."""
        # Arrange
        contour_mat = np.array(
            [
                [[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]],
                [[[100, 200], [300, 400]], [[-1, -1], [-1, -1]]],
            ]
        )
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[100, 100])

            # Assert
            assert result.shape == (2, 100, 100)
            # Verify calls were made in correct order
            call_args_list = mock_render.call_args_list
            assert len(call_args_list) == 2
            # First call
            assert np.array_equal(call_args_list[0][0][0], contour_mat[0])
            assert call_args_list[0][1]["frame_size"] == [100, 100]
            assert call_args_list[0][1]["thickness"] == 1
            # Second call
            assert np.array_equal(call_args_list[1][0][0], contour_mat[1])
            assert call_args_list[1][1]["frame_size"] == [100, 100]
            assert call_args_list[1][1]["thickness"] == 1

    def test_single_pixel_frame_size(self):
        """Test processing with minimal frame size."""
        # Arrange
        contour_mat = np.array([[[[0, 0]], [[-1, -1]]]])
        expected_outline = np.array([[True]], dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[1, 1])

            # Assert
            assert result.shape == (1, 1, 1)
            assert result.dtype == bool
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == [1, 1]
            assert call_args[1]["thickness"] == 1

    def test_asymmetric_frame_size(self):
        """Test processing with asymmetric frame dimensions."""
        # Arrange
        contour_mat = np.array([[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]])
        expected_outline = np.ones((100, 200), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[100, 200])

            # Assert
            assert result.shape == (1, 100, 200)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == [100, 200]
            assert call_args[1]["thickness"] == 1

    @pytest.mark.parametrize("input_dtype", [np.int32, np.float32, np.float64])
    def test_different_input_dtypes(self, input_dtype):
        """Test processing with different input data types."""
        # Arrange
        contour_mat = np.array(
            [[[[10, 20], [30, 40]], [[-1, -1], [-1, -1]]]], dtype=input_dtype
        )
        expected_outline = np.ones((100, 100), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[100, 100])

            # Assert
            assert result.shape == (1, 100, 100)
            assert result.dtype == bool
            # Verify the input to render_outline maintains the original dtype
            passed_contour = mock_render.call_args[0][0]
            assert passed_contour.dtype == input_dtype

    def test_contour_matrix_with_zero_points(self):
        """Test processing contour matrix with zero points dimension."""
        # Arrange
        contour_mat = np.empty((1, 0, 0, 2))
        expected_outline = np.zeros((100, 100), dtype=bool)

        with patch("mouse_tracking.utils.segmentation.render_outline") as mock_render:
            mock_render.return_value = expected_outline.astype(np.uint8)

            # Act
            result = get_frame_outlines(contour_mat, frame_size=[100, 100])

            # Assert
            assert result.shape == (1, 100, 100)
            assert result.dtype == bool
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert np.array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1]["frame_size"] == [100, 100]
            assert call_args[1]["thickness"] == 1
