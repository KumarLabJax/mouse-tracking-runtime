"""
Unit tests for the get_frame_masks function from mouse_tracking.utils.segmentation.

This module tests the get_frame_masks function which processes contour matrices
to generate boolean masks for each animal in a frame. The function renders
contours as filled regions using render_blob and returns a stack of masks
for batch processing applications.

The tests cover:
- Single and multiple animal mask generation
- Different frame sizes and custom configurations
- Boolean conversion from various numeric types
- Edge cases like empty contour matrices
- Integration with render_blob function
- Error handling and exception scenarios
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import get_frame_masks


class TestGetFrameMasks:
    """Test suite for get_frame_masks function."""

    def test_multiple_animals_normal_usage(self):
        """Test processing contour matrix with multiple animals."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                        [50, 60],
                    ],
                    [  # Contour 2 (padding)
                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                    ],
                ],
                [  # Animal 2
                    [  # Contour 1
                        [70, 80],
                        [90, 100],
                        [110, 120],
                    ],
                    [  # Contour 2 (padding)
                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.side_effect = [
                np.array([[True, False], [False, True]]),  # Animal 1 mask
                np.array([[False, True], [True, False]]),  # Animal 2 mask
            ]

            # Act
            result = get_frame_masks(contour_mat, frame_size=[2, 2])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 2, 2)  # (n_animals, height, width)
            assert result.dtype == bool

            # Check that render_blob was called correctly
            assert mock_render.call_count == 2
            call_args = mock_render.call_args_list
            np.testing.assert_array_equal(call_args[0][0][0], contour_mat[0])
            np.testing.assert_array_equal(call_args[1][0][0], contour_mat[1])
            assert call_args[0][1] == {"frame_size": [2, 2]}
            assert call_args[1][1] == {"frame_size": [2, 2]}

    def test_single_animal(self):
        """Test processing contour matrix with single animal."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                        [50, 60],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.array([[True, False], [False, True]])

            # Act
            result = get_frame_masks(contour_mat, frame_size=[2, 2])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 2, 2)  # (n_animals, height, width)
            assert result.dtype == bool

            # Check that render_blob was called once
            mock_render.assert_called_once()
            np.testing.assert_array_equal(mock_render.call_args[0][0], contour_mat[0])

    def test_empty_contour_matrix(self):
        """Test processing empty contour matrix."""
        # Arrange
        contour_mat = np.array([]).reshape(0, 0, 0, 2)

        # Act
        result = get_frame_masks(contour_mat, frame_size=[800, 600])

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 800, 600)
        assert result.dtype == float  # np.zeros creates float by default

    def test_default_frame_size(self):
        """Test using default frame size."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.zeros((800, 800), dtype=bool)

            # Act
            result = get_frame_masks(contour_mat)

            # Assert
            assert result.shape == (1, 800, 800)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            np.testing.assert_array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1] == {"frame_size": [800, 800]}

    def test_custom_frame_size(self):
        """Test using custom frame size."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
            ]
        )
        frame_size = [640, 480]

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.zeros((640, 480), dtype=bool)

            # Act
            result = get_frame_masks(contour_mat, frame_size=frame_size)

            # Assert
            assert result.shape == (1, 640, 480)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            np.testing.assert_array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1] == {"frame_size": frame_size}

    def test_render_blob_returns_non_boolean(self):
        """Test that non-boolean output from render_blob is converted to boolean."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            # Return non-boolean array (integers)
            mock_render.return_value = np.array([[1, 0], [0, 255]], dtype=np.uint8)

            # Act
            result = get_frame_masks(contour_mat, frame_size=[2, 2])

            # Assert
            assert result.dtype == bool
            expected = np.array([[[True, False], [False, True]]])
            np.testing.assert_array_equal(result, expected)

    def test_multiple_animals_different_mask_patterns(self):
        """Test multiple animals with different mask patterns."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
                [  # Animal 2
                    [  # Contour 1
                        [50, 60],
                        [70, 80],
                    ],
                ],
                [  # Animal 3
                    [  # Contour 1
                        [90, 100],
                        [110, 120],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.side_effect = [
                np.array([[True, True], [False, False]]),  # Animal 1
                np.array([[False, False], [True, True]]),  # Animal 2
                np.array([[True, False], [False, True]]),  # Animal 3
            ]

            # Act
            result = get_frame_masks(contour_mat, frame_size=[2, 2])

            # Assert
            assert result.shape == (3, 2, 2)
            assert result.dtype == bool

            # Check individual animal masks
            expected_animal1 = np.array([[True, True], [False, False]])
            expected_animal2 = np.array([[False, False], [True, True]])
            expected_animal3 = np.array([[True, False], [False, True]])

            np.testing.assert_array_equal(result[0], expected_animal1)
            np.testing.assert_array_equal(result[1], expected_animal2)
            np.testing.assert_array_equal(result[2], expected_animal3)

    def test_large_contour_matrix(self):
        """Test processing a large contour matrix."""
        # Arrange
        n_animals = 5
        n_contours = 3
        n_points = 10
        contour_mat = np.random.randint(
            0, 100, size=(n_animals, n_contours, n_points, 2)
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.zeros((100, 100), dtype=bool)

            # Act
            result = get_frame_masks(contour_mat, frame_size=[100, 100])

            # Assert
            assert result.shape == (n_animals, 100, 100)
            assert result.dtype == bool
            assert mock_render.call_count == n_animals

    def test_render_blob_exception_handling(self):
        """Test behavior when render_blob raises an exception."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.side_effect = ValueError("render_blob failed")

            # Act & Assert
            with pytest.raises(ValueError, match="render_blob failed"):
                get_frame_masks(contour_mat, frame_size=[2, 2])

    def test_zero_animals(self):
        """Test processing contour matrix with zero animals."""
        # Arrange
        contour_mat = np.array([]).reshape(0, 5, 10, 2)

        # Act
        result = get_frame_masks(contour_mat, frame_size=[100, 100])

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 100, 100)
        assert result.dtype == float  # np.zeros creates float by default

    def test_rectangular_frame_size(self):
        """Test with rectangular (non-square) frame size."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.zeros((300, 200), dtype=bool)

            # Act
            result = get_frame_masks(contour_mat, frame_size=[300, 200])

            # Assert
            assert result.shape == (1, 300, 200)
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            np.testing.assert_array_equal(call_args[0][0], contour_mat[0])
            assert call_args[1] == {"frame_size": [300, 200]}

    def test_frame_size_tuple_vs_list(self):
        """Test that frame_size works with both tuple and list."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.zeros((100, 100), dtype=bool)

            # Act - Test with tuple
            result_tuple = get_frame_masks(contour_mat, frame_size=(100, 100))

            # Reset mock
            mock_render.reset_mock()

            # Act - Test with list
            result_list = get_frame_masks(contour_mat, frame_size=[100, 100])

            # Assert
            assert result_tuple.shape == result_list.shape
            assert mock_render.call_count == 1

    def test_maintains_contour_order(self):
        """Test that the function maintains the order of animals in the contour matrix."""
        # Arrange
        contour_mat = np.array(
            [
                [  # Animal 1
                    [  # Contour 1
                        [10, 20],
                        [30, 40],
                    ],
                ],
                [  # Animal 2
                    [  # Contour 1
                        [50, 60],
                        [70, 80],
                    ],
                ],
            ]
        )

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.side_effect = [
                np.array([[True, False]]),  # Animal 1 - distinct pattern
                np.array([[False, True]]),  # Animal 2 - distinct pattern
            ]

            # Act
            result = get_frame_masks(contour_mat, frame_size=[1, 2])

            # Assert
            assert result.shape == (2, 1, 2)
            np.testing.assert_array_equal(result[0], [[True, False]])
            np.testing.assert_array_equal(result[1], [[False, True]])

    @pytest.mark.parametrize(
        "n_animals,frame_height,frame_width",
        [
            (1, 50, 50),
            (2, 100, 100),
            (3, 200, 150),
            (5, 800, 600),
        ],
    )
    def test_parametrized_dimensions(self, n_animals, frame_height, frame_width):
        """Test various combinations of number of animals and frame dimensions."""
        # Arrange
        contour_mat = np.ones((n_animals, 2, 3, 2), dtype=np.int32)

        with patch("mouse_tracking.utils.segmentation.render_blob") as mock_render:
            mock_render.return_value = np.zeros((frame_height, frame_width), dtype=bool)

            # Act
            result = get_frame_masks(
                contour_mat, frame_size=[frame_height, frame_width]
            )

            # Assert
            assert result.shape == (n_animals, frame_height, frame_width)
            assert result.dtype == bool
            assert mock_render.call_count == n_animals

    def test_empty_frame_stack_return_type(self):
        """Test that empty frame stack returns the correct type and shape."""
        # Arrange
        contour_mat = np.array([]).reshape(0, 2, 3, 2)
        frame_size = [400, 300]

        # Act
        result = get_frame_masks(contour_mat, frame_size=frame_size)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 400, 300)
        # Note: np.zeros returns float64 by default, but this matches the function's behavior
        assert result.dtype in [np.float64, float]
