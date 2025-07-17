"""
Unit tests for the get_contour_stack function from mouse_tracking.utils.segmentation.

This module tests the get_contour_stack function which converts padded contour matrices
into lists of OpenCV-compatible contour arrays by removing padding and extracting
valid contour data. The function handles both 2D and 3D contour matrices and ensures
proper formatting for subsequent OpenCV operations.

The tests cover:
- 2D contour matrix processing (single contour)
- 3D contour matrix processing (multiple contours)
- Padding removal with default and custom padding values
- Edge cases like empty arrays and all-padding matrices
- Error handling for invalid input shapes
- Integration with get_trimmed_contour function
"""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import get_contour_stack


class TestGetContourStack:
    """Test suite for get_contour_stack function."""

    def test_2d_single_contour(self):
        """Test processing a 2D contour matrix (single contour)."""
        # Arrange
        contour_mat = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
                [-1, -1],  # padding
            ]
        )
        expected_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_2d_single_contour_no_padding(self):
        """Test processing a 2D contour matrix without padding."""
        # Arrange
        contour_mat = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ]
        )
        expected_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_2d_all_padding(self):
        """Test processing a 2D contour matrix that is all padding."""
        # Arrange
        contour_mat = np.array(
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ]
        )
        expected_contour = np.array([], dtype=np.int32).reshape(0, 2)

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_3d_multiple_contours(self):
        """Test processing a 3D contour matrix with multiple contours."""
        # Arrange
        contour_mat = np.array(
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
                [  # Third contour (all padding - should break)
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ],
            ]
        )
        expected_contours = [
            np.array([[10, 20], [30, 40]], dtype=np.int32),
            np.array([[50, 60], [70, 80], [90, 100]], dtype=np.int32),
        ]

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 2
        for i, expected in enumerate(expected_contours):
            np.testing.assert_array_equal(result[i], expected)

    def test_3d_single_contour_in_stack(self):
        """Test processing a 3D contour matrix with only one valid contour."""
        # Arrange
        contour_mat = np.array(
            [
                [  # First contour
                    [10, 20],
                    [30, 40],
                    [50, 60],
                ],
                [  # Second contour (all padding - should break)
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ],
            ]
        )
        expected_contour = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_3d_empty_stack(self):
        """Test processing a 3D contour matrix where the first contour is all padding."""
        # Arrange
        contour_mat = np.array(
            [
                [  # First contour (all padding - should break immediately)
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ],
                [  # Second contour (should not be processed)
                    [50, 60],
                    [70, 80],
                    [90, 100],
                ],
            ]
        )

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0

    def test_none_input(self):
        """Test processing None input."""
        # Act
        result = get_contour_stack(None)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0

    def test_custom_default_value(self):
        """Test processing with a custom default padding value."""
        # Arrange
        contour_mat = np.array(
            [
                [10, 20],
                [30, 40],
                [999, 999],  # custom padding
            ]
        )
        expected_contour = np.array(
            [
                [10, 20],
                [30, 40],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_contour_stack(contour_mat, default_val=999)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_custom_default_value_3d(self):
        """Test processing 3D matrix with custom default padding value."""
        # Arrange
        contour_mat = np.array(
            [
                [  # First contour
                    [10, 20],
                    [30, 40],
                    [999, 999],  # custom padding
                ],
                [  # Second contour (all custom padding - should break)
                    [999, 999],
                    [999, 999],
                    [999, 999],
                ],
            ]
        )
        expected_contour = np.array(
            [
                [10, 20],
                [30, 40],
            ],
            dtype=np.int32,
        )

        # Act
        result = get_contour_stack(contour_mat, default_val=999)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_empty_2d_array(self):
        """Test processing an empty 2D array."""
        # Arrange
        contour_mat = np.array([]).reshape(0, 2)
        expected_contour = np.array([], dtype=np.int32).reshape(0, 2)

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_empty_3d_array(self):
        """Test processing an empty 3D array."""
        # Arrange
        contour_mat = np.array([]).reshape(0, 0, 2)

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 0

    def test_single_point_2d_contour(self):
        """Test processing a 2D contour with a single point."""
        # Arrange
        contour_mat = np.array([[10, 20]])
        expected_contour = np.array([[10, 20]], dtype=np.int32)

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected_contour)

    def test_invalid_1d_array_raises_error(self):
        """Test that 1D array raises ValueError."""
        # Arrange
        contour_mat = np.array([10, 20, 30])

        # Act & Assert
        with pytest.raises(ValueError, match="Contour matrix invalid"):
            get_contour_stack(contour_mat)

    def test_invalid_4d_array_raises_error(self):
        """Test that 4D array raises ValueError."""
        # Arrange
        contour_mat = np.array([[[[10, 20]]]])

        # Act & Assert
        with pytest.raises(ValueError, match="Contour matrix invalid"):
            get_contour_stack(contour_mat)

    def test_invalid_scalar_raises_error(self):
        """Test that scalar input raises ValueError."""
        # Arrange
        contour_mat = 42

        # Act & Assert
        with pytest.raises(ValueError, match="Contour matrix invalid"):
            get_contour_stack(contour_mat)

    def test_calls_get_trimmed_contour_correctly(self):
        """Test that get_trimmed_contour is called with correct parameters."""
        # Arrange
        contour_mat = np.array(
            [
                [10, 20],
                [30, 40],
                [-1, -1],
            ]
        )

        with patch(
            "mouse_tracking.utils.segmentation.get_trimmed_contour"
        ) as mock_get_trimmed:
            mock_get_trimmed.return_value = np.array(
                [[10, 20], [30, 40]], dtype=np.int32
            )

            # Act
            result = get_contour_stack(contour_mat, default_val=999)

            # Assert
            mock_get_trimmed.assert_called_once_with(contour_mat, 999)
            assert isinstance(result, list)
            assert len(result) == 1

    def test_calls_get_trimmed_contour_for_3d_array(self):
        """Test that get_trimmed_contour is called for each contour in 3D array."""
        # Arrange
        contour_mat = np.array(
            [
                [  # First contour
                    [10, 20],
                    [30, 40],
                    [-1, -1],
                ],
                [  # Second contour
                    [50, 60],
                    [70, 80],
                    [-1, -1],
                ],
            ]
        )

        with patch(
            "mouse_tracking.utils.segmentation.get_trimmed_contour"
        ) as mock_get_trimmed:
            mock_get_trimmed.side_effect = [
                np.array([[10, 20], [30, 40]], dtype=np.int32),
                np.array([[50, 60], [70, 80]], dtype=np.int32),
            ]

            # Act
            result = get_contour_stack(contour_mat, default_val=999)

            # Assert
            assert isinstance(result, list)
            assert len(result) == 2
            assert mock_get_trimmed.call_count == 2
            expected_calls = [
                ((contour_mat[0], 999), {}),
                ((contour_mat[1], 999), {}),
            ]
            actual_calls = [
                (call.args, call.kwargs) for call in mock_get_trimmed.call_args_list
            ]

            # Check that calls were made with correct arguments
            assert len(actual_calls) == 2
            for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
                actual_args, actual_kwargs = actual_calls[i]
                np.testing.assert_array_equal(actual_args[0], expected_args[0])
                assert actual_args[1] == expected_args[1]
                assert actual_kwargs == expected_kwargs

    @pytest.mark.parametrize(
        "input_shape,expected_length",
        [
            ((5, 2), 1),  # 2D array -> single contour
            ((3, 5, 2), 3),  # 3D array -> multiple contours (max possible)
            ((0, 2), 1),  # Empty 2D array -> single empty contour
            ((0, 0, 2), 0),  # Empty 3D array -> no contours
        ],
    )
    def test_parametrized_input_shapes(self, input_shape, expected_length):
        """Test various input shapes and their expected output lengths."""
        # Arrange
        if len(input_shape) == 2:
            contour_mat = np.ones(input_shape, dtype=np.int32)
        else:
            contour_mat = np.ones(input_shape, dtype=np.int32)

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        # For 3D arrays, actual length depends on padding, so we check max possible
        if len(input_shape) == 3:
            assert len(result) <= expected_length
        else:
            assert len(result) == expected_length

    def test_maintains_opencv_compliance(self):
        """Test that returned contours maintain OpenCV compliance."""
        # Arrange
        contour_mat = np.array(
            [
                [10, 20],
                [30, 40],
                [50, 60],
            ]
        )

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        for contour in result:
            assert isinstance(contour, np.ndarray)
            assert contour.dtype == np.int32
            assert contour.ndim == 2
            assert contour.shape[1] == 2  # x, y coordinates

    def test_break_on_all_padding_3d(self):
        """Test that processing stops when encountering all-padding contour in 3D array."""
        # Arrange
        contour_mat = np.array(
            [
                [  # First contour - valid
                    [10, 20],
                    [30, 40],
                    [-1, -1],
                ],
                [  # Second contour - all padding (should break here)
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ],
                [  # Third contour - valid but should not be processed
                    [50, 60],
                    [70, 80],
                    [90, 100],
                ],
            ]
        )

        # Act
        result = get_contour_stack(contour_mat)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 1  # Only first contour should be processed
        expected_contour = np.array([[10, 20], [30, 40]], dtype=np.int32)
        np.testing.assert_array_equal(result[0], expected_contour)
