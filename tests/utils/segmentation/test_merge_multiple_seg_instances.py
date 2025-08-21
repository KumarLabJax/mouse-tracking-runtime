"""
Unit tests for the merge_multiple_seg_instances function from mouse_tracking.utils.segmentation.

This module tests the merge_multiple_seg_instances function which merges multiple segmentation
predictions together into padded matrices for batch processing.
"""

import numpy as np
import pytest

from mouse_tracking.utils.segmentation import merge_multiple_seg_instances


class TestMergeMultipleSegInstances:
    """Test class for merge_multiple_seg_instances function."""

    def test_single_matrix_basic(self):
        """Test with single matrix and flag array."""
        # Arrange
        matrix = np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32)
        flag = np.array([1, 0], dtype=np.int32)
        matrix_list = [matrix]
        flag_list = [flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (1, 2, 2, 2)
        assert result_flags.shape == (1, 2)
        assert result_matrix.dtype == np.int32
        assert result_flags.dtype == np.int32

        expected_matrix = np.array(
            [[[[10, 20], [30, 40]], [[50, 60], [70, 80]]]], dtype=np.int32
        )
        expected_flags = np.array([[1, 0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_multiple_matrices_same_shape(self):
        """Test with multiple matrices of the same shape."""
        # Arrange
        matrix1 = np.array([[[10, 20]], [[30, 40]]], dtype=np.int32)
        matrix2 = np.array([[[50, 60]], [[70, 80]]], dtype=np.int32)
        flag1 = np.array([1, 0], dtype=np.int32)
        flag2 = np.array([1, 1], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (2, 2, 1, 2)
        assert result_flags.shape == (2, 2)

        expected_matrix = np.array(
            [[[[10, 20]], [[30, 40]]], [[[50, 60]], [[70, 80]]]], dtype=np.int32
        )
        expected_flags = np.array([[1, 0], [1, 1]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_multiple_matrices_different_shapes(self):
        """Test with multiple matrices of different shapes - core functionality."""
        # Arrange
        matrix1 = np.array(
            [[[10, 20], [30, 40]], [[50, 60], [70, 80]]], dtype=np.int32
        )  # (2, 2, 2)
        matrix2 = np.array([[[90, 100]]], dtype=np.int32)  # (1, 1, 2)
        matrix3 = np.array(
            [[[110, 120]], [[130, 140]], [[150, 160]]], dtype=np.int32
        )  # (3, 1, 2)
        flag1 = np.array([1, 0], dtype=np.int32)
        flag2 = np.array([1], dtype=np.int32)
        flag3 = np.array([1, 1, 0], dtype=np.int32)
        matrix_list = [matrix1, matrix2, matrix3]
        flag_list = [flag1, flag2, flag3]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (3, 3, 2, 2)  # Max shapes: (3, 2, 2)
        assert result_flags.shape == (3, 3)

        expected_matrix = np.array(
            [
                [[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[-1, -1], [-1, -1]]],
                [[[90, 100], [-1, -1]], [[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]],
                [
                    [[110, 120], [-1, -1]],
                    [[130, 140], [-1, -1]],
                    [[150, 160], [-1, -1]],
                ],
            ],
            dtype=np.int32,
        )
        expected_flags = np.array([[1, 0, -1], [1, -1, -1], [1, 1, 0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_custom_default_value(self):
        """Test with custom default padding value."""
        # Arrange
        matrix1 = np.array([[[10, 20]]], dtype=np.int32)
        matrix2 = np.array([[[30, 40]], [[50, 60]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([1, 0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]
        default_val = -999

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list, default_val
        )

        # Assert
        assert result_matrix.shape == (2, 2, 1, 2)
        assert result_flags.shape == (2, 2)

        expected_matrix = np.array(
            [[[[10, 20]], [[-999, -999]]], [[[30, 40]], [[50, 60]]]], dtype=np.int32
        )
        expected_flags = np.array([[1, -999], [1, 0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_zero_default_value(self):
        """Test with zero as default padding value."""
        # Arrange
        matrix1 = np.array([[[10, 20]]], dtype=np.int32)
        matrix2 = np.array([[[30, 40]], [[50, 60]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([1, 0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]
        default_val = 0

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list, default_val
        )

        # Assert
        expected_matrix = np.array(
            [[[[10, 20]], [[0, 0]]], [[[30, 40]], [[50, 60]]]], dtype=np.int32
        )
        expected_flags = np.array([[1, 0], [1, 0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_empty_matrices_list(self):
        """Test with empty matrices and flags lists - should return default data."""
        # Arrange
        matrix_list = []
        flag_list = []

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (1, 1, 1, 2)
        assert result_flags.shape == (1, 1)
        assert result_matrix.dtype == np.int32
        assert result_flags.dtype == np.int32

        expected_matrix = np.full([1, 1, 1, 2], -1, dtype=np.int32)
        expected_flags = np.full([1, 1], -1, dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_no_detections_scenario_real_world_crash(self):
        """Test real-world scenario: videos without mice - function should handle gracefully.

        Previously this would crash with:
        "zero-size array to reduction operation maximum which has no identity"

        Now the function handles empty lists gracefully by returning default padded data.
        """
        # Arrange - Simulate the exact scenario from multi-segmentation pipeline
        # when no mice are detected in any frame
        frame_contours = []  # No contours detected in any frame
        frame_flags = []  # No flags for any frame

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            frame_contours, frame_flags
        )

        # Assert - Should return default padded data instead of crashing
        assert result_matrix.shape == (1, 1, 1, 2)
        assert result_flags.shape == (1, 1)
        assert result_matrix.dtype == np.int32
        assert result_flags.dtype == np.int32

        expected_matrix = np.full([1, 1, 1, 2], -1, dtype=np.int32)
        expected_flags = np.full([1, 1], -1, dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_no_detections_with_custom_default_value(self):
        """Test that empty lists scenario returns default data with custom default value."""
        # Arrange
        matrix_list = []
        flag_list = []
        custom_default = -999

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list, custom_default
        )

        # Assert - Should return default padded data with custom default value
        assert result_matrix.shape == (1, 1, 1, 2)
        assert result_flags.shape == (1, 1)
        assert result_matrix.dtype == np.int32
        assert result_flags.dtype == np.int32

        expected_matrix = np.full([1, 1, 1, 2], custom_default, dtype=np.int32)
        expected_flags = np.full([1, 1], custom_default, dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_edge_case_zero_predictions_various_defaults(self):
        """Test zero predictions scenario with various default values to ensure consistency."""
        # Arrange
        matrix_list = []
        flag_list = []

        # Test with different default values - all should return consistent default data
        for default_val in [-1, 0, 1, -100, 100, -999]:
            # Act
            result_matrix, result_flags = merge_multiple_seg_instances(
                matrix_list, flag_list, default_val
            )

            # Assert
            assert result_matrix.shape == (1, 1, 1, 2)
            assert result_flags.shape == (1, 1)
            assert result_matrix.dtype == np.int32
            assert result_flags.dtype == np.int32

            expected_matrix = np.full([1, 1, 1, 2], default_val, dtype=np.int32)
            expected_flags = np.full([1, 1], default_val, dtype=np.int32)

            np.testing.assert_array_equal(result_matrix, expected_matrix)
            np.testing.assert_array_equal(result_flags, expected_flags)

    def test_single_empty_matrix(self):
        """Test with single empty matrix (zero segmentation data)."""
        # Arrange
        matrix = np.zeros((1, 0, 2), dtype=np.int32)  # dim2 = 0
        flag = np.zeros((1,), dtype=np.int32)
        matrix_list = [matrix]
        flag_list = [flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (1, 1, 0, 2)
        assert result_flags.shape == (1, 1)

        # Should be filled with default values since original had no segmentation data
        expected_matrix = np.full((1, 1, 0, 2), -1, dtype=np.int32)
        expected_flags = np.full((1, 1), -1, dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_mixed_empty_and_valid_matrices(self):
        """Test with mix of empty and valid matrices."""
        # Arrange
        matrix1 = np.array([[[10, 20]]], dtype=np.int32)  # Valid
        matrix2 = np.zeros((1, 0, 2), dtype=np.int32)  # Empty (dim2 = 0)
        matrix3 = np.array([[[30, 40]], [[50, 60]]], dtype=np.int32)  # Valid
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([1], dtype=np.int32)
        flag3 = np.array([1, 0], dtype=np.int32)
        matrix_list = [matrix1, matrix2, matrix3]
        flag_list = [flag1, flag2, flag3]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (3, 2, 1, 2)
        assert result_flags.shape == (3, 2)

        expected_matrix = np.array(
            [
                [[[10, 20]], [[-1, -1]]],
                [
                    [[-1, -1]],
                    [[-1, -1]],
                ],  # Empty matrix gets skipped, filled with defaults
                [[[30, 40]], [[50, 60]]],
            ],
            dtype=np.int32,
        )
        expected_flags = np.array(
            [
                [1, -1],
                [-1, -1],  # Empty matrix gets skipped, filled with defaults
                [1, 0],
            ],
            dtype=np.int32,
        )

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_all_empty_matrices(self):
        """Test with all empty matrices (all dim2 = 0)."""
        # Arrange
        matrix1 = np.zeros((1, 0, 2), dtype=np.int32)
        matrix2 = np.zeros((2, 0, 2), dtype=np.int32)
        flag1 = np.zeros((1,), dtype=np.int32)
        flag2 = np.zeros((2,), dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (2, 2, 0, 2)
        assert result_flags.shape == (2, 2)

        # All should be filled with default values
        expected_matrix = np.full((2, 2, 0, 2), -1, dtype=np.int32)
        expected_flags = np.full((2, 2), -1, dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_mismatched_list_lengths(self):
        """Test that function raises AssertionError when list lengths don't match."""
        # Arrange
        matrix1 = np.array([[[10, 20]]], dtype=np.int32)
        matrix2 = np.array([[[30, 40]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        matrix_list = [matrix1, matrix2]  # 2 matrices
        flag_list = [flag1]  # 1 flag array

        # Act & Assert
        with pytest.raises(AssertionError):
            merge_multiple_seg_instances(matrix_list, flag_list)

    def test_different_matrix_data_types(self):
        """Test with different input data types (should be converted to int32)."""
        # Arrange
        matrix1 = np.array([[[10, 20]]], dtype=np.float32)
        matrix2 = np.array([[[30, 40]]], dtype=np.int16)
        flag1 = np.array([1], dtype=np.bool_)
        flag2 = np.array([0], dtype=np.int64)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.dtype == np.int32
        assert result_flags.dtype == np.int32

        expected_matrix = np.array([[[[10, 20]]], [[[30, 40]]]], dtype=np.int32)
        expected_flags = np.array([[1], [0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_large_matrices(self):
        """Test with large matrices to verify memory efficiency."""
        # Arrange
        large_matrix = np.random.randint(0, 100, (10, 50, 2), dtype=np.int32)
        small_matrix = np.array([[[1, 2]]], dtype=np.int32)
        large_flag = np.random.randint(0, 2, (10,), dtype=np.int32)
        small_flag = np.array([1], dtype=np.int32)
        matrix_list = [large_matrix, small_matrix]
        flag_list = [large_flag, small_flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (2, 10, 50, 2)
        assert result_flags.shape == (2, 10)

        # Check that large matrix data is preserved
        np.testing.assert_array_equal(result_matrix[0], large_matrix)
        np.testing.assert_array_equal(result_flags[0], large_flag)

        # Check that small matrix data is padded correctly
        expected_small = np.full((10, 50, 2), -1, dtype=np.int32)
        expected_small[0, 0] = [1, 2]
        np.testing.assert_array_equal(result_matrix[1], expected_small)

        expected_small_flag = np.full((10,), -1, dtype=np.int32)
        expected_small_flag[0] = 1
        np.testing.assert_array_equal(result_flags[1], expected_small_flag)

    def test_negative_coordinates(self):
        """Test with negative coordinate values."""
        # Arrange
        matrix1 = np.array([[[-10, -20]]], dtype=np.int32)
        matrix2 = np.array([[[30, -40]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        expected_matrix = np.array([[[[-10, -20]]], [[[30, -40]]]], dtype=np.int32)
        expected_flags = np.array([[1], [0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_very_large_coordinates(self):
        """Test with very large coordinate values."""
        # Arrange
        max_val = np.iinfo(np.int32).max
        matrix1 = np.array([[[max_val, max_val]]], dtype=np.int32)
        matrix2 = np.array([[[0, 0]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        expected_matrix = np.array([[[[max_val, max_val]]], [[[0, 0]]]], dtype=np.int32)
        expected_flags = np.array([[1], [0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    @pytest.mark.parametrize("default_val", [-1, 0, 1, -100, 100, -999])
    def test_various_default_values(self, default_val):
        """Test with various default padding values."""
        # Arrange
        matrix1 = np.array([[[10, 20]]], dtype=np.int32)
        matrix2 = np.array([[[30, 40]], [[50, 60]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([1, 0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list, default_val
        )

        # Assert
        expected_matrix = np.array(
            [[[[10, 20]], [[default_val, default_val]]], [[[30, 40]], [[50, 60]]]],
            dtype=np.int32,
        )
        expected_flags = np.array([[1, default_val], [1, 0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_return_type_and_shape(self):
        """Test that return types and shapes are correct."""
        # Arrange
        matrix = np.array([[[10, 20]]], dtype=np.int32)
        flag = np.array([1], dtype=np.int32)
        matrix_list = [matrix]
        flag_list = [flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert isinstance(result_matrix, np.ndarray)
        assert isinstance(result_flags, np.ndarray)
        assert result_matrix.dtype == np.int32
        assert result_flags.dtype == np.int32
        assert (
            len(result_matrix.shape) == 4
        )  # [n_predictions, max_dim1, max_dim2, max_dim3]
        assert len(result_flags.shape) == 2  # [n_predictions, max_flag_dim]

    def test_memory_layout_c_contiguous(self):
        """Test that resulting arrays have efficient memory layout."""
        # Arrange
        matrix = np.array([[[10, 20]]], dtype=np.int32)
        flag = np.array([1], dtype=np.int32)
        matrix_list = [matrix]
        flag_list = [flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.flags.c_contiguous or result_matrix.flags.f_contiguous
        assert result_flags.flags.c_contiguous or result_flags.flags.f_contiguous

    def test_no_modification_of_input(self):
        """Test that input matrices and flags are not modified."""
        # Arrange
        original_matrix = np.array([[[10, 20]]], dtype=np.int32)
        original_flag = np.array([1], dtype=np.int32)
        matrix_copy = original_matrix.copy()
        flag_copy = original_flag.copy()
        matrix_list = [original_matrix]
        flag_list = [original_flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        np.testing.assert_array_equal(original_matrix, matrix_copy)
        np.testing.assert_array_equal(original_flag, flag_copy)
        assert result_matrix is not original_matrix
        assert result_flags is not original_flag

    def test_edge_case_all_zero_coordinates(self):
        """Test with all zero coordinates."""
        # Arrange
        matrix1 = np.array([[[0, 0]]], dtype=np.int32)
        matrix2 = np.array([[[0, 0]], [[0, 0]]], dtype=np.int32)
        flag1 = np.array([0], dtype=np.int32)
        flag2 = np.array([0, 0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        expected_matrix = np.array(
            [[[[0, 0]], [[-1, -1]]], [[[0, 0]], [[0, 0]]]], dtype=np.int32
        )
        expected_flags = np.array([[0, -1], [0, 0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_max_shape_calculation(self):
        """Test that max shape calculation is correct."""
        # Arrange
        matrix1 = np.array([[[1, 2]]], dtype=np.int32)  # (1, 1, 2)
        matrix2 = np.array([[[3, 4]], [[5, 6]]], dtype=np.int32)  # (2, 1, 2)
        matrix3 = np.array([[[7, 8], [9, 10]]], dtype=np.int32)  # (1, 2, 2)
        flag1 = np.array([1], dtype=np.int32)  # (1,)
        flag2 = np.array([1, 0], dtype=np.int32)  # (2,)
        flag3 = np.array([1], dtype=np.int32)  # (1,)
        matrix_list = [matrix1, matrix2, matrix3]
        flag_list = [flag1, flag2, flag3]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        # Max shapes should be: matrix (2, 2, 2), flags (2,)
        assert result_matrix.shape == (3, 2, 2, 2)
        assert result_flags.shape == (3, 2)

    def test_integration_with_realistic_segmentation_data(self):
        """Integration test with realistic segmentation data."""
        # Arrange - create realistic data like from multi-mouse segmentation
        mouse1_contour = np.array(
            [[[100, 100], [200, 100]], [[150, 150], [250, 150]]], dtype=np.int32
        )
        mouse2_contour = np.array([[[300, 300]]], dtype=np.int32)
        mouse1_flag = np.array([1, 0], dtype=np.int32)
        mouse2_flag = np.array([1], dtype=np.int32)
        matrix_list = [mouse1_contour, mouse2_contour]
        flag_list = [mouse1_flag, mouse2_flag]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        assert result_matrix.shape == (2, 2, 2, 2)
        assert result_flags.shape == (2, 2)

        expected_matrix = np.array(
            [
                [[[100, 100], [200, 100]], [[150, 150], [250, 150]]],
                [[[300, 300], [-1, -1]], [[-1, -1], [-1, -1]]],
            ],
            dtype=np.int32,
        )
        expected_flags = np.array([[1, 0], [1, -1]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_single_point_contours(self):
        """Test with contours containing single points."""
        # Arrange
        matrix1 = np.array([[[100, 200]]], dtype=np.int32)
        matrix2 = np.array([[[300, 400]]], dtype=np.int32)
        flag1 = np.array([1], dtype=np.int32)
        flag2 = np.array([0], dtype=np.int32)
        matrix_list = [matrix1, matrix2]
        flag_list = [flag1, flag2]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        expected_matrix = np.array([[[[100, 200]]], [[[300, 400]]]], dtype=np.int32)
        expected_flags = np.array([[1], [0]], dtype=np.int32)

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)

    def test_comprehensive_shape_combinations(self):
        """Test comprehensive combinations of different shapes."""
        # Arrange
        matrix1 = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32
        )  # (2, 2, 2)
        matrix2 = np.array([[[9, 10]]], dtype=np.int32)  # (1, 1, 2)
        matrix3 = np.array(
            [[[11, 12]], [[13, 14]], [[15, 16]]], dtype=np.int32
        )  # (3, 1, 2)
        matrix4 = np.array(
            [[[17, 18], [19, 20], [21, 22]]], dtype=np.int32
        )  # (1, 3, 2)
        flag1 = np.array([1, 0], dtype=np.int32)  # (2,)
        flag2 = np.array([1], dtype=np.int32)  # (1,)
        flag3 = np.array([1, 1, 0], dtype=np.int32)  # (3,)
        flag4 = np.array([1], dtype=np.int32)  # (1,)
        matrix_list = [matrix1, matrix2, matrix3, matrix4]
        flag_list = [flag1, flag2, flag3, flag4]

        # Act
        result_matrix, result_flags = merge_multiple_seg_instances(
            matrix_list, flag_list
        )

        # Assert
        # Max shapes should be: matrix (3, 3, 2), flags (3,)
        assert result_matrix.shape == (4, 3, 3, 2)
        assert result_flags.shape == (4, 3)

        # Check that all data is preserved and padded correctly
        expected_matrix = np.array(
            [
                [  # matrix1
                    [[1, 2], [3, 4], [-1, -1]],
                    [[5, 6], [7, 8], [-1, -1]],
                    [[-1, -1], [-1, -1], [-1, -1]],
                ],
                [  # matrix2
                    [[9, 10], [-1, -1], [-1, -1]],
                    [[-1, -1], [-1, -1], [-1, -1]],
                    [[-1, -1], [-1, -1], [-1, -1]],
                ],
                [  # matrix3
                    [[11, 12], [-1, -1], [-1, -1]],
                    [[13, 14], [-1, -1], [-1, -1]],
                    [[15, 16], [-1, -1], [-1, -1]],
                ],
                [  # matrix4
                    [[17, 18], [19, 20], [21, 22]],
                    [[-1, -1], [-1, -1], [-1, -1]],
                    [[-1, -1], [-1, -1], [-1, -1]],
                ],
            ],
            dtype=np.int32,
        )
        expected_flags = np.array(
            [[1, 0, -1], [1, -1, -1], [1, 1, 0], [1, -1, -1]], dtype=np.int32
        )

        np.testing.assert_array_equal(result_matrix, expected_matrix)
        np.testing.assert_array_equal(result_flags, expected_flags)
