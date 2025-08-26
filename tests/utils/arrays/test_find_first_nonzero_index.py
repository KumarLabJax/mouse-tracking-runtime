import numpy as np
import pytest

from mouse_tracking.utils.arrays import find_first_nonzero_index


class TestSafeFindFirstBasicFunctionality:
    """Test basic functionality of find_first_nonzero_index."""

    def test_first_nonzero_at_beginning(self):
        """Test when first non-zero element is at index 0."""
        # Arrange
        input_array = np.array([5, 0, 0, 3])
        expected_index = 0

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_first_nonzero_in_middle(self):
        """Test when first non-zero element is in the middle."""
        # Arrange
        input_array = np.array([0, 0, 7, 0, 2])
        expected_index = 2

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_first_nonzero_at_end(self):
        """Test when first non-zero element is at the last index."""
        # Arrange
        input_array = np.array([0, 0, 0, 9])
        expected_index = 3

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_multiple_nonzero_elements(self):
        """Test array with multiple non-zero elements returns first index."""
        # Arrange
        input_array = np.array([0, 3, 5, 7, 2])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_all_nonzero_elements(self):
        """Test array where all elements are non-zero."""
        # Arrange
        input_array = np.array([1, 2, 3, 4, 5])
        expected_index = 0

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index


class TestSafeFindFirstEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zero_elements(self):
        """Test array where all elements are zero."""
        # Arrange
        input_array = np.array([0, 0, 0, 0])
        expected_result = -1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_result

    def test_empty_array(self):
        """Test empty array."""
        # Arrange
        input_array = np.array([])
        expected_result = -1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_result

    def test_single_zero_element(self):
        """Test array with single zero element."""
        # Arrange
        input_array = np.array([0])
        expected_result = -1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_result

    def test_single_nonzero_element(self):
        """Test array with single non-zero element."""
        # Arrange
        input_array = np.array([42])
        expected_index = 0

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index


class TestSafeFindFirstDataTypes:
    """Test different numpy data types."""

    def test_integer_types(self):
        """Test with different integer types."""
        # Arrange
        test_cases = [
            (np.array([0, 1, 2], dtype=np.int8), 1),
            (np.array([0, 1, 2], dtype=np.int16), 1),
            (np.array([0, 1, 2], dtype=np.int32), 1),
            (np.array([0, 1, 2], dtype=np.int64), 1),
            (np.array([0, 1, 2], dtype=np.uint8), 1),
            (np.array([0, 1, 2], dtype=np.uint16), 1),
        ]

        for input_array, expected_index in test_cases:
            # Act
            result = find_first_nonzero_index(input_array)

            # Assert
            assert result == expected_index

    def test_float_types(self):
        """Test with floating point numbers."""
        # Arrange
        input_array = np.array([0.0, 0.0, 1.5, 2.7])
        expected_index = 2

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_complex_numbers(self):
        """Test with complex numbers."""
        # Arrange
        input_array = np.array([0 + 0j, 1 + 2j, 3 + 0j])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_boolean_type(self):
        """Test with boolean arrays."""
        # Arrange
        input_array = np.array([False, False, True, False])
        expected_index = 2

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_all_false_boolean(self):
        """Test with all False boolean array."""
        # Arrange
        input_array = np.array([False, False, False])
        expected_result = -1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_result


class TestSafeFindFirstSpecialValues:
    """Test with special numerical values."""

    def test_with_negative_numbers(self):
        """Test with negative numbers (which are non-zero)."""
        # Arrange
        input_array = np.array([0, -1, 0, 2])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_with_very_small_numbers(self):
        """Test with very small but non-zero numbers."""
        # Arrange
        input_array = np.array([0.0, 1e-10, 0.0])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_with_infinity(self):
        """Test with infinity values."""
        # Arrange
        input_array = np.array([0.0, np.inf, 0.0])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_with_negative_infinity(self):
        """Test with negative infinity values."""
        # Arrange
        input_array = np.array([0.0, -np.inf, 0.0])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_with_nan_values(self):
        """Test with NaN values (NaN is considered non-zero)."""
        # Arrange
        input_array = np.array([0.0, np.nan, 0.0])
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index


class TestSafeFindFirstInputTypes:
    """Test different input types and conversions."""

    def test_python_list_input(self):
        """Test with Python list as input."""
        # Arrange
        input_list = [0, 0, 3, 0]
        expected_index = 2

        # Act
        result = find_first_nonzero_index(input_list)

        # Assert
        assert result == expected_index

    def test_tuple_input(self):
        """Test with tuple as input."""
        # Arrange
        input_tuple = (0, 5, 0, 7)
        expected_index = 1

        # Act
        result = find_first_nonzero_index(input_tuple)

        # Assert
        assert result == expected_index

    def test_nested_list_input(self):
        """Test with nested list (should work with np.where)."""
        # Arrange
        input_nested = [[0, 1], [2, 0]]
        expected_index = 0  # First non-zero in flattened view

        # Act
        result = find_first_nonzero_index(input_nested)

        # Assert
        assert result == expected_index


class TestSafeFindFirstReturnType:
    """Test return value types and properties."""

    def test_return_type_is_int_for_found(self):
        """Test that return type is int when element is found."""
        # Arrange
        input_array = np.array([0, 1, 0])

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert isinstance(result, int | np.integer)

    def test_return_type_is_int_for_not_found(self):
        """Test that return type is int when no element is found."""
        # Arrange
        input_array = np.array([0, 0, 0])

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert isinstance(result, int | np.integer)
        assert result == -1

    def test_return_value_bounds(self):
        """Test that returned index is within valid bounds."""
        # Arrange
        input_arrays = [
            np.array([1, 0, 0]),  # Should return 0
            np.array([0, 1, 0]),  # Should return 1
            np.array([0, 0, 1]),  # Should return 2
            np.array([0, 0, 0]),  # Should return -1
        ]

        for _i, input_array in enumerate(input_arrays):
            # Act
            result = find_first_nonzero_index(input_array)

            # Assert
            if result != -1:
                assert 0 <= result < len(input_array)
                # Verify the element at returned index is actually non-zero
                assert input_array[result] != 0


class TestSafeFindFirstLargeArrays:
    """Test performance and correctness with larger arrays."""

    def test_large_array_with_early_nonzero(self):
        """Test large array with non-zero element near beginning."""
        # Arrange
        input_array = np.zeros(10000)
        input_array[5] = 1
        expected_index = 5

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_large_array_with_late_nonzero(self):
        """Test large array with non-zero element near end."""
        # Arrange
        input_array = np.zeros(10000)
        input_array[9995] = 1
        expected_index = 9995

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_index

    def test_large_array_all_zeros(self):
        """Test large array with all zeros."""
        # Arrange
        input_array = np.zeros(10000)
        expected_result = -1

        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        assert result == expected_result


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize(
    "input_data,expected_result",
    [
        # Basic cases
        ([0, 0, 1, 0], 2),
        ([1, 0, 0, 0], 0),
        ([0, 0, 0, 1], 3),
        ([1, 2, 3, 4], 0),
        # Edge cases
        ([0, 0, 0, 0], -1),
        ([0], -1),
        ([1], 0),
        ([], -1),
        # Special values
        ([0, -1, 0], 1),
        ([0.0, 1e-10], 1),
        ([False, True], 1),
        ([False, False], -1),
        # Different types
        ([0 + 0j, 1 + 0j], 1),
        ([0.0, 0.0, 2.5], 2),
    ],
)
def test_find_first_nonzero_index_parametrized(input_data, expected_result):
    """Parametrized test for various input/output combinations."""
    # Arrange
    input_array = np.array(input_data)

    # Act
    result = find_first_nonzero_index(input_array)

    # Assert
    assert result == expected_result


def test_find_first_nonzero_index_correctness_verification():
    """Test that the function correctly identifies the first non-zero element."""
    # Arrange
    test_arrays = [
        np.array([0, 0, 5, 3, 0, 7]),
        np.array([1, 2, 3]),
        np.array([0, 0, 0, 0, 1]),
        np.random.choice([0, 1], size=100, p=[0.8, 0.2]),  # Random sparse array
    ]

    for input_array in test_arrays:
        # Act
        result = find_first_nonzero_index(input_array)

        # Assert
        if result == -1:
            # If -1 returned, verify all elements are zero
            assert np.all(input_array == 0)
        else:
            # If index returned, verify it's the first non-zero
            assert input_array[result] != 0
            # Verify all elements before this index are zero
            if result > 0:
                assert np.all(input_array[:result] == 0)


def test_find_first_nonzero_index_multidimensional_arrays():
    """Test behavior with multidimensional arrays (np.where returns first dimension indices)."""
    # Arrange
    input_2d = np.array([[0, 0], [1, 0]])
    # np.where(input_2d) returns ([1], [0]) - row indices and column indices
    # np.where(input_2d)[0] gives [1] - the row index of first non-zero element
    expected_index = 1  # First row index with non-zero element

    # Act
    result = find_first_nonzero_index(input_2d)

    # Assert
    assert result == expected_index

    # Arrange - 3D array
    input_3d = np.zeros((3, 2, 2))
    input_3d[2, 0, 1] = 5  # Non-zero element at position [2, 0, 1]
    # np.where(input_3d)[0] will return [2] - the first dimension index
    expected_index_3d = 2  # First dimension index with non-zero element

    # Act
    result_3d = find_first_nonzero_index(input_3d)

    # Assert
    assert result_3d == expected_index_3d
