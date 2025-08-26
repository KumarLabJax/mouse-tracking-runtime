import numpy as np
import pytest

from mouse_tracking.utils.run_length_encode import rle


class TestRLEBasicFunctionality:
    """Test basic run-length encoding functionality."""

    def test_simple_runs(self):
        """Test encoding of simple consecutive runs."""
        # Arrange
        input_array = np.array([1, 1, 2, 2, 2, 3])
        expected_starts = np.array([0, 2, 5])
        expected_durations = np.array([2, 3, 1])
        expected_values = np.array([1, 2, 3])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_single_element(self):
        """Test encoding of single element array."""
        # Arrange
        input_array = np.array([42])
        expected_starts = np.array([0])
        expected_durations = np.array([1])
        expected_values = np.array([42])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_all_same_values(self):
        """Test encoding when all elements are identical."""
        # Arrange
        input_array = np.array([7, 7, 7, 7, 7])
        expected_starts = np.array([0])
        expected_durations = np.array([5])
        expected_values = np.array([7])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_all_different_values(self):
        """Test encoding when all elements are different."""
        # Arrange
        input_array = np.array([1, 2, 3, 4, 5])
        expected_starts = np.array([0, 1, 2, 3, 4])
        expected_durations = np.array([1, 1, 1, 1, 1])
        expected_values = np.array([1, 2, 3, 4, 5])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)


class TestRLEEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test encoding of empty array."""
        # Arrange
        input_array = np.array([])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        assert starts is None
        assert durations is None
        assert values is None

    def test_two_element_same(self):
        """Test encoding of two identical elements."""
        # Arrange
        input_array = np.array([5, 5])
        expected_starts = np.array([0])
        expected_durations = np.array([2])
        expected_values = np.array([5])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_two_element_different(self):
        """Test encoding of two different elements."""
        # Arrange
        input_array = np.array([1, 2])
        expected_starts = np.array([0, 1])
        expected_durations = np.array([1, 1])
        expected_values = np.array([1, 2])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)


class TestRLEDataTypes:
    """Test different numpy data types."""

    def test_integer_types(self):
        """Test with different integer types."""
        # Arrange
        test_cases = [
            np.array([1, 1, 2], dtype=np.int8),
            np.array([1, 1, 2], dtype=np.int16),
            np.array([1, 1, 2], dtype=np.int32),
            np.array([1, 1, 2], dtype=np.int64),
            np.array([1, 1, 2], dtype=np.uint8),
            np.array([1, 1, 2], dtype=np.uint16),
        ]

        for input_array in test_cases:
            # Act
            with pytest.warns(DeprecationWarning):
                starts, durations, values = rle(input_array)

            # Assert
            np.testing.assert_array_equal(starts, [0, 2])
            np.testing.assert_array_equal(durations, [2, 1])
            np.testing.assert_array_equal(values, [1, 2])

    def test_float_types(self):
        """Test with floating point numbers."""
        # Arrange
        input_array = np.array([1.5, 1.5, 2.7, 2.7, 2.7])
        expected_starts = np.array([0, 2])
        expected_durations = np.array([2, 3])
        expected_values = np.array([1.5, 2.7])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_boolean_type(self):
        """Test with boolean arrays."""
        # Arrange
        input_array = np.array([True, True, False, False, True])
        expected_starts = np.array([0, 2, 4])
        expected_durations = np.array([2, 2, 1])
        expected_values = np.array([True, False, True])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)


class TestRLESpecialValues:
    """Test with special numerical values."""

    def test_with_zeros(self):
        """Test encoding arrays containing zeros."""
        # Arrange
        input_array = np.array([0, 0, 1, 1, 0])
        expected_starts = np.array([0, 2, 4])
        expected_durations = np.array([2, 2, 1])
        expected_values = np.array([0, 1, 0])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_with_negative_numbers(self):
        """Test encoding arrays with negative numbers."""
        # Arrange
        input_array = np.array([-1, -1, 0, 0, 1, 1])
        expected_starts = np.array([0, 2, 4])
        expected_durations = np.array([2, 2, 2])
        expected_values = np.array([-1, 0, 1])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_with_nan_values(self):
        """Test encoding arrays containing NaN values.

        Note: NaN != NaN in NumPy, so consecutive NaNs are treated as separate runs.
        """
        # Arrange
        input_array = np.array([1.0, np.nan, np.nan, 2.0])
        # Since NaN != NaN, each NaN is a separate run
        expected_starts = np.array([0, 1, 2, 3])
        expected_durations = np.array([1, 1, 1, 1])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        # NaN comparison requires special handling
        assert values[0] == 1.0
        assert np.isnan(values[1])
        assert np.isnan(values[2])
        assert values[3] == 2.0


class TestRLEInputTypes:
    """Test different input types and conversions."""

    def test_python_list_input(self):
        """Test with Python list as input."""
        # Arrange
        input_list = [1, 1, 2, 2, 3]
        expected_starts = np.array([0, 2, 4])
        expected_durations = np.array([2, 2, 1])
        expected_values = np.array([1, 2, 3])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_list)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_tuple_input(self):
        """Test with tuple as input."""
        # Arrange
        input_tuple = (1, 1, 2, 2, 3)
        expected_starts = np.array([0, 2, 4])
        expected_durations = np.array([2, 2, 1])
        expected_values = np.array([1, 2, 3])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_tuple)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)


class TestRLEComplexPatterns:
    """Test complex run patterns."""

    def test_alternating_pattern(self):
        """Test alternating values pattern."""
        # Arrange
        input_array = np.array([1, 2, 1, 2, 1, 2])
        expected_starts = np.array([0, 1, 2, 3, 4, 5])
        expected_durations = np.array([1, 1, 1, 1, 1, 1])
        expected_values = np.array([1, 2, 1, 2, 1, 2])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)

    def test_long_runs_mixed_with_short(self):
        """Test mix of long and short runs."""
        # Arrange
        input_array = np.array([1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3])
        #                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12]
        # Run 1: Five 1's starting at index 0
        # Run 2: One 2 starting at index 5
        # Run 3: Seven 3's starting at index 6
        expected_starts = np.array([0, 5, 6])
        expected_durations = np.array([5, 1, 7])
        expected_values = np.array([1, 2, 3])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(durations, expected_durations)
        np.testing.assert_array_equal(values, expected_values)


class TestRLEReturnTypes:
    """Test return value types and properties."""

    def test_return_types_non_empty(self):
        """Test that return types are correct for non-empty arrays."""
        # Arrange
        input_array = np.array([1, 1, 2])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        assert isinstance(starts, np.ndarray)
        assert isinstance(durations, np.ndarray)
        assert isinstance(values, np.ndarray)

    def test_return_types_empty(self):
        """Test that return types are correct for empty arrays."""
        # Arrange
        input_array = np.array([])

        # Act
        with pytest.warns(DeprecationWarning):
            starts, durations, values = rle(input_array)

        # Assert
        assert starts is None
        assert durations is None
        assert values is None

    def test_return_array_lengths_consistent(self):
        """Test that all returned arrays have the same length."""
        # Arrange
        test_cases = [
            np.array([1, 1, 2, 2, 3]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1]),
        ]

        for input_array in test_cases:
            # Act
            with pytest.warns(DeprecationWarning):
                starts, durations, values = rle(input_array)

            # Assert
            assert len(starts) == len(durations) == len(values)


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize(
    "input_data,expected_result",
    [
        # Basic cases
        ([1, 1, 2, 2, 2], ([0, 2], [2, 3], [1, 2])),
        ([1], ([0], [1], [1])),
        ([1, 2, 3], ([0, 1, 2], [1, 1, 1], [1, 2, 3])),
        # Special values
        ([0, 0, 1, 1], ([0, 2], [2, 2], [0, 1])),
        ([-1, -1, 0, 1], ([0, 2, 3], [2, 1, 1], [-1, 0, 1])),
        # Boolean
        ([True, False, False, True], ([0, 1, 3], [1, 2, 1], [True, False, True])),
    ],
)
def test_rle_parametrized(input_data, expected_result):
    """Parametrized test for various input/output combinations."""
    # Arrange
    input_array = np.array(input_data)
    expected_starts, expected_durations, expected_values = expected_result

    # Act
    with pytest.warns(DeprecationWarning):
        starts, durations, values = rle(input_array)

    # Assert
    np.testing.assert_array_equal(starts, expected_starts)
    np.testing.assert_array_equal(durations, expected_durations)
    np.testing.assert_array_equal(values, expected_values)


def test_rle_roundtrip_reconstruction():
    """Test that RLE encoding can be used to reconstruct original array."""
    # Arrange
    original_array = np.array([1, 1, 2, 2, 2, 3, 4, 4, 4, 4])

    # Act
    with pytest.warns(DeprecationWarning):
        starts, durations, values = rle(original_array)

    # Reconstruct array from RLE
    reconstructed = np.concatenate(
        [
            np.full(duration, value)
            for duration, value in zip(durations, values, strict=False)
        ]
    )

    # Assert
    np.testing.assert_array_equal(original_array, reconstructed)
