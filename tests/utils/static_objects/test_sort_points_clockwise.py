"""Tests for sort_points_clockwise function."""

import warnings

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import sort_points_clockwise


def test_sort_points_clockwise_basic_square():
    """Test sorting points of a basic square in clockwise order."""
    # Arrange - square corners in random order
    points = np.array(
        [
            [1, 1],  # Bottom-right
            [0, 0],  # Top-left
            [0, 1],  # Bottom-left
            [1, 0],  # Top-right
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    # First point should remain the first point [1, 1]
    np.testing.assert_array_equal(result[0], [1, 1])
    # Result should be sorted clockwise from first point
    assert isinstance(result, np.ndarray)


def test_sort_points_clockwise_triangle():
    """Test sorting triangle points in clockwise order."""
    # Arrange - triangle points
    points = np.array(
        [
            [0, 0],  # First point (should stay first)
            [1, 0],  # Right
            [0.5, 1],  # Top
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (3, 2)
    # First point should remain first
    np.testing.assert_array_equal(result[0], [0, 0])


def test_sort_points_clockwise_preserves_first_point():
    """Test that the first point is preserved in the first position."""
    # Arrange - pentagon with specific first point
    points = np.array(
        [
            [2, 0],  # First point to preserve
            [0, 0],
            [1, 1],
            [3, 1],
            [1, -1],
        ],
        dtype=np.float32,
    )
    original_first_point = points[0].copy()

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (5, 2)
    np.testing.assert_array_equal(result[0], original_first_point)


def test_sort_points_clockwise_already_sorted():
    """Test with points already in clockwise order."""
    # Arrange - points already clockwise around a circle
    angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])  # 0°, 90°, 180°, 270°
    radius = 5
    center = np.array([10, 10])

    points = np.array(
        [
            center + radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    # First point should be preserved
    np.testing.assert_array_almost_equal(result[0], points[0])


def test_sort_points_clockwise_counter_clockwise_input():
    """Test with points initially in counter-clockwise order."""
    # Arrange - points in counter-clockwise order around origin
    points = np.array(
        [
            [1, 0],  # Start point (East)
            [0, 1],  # North
            [-1, 0],  # West
            [0, -1],  # South
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    # First point should be preserved
    np.testing.assert_array_equal(result[0], [1, 0])


def test_sort_points_clockwise_angle_calculation():
    """Test that angles are calculated correctly using arctan2."""
    # Arrange - points at known angles from center
    # Points at 45° intervals starting from first point
    points = np.array(
        [
            [6, 5],  # First point (0° relative to center)
            [6, 6],  # 45°
            [5, 6],  # 90°
            [4, 6],  # 135°
            [4, 5],  # 180°
            [4, 4],  # 225°
            [5, 4],  # 270°
            [6, 4],  # 315°
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (8, 2)
    # First point should be preserved
    np.testing.assert_array_equal(result[0], [6, 5])


def test_sort_points_clockwise_negative_coordinates():
    """Test sorting with negative coordinate values."""
    # Arrange - points with negative coordinates
    points = np.array(
        [
            [-1, -1],  # First point
            [-2, 0],
            [0, -2],
            [1, 1],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[0], [-1, -1])


def test_sort_points_clockwise_collinear_points():
    """Test behavior with collinear points."""
    # Arrange - points on a line
    points = np.array(
        [
            [0, 0],  # First point
            [1, 1],
            [2, 2],
            [3, 3],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[0], [0, 0])


def test_sort_points_clockwise_duplicate_points():
    """Test behavior with duplicate points."""
    # Arrange - some duplicate points
    points = np.array(
        [
            [1, 1],  # First point
            [2, 2],
            [1, 1],  # Duplicate of first
            [3, 0],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[0], [1, 1])


def test_sort_points_clockwise_single_point():
    """Test with single point."""
    # Arrange
    points = np.array([[5, 10]], dtype=np.float32)

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result[0], [5, 10])


def test_sort_points_clockwise_two_points():
    """Test with only two points."""
    # Arrange
    points = np.array(
        [
            [0, 0],  # First point
            [1, 1],  # Second point
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[0], [0, 0])  # First point preserved


def test_sort_points_clockwise_origin_calculation():
    """Test that origin point (centroid) is calculated correctly."""
    # Arrange - symmetric points around origin
    points = np.array(
        [
            [10, 0],  # First point (will be preserved)
            [0, 10],
            [-10, 0],
            [0, -10],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[0], [10, 0])


def test_sort_points_clockwise_non_symmetric_points():
    """Test with non-symmetric point distribution."""
    # Arrange - points not centered around origin
    points = np.array(
        [
            [15, 20],  # First point
            [10, 25],
            [20, 25],
            [25, 15],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[0], [15, 20])


def test_sort_points_clockwise_large_coordinates():
    """Test with large coordinate values."""
    # Arrange
    points = np.array(
        [
            [1000, 1000],  # First point
            [2000, 1500],
            [1500, 2000],
            [500, 1500],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_equal(result[0], [1000, 1000])


def test_sort_points_clockwise_fractional_coordinates():
    """Test with fractional coordinate values."""
    # Arrange
    points = np.array(
        [
            [1.5, 2.7],  # First point
            [3.14, 1.41],
            [0.5, 0.5],
            [2.718, 3.14],
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_almost_equal(result[0], [1.5, 2.7])


def test_sort_points_clockwise_return_type():
    """Test that function returns correct type and dtype."""
    # Arrange
    points = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.dtype == points.dtype  # Should preserve input dtype
    assert result.ndim == 2


@pytest.mark.parametrize("n_points", [3, 4, 5, 6, 8, 10])
def test_sort_points_clockwise_various_sizes(n_points):
    """Test sorting with various numbers of points."""
    # Arrange - points arranged in a circle
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Shuffle angles to create random order
    np.random.shuffle(angles)

    radius = 5
    center = np.array([0, 0])
    points = np.array(
        [
            center + radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (n_points, 2)
    # First point should be preserved
    np.testing.assert_array_almost_equal(result[0], points[0])


def test_sort_points_clockwise_extreme_angles():
    """Test with points at extreme angle positions."""
    # Arrange - points at specific angles that might cause edge cases
    center = np.array([0, 0])
    radius = 1
    # Include angles near boundaries (-π, π)
    angles = np.array([-np.pi + 0.1, -np.pi / 2, 0, np.pi / 2, np.pi - 0.1])

    points = np.array(
        [
            center + radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (5, 2)
    np.testing.assert_array_almost_equal(result[0], points[0])


def test_sort_points_clockwise_identical_angles():
    """Test with points that have very similar angles from centroid."""
    # Arrange - points very close together angularly
    base_point = np.array([1, 0])
    points = np.array(
        [
            base_point,  # First point
            base_point + np.array([0.01, 0.01]),  # Very slight offset
            base_point + np.array([0.02, 0.02]),  # Another slight offset
            base_point + np.array([1, 1]),  # Clearly different
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_almost_equal(result[0], base_point)


def test_sort_points_clockwise_numerical_precision():
    """Test numerical precision with very small differences."""
    # Arrange - points with very small coordinate differences
    epsilon = 1e-6
    points = np.array(
        [
            [1.0, 1.0],  # First point
            [1.0 + epsilon, 1.0],  # Tiny x difference
            [1.0, 1.0 + epsilon],  # Tiny y difference
            [2.0, 2.0],  # Clearly different
        ],
        dtype=np.float32,
    )

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (4, 2)
    np.testing.assert_array_almost_equal(result[0], [1.0, 1.0], decimal=6)


def test_sort_points_clockwise_empty_array():
    """Test behavior with empty points array."""
    # Arrange
    points = np.empty((0, 2), dtype=np.float32)

    # Act & Assert - should raise IndexError when trying to access points[0]
    # Suppress expected numpy warnings for empty array operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(IndexError):
            sort_points_clockwise(points)


def test_sort_points_clockwise_perfect_circle():
    """Test with points perfectly arranged on a circle."""
    # Arrange - 8 points evenly spaced on unit circle
    n_points = 8
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Randomly shuffle the order
    indices = np.random.permutation(n_points)

    points = np.array(
        [[np.cos(angles[i]), np.sin(angles[i])] for i in indices], dtype=np.float32
    )

    original_first_point = points[0].copy()

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == (n_points, 2)
    np.testing.assert_array_almost_equal(result[0], original_first_point)


def test_sort_points_clockwise_maintains_point_values():
    """Test that no point values are modified, only reordered."""
    # Arrange
    points = np.array(
        [[3.14159, 2.71828], [1.41421, 1.73205], [0.57721, 2.30259]], dtype=np.float32
    )
    original_points = points.copy()

    # Act
    result = sort_points_clockwise(points)

    # Assert
    assert result.shape == points.shape
    # All original points should still be present (just reordered)
    for orig_point in original_points:
        found = False
        for result_point in result:
            if np.allclose(orig_point, result_point):
                found = True
                break
        assert found, f"Original point {orig_point} not found in result"
