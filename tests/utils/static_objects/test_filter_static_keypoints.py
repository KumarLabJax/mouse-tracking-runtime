"""Tests for filter_static_keypoints function."""

import warnings

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import filter_static_keypoints


def test_filter_static_keypoints_static_predictions():
    """Test filtering with perfectly static keypoint predictions."""
    # Arrange - identical predictions (no motion)
    predictions = np.array(
        [
            [[10, 20], [30, 40], [50, 60]],
            [[10, 20], [30, 40], [50, 60]],
            [[10, 20], [30, 40], [50, 60]],
        ],
        dtype=np.float32,
    )
    tolerance = 25.0

    # Act
    result = filter_static_keypoints(predictions, tolerance)

    # Assert
    expected = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)
    assert result.shape == (3, 2)


def test_filter_static_keypoints_small_motion_within_tolerance():
    """Test filtering with small motion within tolerance."""
    # Arrange - small variations within tolerance
    predictions = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[10.1, 20.1], [30.1, 40.1]],
            [[9.9, 19.9], [29.9, 39.9]],
        ],
        dtype=np.float32,
    )
    tolerance = 1.0

    # Act
    result = filter_static_keypoints(predictions, tolerance)

    # Assert - should return the mean
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_motion_exceeds_tolerance_raises_error():
    """Test that ValueError is raised when motion exceeds tolerance."""
    # Arrange - large motion that exceeds tolerance
    predictions = np.array(
        [
            [[0, 0], [10, 10]],
            [[50, 50], [60, 60]],  # Large motion
            [[100, 100], [110, 110]],  # Even larger motion
        ],
        dtype=np.float32,
    )
    tolerance = 1.0  # Very tight tolerance

    # Act & Assert
    with pytest.raises(ValueError, match="Predictions are moving!"):
        filter_static_keypoints(predictions, tolerance)


def test_filter_static_keypoints_wrong_shape_raises_assertion():
    """Test that AssertionError is raised for wrong input shape."""
    # Arrange - wrong shape (2D instead of 3D)
    predictions = np.array([[10, 20], [30, 40]], dtype=np.float32)

    # Act & Assert
    with pytest.raises(AssertionError):
        filter_static_keypoints(predictions)


def test_filter_static_keypoints_single_prediction():
    """Test with single prediction (no motion by definition)."""
    # Arrange - single prediction
    predictions = np.array([[[15, 25], [35, 45], [55, 65]]], dtype=np.float32)

    # Act
    result = filter_static_keypoints(predictions)

    # Assert - should return the single prediction unchanged
    expected = predictions[0]
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("tolerance", [0.1, 1.0, 5.0, 10.0, 25.0, 50.0])
def test_filter_static_keypoints_various_tolerances(tolerance):
    """Test filtering with various tolerance values."""
    # Arrange - predictions with small controlled motion
    motion_size = tolerance * 0.5  # Motion within tolerance
    predictions = np.array(
        [
            [[10, 20]],
            [[10 + motion_size, 20 + motion_size]],
            [[10 - motion_size, 20 - motion_size]],
        ],
        dtype=np.float32,
    )

    # Act
    result = filter_static_keypoints(predictions, tolerance)

    # Assert - should pass and return mean
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_motion_calculation_standard_deviation():
    """Test that motion is calculated using standard deviation correctly."""
    # Arrange - controlled predictions to verify std calculation
    predictions = np.array(
        [[[0, 0], [10, 10]], [[1, 1], [11, 11]], [[2, 2], [12, 12]]], dtype=np.float32
    )

    # Calculate expected standard deviation manually
    # std_x = [1, 1], std_y = [1, 1] → motion = [sqrt(2), sqrt(2)]

    # Should pass with tolerance > sqrt(2)
    result = filter_static_keypoints(predictions, tolerance=2.0)
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)

    # Should fail with tolerance < sqrt(2)
    with pytest.raises(ValueError):
        filter_static_keypoints(predictions, tolerance=1.0)


def test_filter_static_keypoints_hypot_distance_calculation():
    """Test that motion uses hypot (Euclidean distance) correctly."""
    # Arrange - predictions where one keypoint has motion only in x-direction
    predictions = np.array(
        [
            [[0, 5], [0, 0]],  # Second keypoint has no motion
            [[3, 5], [0, 0]],  # First keypoint moves 3 pixels in x
            [[4, 5], [0, 0]],  # First keypoint moves 4 pixels in x
        ],
        dtype=np.float32,
    )

    # First keypoint: std_x = std([0,3,4]) ≈ 2.0, std_y = 0 → motion ≈ 2.0
    # Second keypoint: std_x = 0, std_y = 0 → motion = 0

    # Should pass with tolerance > 2.0
    result = filter_static_keypoints(predictions, tolerance=3.0)
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_multi_keypoint_different_motions():
    """Test with multiple keypoints having different amounts of motion."""
    # Arrange - some keypoints static, others moving
    predictions = np.array(
        [
            [[0, 0], [10, 10], [20, 20]],  # All at base positions
            [[0, 0], [10.5, 10.5], [20, 20]],  # Only middle keypoint moves slightly
            [[0, 0], [10, 10], [20, 20]],  # Back to base
        ],
        dtype=np.float32,
    )

    # Only middle keypoint has motion
    tolerance = 1.0

    # Act
    result = filter_static_keypoints(predictions, tolerance)

    # Assert
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_edge_case_exactly_at_tolerance():
    """Test behavior when motion is exactly at tolerance threshold."""
    # Arrange - motion exactly at tolerance
    tolerance = 2.0
    motion_distance = tolerance  # Exactly at threshold

    predictions = np.array(
        [
            [[0, 0]],
            [[motion_distance, 0]],  # Motion exactly equal to tolerance
            [[0, 0]],
        ],
        dtype=np.float32,
    )

    # Should pass (motion <= tolerance)
    result = filter_static_keypoints(predictions, tolerance)
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_large_number_keypoints():
    """Test with many keypoints to verify performance and correctness."""
    # Arrange - many keypoints with small motion
    n_keypoints = 20
    n_predictions = 5
    base_positions = np.random.rand(n_keypoints, 2) * 100

    predictions = []
    for _ in range(n_predictions):
        # Add small random motion
        noise = np.random.normal(0, 0.1, (n_keypoints, 2))
        predictions.append(base_positions + noise)

    predictions = np.array(predictions, dtype=np.float32)

    # Act
    result = filter_static_keypoints(predictions, tolerance=1.0)

    # Assert
    assert result.shape == (n_keypoints, 2)
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_empty_predictions_handles_gracefully():
    """Test behavior with empty predictions array - should handle gracefully."""
    # Arrange - empty array with correct 3D shape
    predictions = np.zeros((0, 4, 2), dtype=np.float32)

    # Act - suppress expected numpy warnings for empty array operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = filter_static_keypoints(predictions)

    # Assert - should handle gracefully and return empty result with correct shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)
    # Result should be all NaN for empty input (due to np.mean of empty array)
    assert np.all(np.isnan(result))


def test_filter_static_keypoints_return_type_and_dtype():
    """Test that function returns correct type and dtype."""
    # Arrange
    predictions = np.array(
        [[[1.5, 2.5], [3.5, 4.5]], [[1.6, 2.6], [3.6, 4.6]]], dtype=np.float32
    )

    # Act
    result = filter_static_keypoints(predictions)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32  # np.mean preserves input dtype
    assert result.ndim == 2


def test_filter_static_keypoints_asymmetric_motion():
    """Test with asymmetric motion patterns."""
    # Arrange - motion only in one direction for some keypoints
    predictions = np.array(
        [
            [[0, 0], [0, 0]],
            [[1, 0], [0, 1]],  # First moves in x, second in y
            [[0, 0], [0, 0]],
        ],
        dtype=np.float32,
    )

    # Both keypoints have similar motion magnitude
    tolerance = 1.0

    # Act
    result = filter_static_keypoints(predictions, tolerance)

    # Assert
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


@pytest.mark.parametrize("n_predictions,n_keypoints", [(2, 1), (3, 2), (5, 4), (10, 8)])
def test_filter_static_keypoints_various_dimensions(n_predictions, n_keypoints):
    """Test with various numbers of predictions and keypoints."""
    # Arrange - random static predictions
    predictions = np.random.rand(n_predictions, n_keypoints, 2).astype(np.float32)
    # Make them static by copying the first prediction
    for i in range(1, n_predictions):
        predictions[i] = predictions[0]

    # Act
    result = filter_static_keypoints(predictions)

    # Assert
    assert result.shape == (n_keypoints, 2)
    np.testing.assert_array_almost_equal(result, predictions[0])


def test_filter_static_keypoints_default_tolerance():
    """Test that default tolerance value works correctly."""
    # Arrange - predictions with motion within default tolerance (25.0)
    predictions = np.array(
        [
            [[0, 0]],
            [[10, 10]],  # Motion magnitude = sqrt(200) ≈ 14.14 < 25.0
            [[0, 0]],
        ],
        dtype=np.float32,
    )

    # Act - use default tolerance
    result = filter_static_keypoints(predictions)

    # Assert - should pass with default tolerance
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)


def test_filter_static_keypoints_negative_coordinates():
    """Test behavior with negative coordinate values."""
    # Arrange - predictions with negative coordinates
    predictions = np.array(
        [
            [[-10, -20], [30, -40]],
            [[-9.9, -19.9], [30.1, -39.9]],
            [[-10.1, -20.1], [29.9, -40.1]],
        ],
        dtype=np.float32,
    )

    # Act
    result = filter_static_keypoints(predictions, tolerance=1.0)

    # Assert
    expected_mean = np.mean(predictions, axis=0)
    np.testing.assert_array_almost_equal(result, expected_mean)
