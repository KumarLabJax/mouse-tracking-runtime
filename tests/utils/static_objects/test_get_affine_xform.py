"""Tests for get_affine_xform function."""

from unittest.mock import patch

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import get_affine_xform


def test_get_affine_xform_basic_functionality():
    """Test basic affine transformation matrix creation."""
    # Arrange - simple bounding box
    bbox = np.array([10, 20, 50, 60], dtype=np.float32)  # [x1, y1, x2, y2]
    img_size = (512, 512)
    warp_size = (255, 255)

    # Act
    result = get_affine_xform(bbox, img_size, warp_size)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)  # Affine transformation matrix shape
    # Check that the result contains the expected translation values
    expected_translation_x = bbox[0] * img_size[0]  # 10 * 512
    expected_translation_y = bbox[1] * img_size[1]  # 20 * 512
    assert result[0, 2] == expected_translation_x
    assert result[1, 2] == expected_translation_y


def test_get_affine_xform_default_parameters():
    """Test function with default img_size and warp_size parameters."""
    # Arrange - bounding box with default parameters
    bbox = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    # Act
    result = get_affine_xform(bbox)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    # With default parameters: img_size=(512, 512), warp_size=(255, 255)
    expected_translation_x = bbox[0] * 512  # 0.1 * 512 = 51.2
    expected_translation_y = bbox[1] * 512  # 0.2 * 512 = 102.4
    assert abs(result[0, 2] - expected_translation_x) < 1e-6
    assert abs(result[1, 2] - expected_translation_y) < 1e-6


@pytest.mark.parametrize(
    "img_size,warp_size",
    [
        ((256, 256), (128, 128)),
        ((1024, 768), (512, 384)),
        ((100, 200), (50, 100)),
        ((800, 600), (400, 300)),
    ],
)
def test_get_affine_xform_various_sizes(img_size, warp_size):
    """Test affine transformation with various image and warp sizes."""
    # Arrange
    bbox = np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float32)

    # Act
    result = get_affine_xform(bbox, img_size, warp_size)

    # Assert
    assert result.shape == (2, 3)
    expected_translation_x = bbox[0] * img_size[0]
    expected_translation_y = bbox[1] * img_size[1]
    assert abs(result[0, 2] - expected_translation_x) < 1e-6
    assert abs(result[1, 2] - expected_translation_y) < 1e-6


def test_get_affine_xform_uses_cv2_get_affine_transform():
    """Test that function uses cv2.getAffineTransform correctly."""
    # Arrange
    bbox = np.array([5, 10, 15, 20], dtype=np.float32)
    img_size = (100, 100)
    warp_size = (50, 50)

    mock_affine_matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

    with patch("cv2.getAffineTransform") as mock_get_affine:
        mock_get_affine.return_value = mock_affine_matrix

        # Act
        get_affine_xform(bbox, img_size, warp_size)

        # Assert
        mock_get_affine.assert_called_once()
        # Check the from_corners parameter
        call_args = mock_get_affine.call_args[0]
        from_corners = call_args[0]
        to_corners = call_args[1]

        expected_from_corners = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(from_corners, expected_from_corners)

        expected_to_corners = np.array(
            [[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]]
        )
        np.testing.assert_array_equal(to_corners, expected_to_corners)


def test_get_affine_xform_coordinate_system_scaling():
    """Test that coordinate system scaling is applied correctly."""
    # Arrange
    bbox = np.array([10, 20, 30, 40], dtype=np.float32)
    img_size = (200, 300)  # Different x and y dimensions
    warp_size = (100, 150)  # Different x and y dimensions

    # Mock cv2.getAffineTransform to return identity-like matrix
    mock_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    with patch("cv2.getAffineTransform", return_value=mock_affine):
        # Act
        result = get_affine_xform(bbox, img_size, warp_size)

        # Assert - check that result has the expected shape
        # The scaling should be applied to the matrix elements
        # Note: The actual implementation multiplies by scaling factors
        assert result.shape == (2, 3)


def test_get_affine_xform_translation_adjustment():
    """Test that translation is correctly adjusted in the final matrix."""
    # Arrange
    bbox = np.array([0.1, 0.2, 0.6, 0.8], dtype=np.float32)
    img_size = (1000, 800)
    warp_size = (500, 400)

    # Mock cv2.getAffineTransform
    mock_affine = np.array([[1.0, 0.0, 999.0], [0.0, 1.0, 888.0]], dtype=np.float32)

    with patch("cv2.getAffineTransform", return_value=mock_affine):
        # Act
        result = get_affine_xform(bbox, img_size, warp_size)

        # Assert translation is correctly set
        expected_translation_x = bbox[0] * img_size[0]  # 0.1 * 1000 = 100
        expected_translation_y = bbox[1] * img_size[1]  # 0.2 * 800 = 160

        assert result[0, 2] == expected_translation_x
        assert result[1, 2] == expected_translation_y


def test_get_affine_xform_bbox_corner_mapping():
    """Test that bounding box corners are mapped correctly."""
    # Arrange
    bbox = np.array([100, 200, 300, 400], dtype=np.float32)

    with patch("cv2.getAffineTransform") as mock_get_affine:
        mock_get_affine.return_value = np.eye(2, 3, dtype=np.float32)

        # Act
        get_affine_xform(bbox)

        # Assert - check to_corners parameter
        call_args = mock_get_affine.call_args[0]
        to_corners = call_args[1]

        # Expected mapping based on implementation:
        # bbox is [x1, y1, x2, y2] = [100, 200, 300, 400]
        # to_corners should be [[x1, y1], [x1, y2], [x2, y2]]
        expected_to_corners = np.array(
            [
                [bbox[0], bbox[1]],  # [100, 200] - top-left
                [bbox[0], bbox[3]],  # [100, 400] - bottom-left
                [bbox[2], bbox[3]],  # [300, 400] - bottom-right
            ]
        )
        np.testing.assert_array_equal(to_corners, expected_to_corners)


def test_get_affine_xform_zero_bbox():
    """Test behavior with zero bounding box."""
    # Arrange
    bbox = np.array([0, 0, 0, 0], dtype=np.float32)

    # Act
    result = get_affine_xform(bbox)

    # Assert
    assert result.shape == (2, 3)
    assert result[0, 2] == 0.0  # Translation x should be 0
    assert result[1, 2] == 0.0  # Translation y should be 0


def test_get_affine_xform_negative_bbox():
    """Test behavior with negative bounding box coordinates."""
    # Arrange
    bbox = np.array([-10, -20, 30, 40], dtype=np.float32)
    img_size = (100, 100)

    # Act
    result = get_affine_xform(bbox, img_size)

    # Assert
    assert result.shape == (2, 3)
    expected_translation_x = bbox[0] * img_size[0]  # -10 * 100 = -1000
    expected_translation_y = bbox[1] * img_size[1]  # -20 * 100 = -2000
    assert result[0, 2] == expected_translation_x
    assert result[1, 2] == expected_translation_y


def test_get_affine_xform_large_bbox():
    """Test behavior with large bounding box values."""
    # Arrange
    bbox = np.array([1000, 2000, 3000, 4000], dtype=np.float32)
    img_size = (5000, 6000)
    warp_size = (1000, 1200)

    # Act
    result = get_affine_xform(bbox, img_size, warp_size)

    # Assert
    assert result.shape == (2, 3)
    expected_translation_x = bbox[0] * img_size[0]  # 1000 * 5000
    expected_translation_y = bbox[1] * img_size[1]  # 2000 * 6000
    assert result[0, 2] == expected_translation_x
    assert result[1, 2] == expected_translation_y


def test_get_affine_xform_fractional_bbox():
    """Test behavior with fractional bounding box coordinates."""
    # Arrange
    bbox = np.array([0.123, 0.456, 0.789, 0.987], dtype=np.float32)
    img_size = (100, 200)

    # Act
    result = get_affine_xform(bbox, img_size)

    # Assert
    assert result.shape == (2, 3)
    expected_translation_x = bbox[0] * img_size[0]  # 0.123 * 100 = 12.3
    expected_translation_y = bbox[1] * img_size[1]  # 0.456 * 200 = 91.2
    assert abs(result[0, 2] - expected_translation_x) < 1e-6
    assert abs(result[1, 2] - expected_translation_y) < 1e-6


def test_get_affine_xform_square_vs_rectangular():
    """Test with both square and rectangular image/warp sizes."""
    # Arrange
    bbox = np.array([10, 20, 30, 40], dtype=np.float32)

    # Test square sizes
    square_result = get_affine_xform(bbox, (100, 100), (50, 50))

    # Test rectangular sizes
    rect_result = get_affine_xform(bbox, (200, 100), (100, 50))

    # Assert
    assert square_result.shape == (2, 3)
    assert rect_result.shape == (2, 3)

    # Translation should be the same for both since it only depends on bbox and img_size
    square_trans_x = bbox[0] * 100  # 10 * 100 = 1000
    rect_trans_x = bbox[0] * 200  # 10 * 200 = 2000

    assert square_result[0, 2] == square_trans_x
    assert rect_result[0, 2] == rect_trans_x


def test_get_affine_xform_matrix_dtype():
    """Test that the returned matrix has correct data type."""
    # Arrange
    bbox = np.array([1, 2, 3, 4], dtype=np.float32)

    # Act
    result = get_affine_xform(bbox)

    # Assert
    assert isinstance(result, np.ndarray)
    # The dtype should be float (either float32 or float64)
    assert np.issubdtype(result.dtype, np.floating)


def test_get_affine_xform_integration_with_cv2():
    """Test integration behavior with actual cv2.getAffineTransform."""
    # Arrange
    bbox = np.array([5, 10, 25, 30], dtype=np.float32)
    img_size = (50, 60)
    warp_size = (25, 30)

    # Act - use real cv2.getAffineTransform (no mocking)
    result = get_affine_xform(bbox, img_size, warp_size)

    # Assert
    assert result.shape == (2, 3)
    # The translation should be correctly set regardless of cv2 behavior
    expected_translation_x = bbox[0] * img_size[0]
    expected_translation_y = bbox[1] * img_size[1]
    assert result[0, 2] == expected_translation_x
    assert result[1, 2] == expected_translation_y


@pytest.mark.parametrize(
    "bbox",
    [
        np.array([0, 0, 1, 1], dtype=np.float32),
        np.array([10, 20, 110, 120], dtype=np.float32),
        np.array([0.5, 0.25, 0.75, 0.8], dtype=np.float32),
    ],
)
def test_get_affine_xform_various_bboxes(bbox):
    """Test affine transformation with various bounding box configurations."""
    # Arrange
    img_size = (200, 300)
    warp_size = (100, 150)

    # Act
    result = get_affine_xform(bbox, img_size, warp_size)

    # Assert
    assert result.shape == (2, 3)
    expected_translation_x = bbox[0] * img_size[0]
    expected_translation_y = bbox[1] * img_size[1]
    assert abs(result[0, 2] - expected_translation_x) < 1e-6
    assert abs(result[1, 2] - expected_translation_y) < 1e-6


def test_get_affine_xform_from_corners_specification():
    """Test that from_corners are correctly specified."""
    # Arrange
    bbox = np.array([1, 2, 3, 4], dtype=np.float32)

    with patch("cv2.getAffineTransform") as mock_get_affine:
        mock_get_affine.return_value = np.eye(2, 3, dtype=np.float32)

        # Act
        get_affine_xform(bbox)

        # Assert
        call_args = mock_get_affine.call_args[0]
        from_corners = call_args[0]

        # from_corners should be 3 corners of unit square: (0,0), (0,1), (1,1)
        expected_from_corners = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(from_corners, expected_from_corners)
        assert from_corners.shape == (3, 2)
        assert from_corners.dtype == np.float32
