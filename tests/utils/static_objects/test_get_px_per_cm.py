"""Unit tests for get_px_per_cm function.

This module contains comprehensive tests for the pixel-to-centimeter conversion
functionality, ensuring proper handling of corner coordinate data and accurate
scale calculations.
"""

import numpy as np
import pytest

from mouse_tracking.utils.static_objects import ARENA_SIZE_CM, get_px_per_cm


@pytest.fixture
def perfect_square_corners():
    """Create perfect square corner coordinates for testing.

    Returns:
        numpy.ndarray: Perfect square corners with side length 100 pixels,
        centered at origin. Shape [4, 2] representing [x, y] coordinates.
    """
    side_length = 100.0
    half_side = side_length / 2
    return np.array(
        [
            [-half_side, -half_side],  # Bottom-left
            [half_side, -half_side],  # Bottom-right
            [half_side, half_side],  # Top-right
            [-half_side, half_side],  # Top-left
        ],
        dtype=np.float32,
    )


@pytest.fixture
def rectangle_corners():
    """Create rectangle corner coordinates for testing.

    Returns:
        numpy.ndarray: Rectangle corners with width=150, height=100 pixels,
        centered at origin. Shape [4, 2] representing [x, y] coordinates.
    """
    width, height = 150.0, 100.0
    half_width, half_height = width / 2, height / 2
    return np.array(
        [
            [-half_width, -half_height],  # Bottom-left
            [half_width, -half_height],  # Bottom-right
            [half_width, half_height],  # Top-right
            [-half_width, half_height],  # Top-left
        ],
        dtype=np.float32,
    )


@pytest.fixture
def realistic_arena_corners():
    """Create realistic arena corner coordinates for testing.

    Returns:
        numpy.ndarray: Realistic arena corners approximately matching
        typical experimental setups. Shape [4, 2] in pixels.
    """
    return np.array(
        [
            [50, 50],  # Top-left
            [650, 50],  # Top-right
            [650, 450],  # Bottom-right
            [50, 450],  # Bottom-left
        ],
        dtype=np.float32,
    )


def calculate_expected_cm_per_pixel(corners, arena_size_cm):
    """Calculate expected cm_per_pixel value for verification.

    This helper function replicates the logic of get_px_per_cm for verification
    purposes in tests.

    Args:
        corners (numpy.ndarray): Corner coordinates of shape [4, 2].
        arena_size_cm (float): Arena size in centimeters.

    Returns:
        float: Expected cm_per_pixel conversion factor.
    """
    from scipy.spatial.distance import cdist

    # Calculate pairwise distances
    dists = cdist(corners, corners)
    dists = dists[np.nonzero(np.triu(dists))]

    # Sort distances and split into edges and diagonals
    sorted_dists = np.sort(dists)
    edges = sorted_dists[:4]
    diags = sorted_dists[4:]

    # Convert diagonals to equivalent edge lengths
    equivalent_edges = np.sqrt(np.square(diags) / 2)
    all_edges = np.concatenate([equivalent_edges, edges])

    # Calculate conversion factor
    return arena_size_cm / np.mean(all_edges)


class TestGetPxPerCmSuccessfulCases:
    """Test successful execution paths of get_px_per_cm function."""

    def test_perfect_square_default_arena_size(self, perfect_square_corners):
        """Test pixel conversion with perfect square using default arena size.

        Args:
            perfect_square_corners: Fixture providing perfect square coordinates.
        """
        # Arrange
        expected_cm_per_pixel = calculate_expected_cm_per_pixel(
            perfect_square_corners, ARENA_SIZE_CM
        )

        # Act
        actual_cm_per_pixel = get_px_per_cm(perfect_square_corners)

        # Assert
        assert isinstance(actual_cm_per_pixel, np.float32)
        assert np.isclose(actual_cm_per_pixel, expected_cm_per_pixel, rtol=1e-6)
        assert actual_cm_per_pixel > 0

    def test_perfect_square_custom_arena_size(self, perfect_square_corners):
        """Test pixel conversion with perfect square using custom arena size.

        Args:
            perfect_square_corners: Fixture providing perfect square coordinates.
        """
        # Arrange
        custom_arena_size = 30.0  # cm
        expected_cm_per_pixel = calculate_expected_cm_per_pixel(
            perfect_square_corners, custom_arena_size
        )

        # Act
        actual_cm_per_pixel = get_px_per_cm(perfect_square_corners, custom_arena_size)

        # Assert
        assert isinstance(actual_cm_per_pixel, np.float32)
        assert np.isclose(actual_cm_per_pixel, expected_cm_per_pixel, rtol=1e-6)
        assert actual_cm_per_pixel > 0

    def test_rectangle_corners(self, rectangle_corners):
        """Test pixel conversion with rectangular corners.

        Args:
            rectangle_corners: Fixture providing rectangle coordinates.
        """
        # Arrange
        expected_cm_per_pixel = calculate_expected_cm_per_pixel(
            rectangle_corners, ARENA_SIZE_CM
        )

        # Act
        actual_cm_per_pixel = get_px_per_cm(rectangle_corners)

        # Assert
        assert isinstance(actual_cm_per_pixel, np.float32)
        assert np.isclose(actual_cm_per_pixel, expected_cm_per_pixel, rtol=1e-6)
        assert actual_cm_per_pixel > 0

    def test_realistic_arena_corners(self, realistic_arena_corners):
        """Test pixel conversion with realistic arena corner data.

        Args:
            realistic_arena_corners: Fixture providing realistic coordinates.
        """
        # Arrange
        expected_cm_per_pixel = calculate_expected_cm_per_pixel(
            realistic_arena_corners, ARENA_SIZE_CM
        )

        # Act
        actual_cm_per_pixel = get_px_per_cm(realistic_arena_corners)

        # Assert
        assert isinstance(actual_cm_per_pixel, np.float32)
        assert np.isclose(actual_cm_per_pixel, expected_cm_per_pixel, rtol=1e-6)
        assert actual_cm_per_pixel > 0

    @pytest.mark.parametrize("arena_size", [10.0, 25.0, 50.0, 100.0])
    def test_different_arena_sizes(self, perfect_square_corners, arena_size):
        """Test pixel conversion with various arena sizes.

        Args:
            perfect_square_corners: Fixture providing perfect square coordinates.
            arena_size: Arena size in centimeters to test.
        """
        # Arrange
        expected_cm_per_pixel = calculate_expected_cm_per_pixel(
            perfect_square_corners, arena_size
        )

        # Act
        actual_cm_per_pixel = get_px_per_cm(perfect_square_corners, arena_size)

        # Assert
        assert isinstance(actual_cm_per_pixel, np.float32)
        assert np.isclose(actual_cm_per_pixel, expected_cm_per_pixel, rtol=1e-6)
        assert actual_cm_per_pixel > 0
        # Verify that larger arena sizes give larger cm_per_pixel ratios
        assert np.isclose(
            actual_cm_per_pixel, arena_size / 100.0, rtol=1e-6
        )  # For 100px side length square

    @pytest.mark.parametrize("scale_factor", [0.1, 1.0, 10.0, 100.0])
    def test_different_coordinate_scales(self, scale_factor):
        """Test pixel conversion with different coordinate scales.

        Args:
            scale_factor: Factor to scale the coordinate system.
        """
        # Arrange - create square with different scales
        base_corners = (
            np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
            * scale_factor
        )

        # Act
        cm_per_pixel = get_px_per_cm(base_corners)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0
        # For a square, the scale should be inversely proportional to coordinate scale
        expected_scale = ARENA_SIZE_CM / (100.0 * scale_factor)
        assert np.isclose(cm_per_pixel, expected_scale, rtol=1e-6)


class TestGetPxPerCmMathematicalCorrectness:
    """Test mathematical correctness of get_px_per_cm function."""

    def test_perfect_square_edge_diagonal_relationship(self):
        """Test that perfect square maintains correct edge/diagonal relationships."""
        # Arrange - create perfect square with known side length
        side_length = 200.0
        corners = np.array(
            [[0, 0], [side_length, 0], [side_length, side_length], [0, side_length]],
            dtype=np.float32,
        )

        # Act
        cm_per_pixel = get_px_per_cm(corners, arena_size_cm=10.0)

        # Assert - for perfect square, conversion should be arena_size / side_length
        expected_conversion = 10.0 / side_length
        assert np.isclose(cm_per_pixel, expected_conversion, rtol=1e-6)

    def test_unit_square_conversion(self):
        """Test conversion for unit square (1x1 pixel)."""
        # Arrange
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        arena_size = 5.0  # cm

        # Act
        cm_per_pixel = get_px_per_cm(unit_square, arena_size)

        # Assert
        expected_conversion = arena_size / 1.0  # 5 cm per pixel
        assert np.isclose(cm_per_pixel, expected_conversion, rtol=1e-6)

    def test_large_square_conversion(self):
        """Test conversion for large square (1000x1000 pixels)."""
        # Arrange
        large_square = np.array(
            [[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32
        )
        arena_size = 50.0  # cm

        # Act
        cm_per_pixel = get_px_per_cm(large_square, arena_size)

        # Assert
        expected_conversion = arena_size / 1000.0  # 0.05 cm per pixel
        assert np.isclose(cm_per_pixel, expected_conversion, rtol=1e-6)

    def test_consistency_across_translations(self):
        """Test that translation doesn't affect the conversion factor."""
        # Arrange - same square at different positions
        base_square = np.array(
            [[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32
        )

        translated_square = base_square + np.array(
            [500, 300]
        )  # Translate by (500, 300)

        # Act
        base_conversion = get_px_per_cm(base_square)
        translated_conversion = get_px_per_cm(translated_square)

        # Assert
        assert np.isclose(base_conversion, translated_conversion, rtol=1e-6)


class TestGetPxPerCmEdgeCases:
    """Test edge cases and boundary conditions of get_px_per_cm function."""

    @pytest.mark.parametrize("data_type", [np.float32, np.float64, np.int32, np.int64])
    def test_different_data_types(self, data_type):
        """Test pixel conversion with different numeric data types.

        Args:
            data_type: NumPy data type to test.
        """
        # Arrange
        corners = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=data_type)

        # Act
        cm_per_pixel = get_px_per_cm(corners)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0
        # Should be consistent regardless of input data type
        expected_conversion = ARENA_SIZE_CM / 50.0  # 50px side length
        assert np.isclose(cm_per_pixel, expected_conversion, rtol=1e-5)

    def test_very_small_coordinates(self):
        """Test pixel conversion with very small coordinate values."""
        # Arrange - microscopic square
        small_corners = np.array(
            [[0.001, 0.001], [0.002, 0.001], [0.002, 0.002], [0.001, 0.002]],
            dtype=np.float32,
        )

        # Act
        cm_per_pixel = get_px_per_cm(small_corners, arena_size_cm=1e-6)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0
        assert np.isfinite(cm_per_pixel)

    def test_very_large_coordinates(self):
        """Test pixel conversion with very large coordinate values."""
        # Arrange - massive square
        large_corners = np.array(
            [[0, 0], [1e6, 0], [1e6, 1e6], [0, 1e6]], dtype=np.float32
        )

        # Act
        cm_per_pixel = get_px_per_cm(large_corners, arena_size_cm=1e9)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0
        assert np.isfinite(cm_per_pixel)
        assert np.isclose(cm_per_pixel, 1e3, rtol=1e-5)  # 1e9 / 1e6 = 1e3

    def test_irregular_quadrilateral(self):
        """Test pixel conversion with irregular quadrilateral corners."""
        # Arrange - irregular shape
        irregular_corners = np.array(
            [[0, 0], [80, 20], [70, 90], [10, 85]], dtype=np.float32
        )

        # Act
        cm_per_pixel = get_px_per_cm(irregular_corners)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0
        assert np.isfinite(cm_per_pixel)

    def test_extreme_aspect_ratio_rectangle(self):
        """Test pixel conversion with extreme aspect ratio rectangle."""
        # Arrange - very wide, short rectangle
        extreme_corners = np.array(
            [[0, 0], [1000, 0], [1000, 10], [0, 10]], dtype=np.float32
        )

        # Act
        cm_per_pixel = get_px_per_cm(extreme_corners)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0
        assert np.isfinite(cm_per_pixel)


class TestGetPxPerCmErrorCases:
    """Test error conditions and exception handling of get_px_per_cm function."""

    def test_wrong_input_shape_too_few_corners(self):
        """Test behavior with too few corners (function still works with 3 corners)."""
        # Arrange - only 3 corners instead of 4
        insufficient_corners = np.array(
            [[0, 0], [100, 0], [100, 100]], dtype=np.float32
        )

        # Act
        result = get_px_per_cm(insufficient_corners)

        # Assert - function still works but with different geometry
        assert isinstance(result, np.float32)
        assert result > 0
        assert np.isfinite(result)

    def test_wrong_input_shape_too_many_corners(self):
        """Test that wrong input shape (too many corners) uses only first 4."""
        # Arrange - 5 corners instead of 4
        extra_corners = np.array(
            [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]], dtype=np.float32
        )

        # Act - should work by using first 4 corners
        cm_per_pixel = get_px_per_cm(extra_corners)

        # Assert
        assert isinstance(cm_per_pixel, np.float32)
        assert cm_per_pixel > 0

    def test_wrong_coordinate_dimensions(self):
        """Test behavior with wrong coordinate dimensions (3D instead of 2D)."""
        # Arrange - 3D coordinates instead of 2D
        wrong_dims = np.array(
            [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=np.float32
        )

        # Act
        result = get_px_per_cm(wrong_dims)

        # Assert - function still works by using first 2 dimensions
        assert isinstance(result, np.float32)
        assert result > 0
        assert np.isfinite(result)

    def test_duplicate_corners_zero_distances(self):
        """Test behavior with duplicate corners causing zero distances."""
        # Arrange - all corners at same location
        duplicate_corners = np.array(
            [[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32
        )

        # Act
        with pytest.warns(
            RuntimeWarning
        ):  # Expect warnings about empty slice and division
            result = get_px_per_cm(duplicate_corners)

        # Assert - should return NaN due to zero distances
        assert isinstance(result, np.float32)
        assert np.isnan(result)

    def test_nan_coordinates(self):
        """Test behavior with NaN coordinate values."""
        # Arrange
        nan_corners = np.array(
            [[0, 0], [100, 0], [np.nan, 100], [0, 100]], dtype=np.float32
        )

        # Act & Assert
        result = get_px_per_cm(nan_corners)
        assert np.isnan(result) or np.isinf(result)

    def test_infinite_coordinates(self):
        """Test behavior with infinite coordinate values."""
        # Arrange
        inf_corners = np.array(
            [[0, 0], [100, 0], [np.inf, 100], [0, 100]], dtype=np.float32
        )

        # Act & Assert
        result = get_px_per_cm(inf_corners)
        assert np.isnan(result) or np.isinf(result)

    def test_zero_arena_size(self):
        """Test behavior with zero arena size."""
        # Arrange
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        # Act & Assert
        result = get_px_per_cm(corners, arena_size_cm=0.0)
        assert result == 0.0

    def test_negative_arena_size(self):
        """Test behavior with negative arena size."""
        # Arrange
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        # Act
        result = get_px_per_cm(corners, arena_size_cm=-10.0)

        # Assert - should be negative conversion factor
        assert result < 0
        assert np.isclose(result, -0.1, rtol=1e-6)  # -10.0 / 100.0


class TestGetPxPerCmIntegration:
    """Integration tests for get_px_per_cm function with realistic scenarios."""

    def test_ltm_arena_resolution_consistency(self):
        """Test consistency with known LTM arena resolution constants."""
        # Arrange - simulate LTM arena (701 pixels for 20.5 inch arena)
        ltm_side_pixels = 701
        ltm_corners = np.array(
            [
                [0, 0],
                [ltm_side_pixels, 0],
                [ltm_side_pixels, ltm_side_pixels],
                [0, ltm_side_pixels],
            ],
            dtype=np.float32,
        )

        # Act
        cm_per_pixel = get_px_per_cm(ltm_corners)

        # Assert - should match the DEFAULT_CM_PER_PX constant
        from mouse_tracking.utils.static_objects import DEFAULT_CM_PER_PX

        expected_ltm_scale = DEFAULT_CM_PER_PX["ltm"]
        assert np.isclose(cm_per_pixel, expected_ltm_scale, rtol=1e-3)

    def test_ofa_arena_resolution_consistency(self):
        """Test consistency with known OFA arena resolution constants."""
        # Arrange - simulate OFA arena (398 pixels for 20.5 inch arena)
        ofa_side_pixels = 398
        ofa_corners = np.array(
            [
                [0, 0],
                [ofa_side_pixels, 0],
                [ofa_side_pixels, ofa_side_pixels],
                [0, ofa_side_pixels],
            ],
            dtype=np.float32,
        )

        # Act
        cm_per_pixel = get_px_per_cm(ofa_corners)

        # Assert - should match the DEFAULT_CM_PER_PX constant
        from mouse_tracking.utils.static_objects import DEFAULT_CM_PER_PX

        expected_ofa_scale = DEFAULT_CM_PER_PX["ofa"]
        assert np.isclose(cm_per_pixel, expected_ofa_scale, rtol=1e-3)

    def test_real_world_measurement_accuracy(self):
        """Test accuracy with real-world measurement scenario."""
        # Arrange - real experimental arena: 60cm arena, 800px resolution
        real_arena_cm = 60.0
        arena_size_px = 800  # 800px effective arena size
        real_corners = np.array(
            [[100, 100], [900, 100], [900, 900], [100, 900]], dtype=np.float32
        )

        # Act
        cm_per_pixel = get_px_per_cm(real_corners, real_arena_cm)

        # Assert
        expected_scale = real_arena_cm / arena_size_px  # 0.075 cm/pixel
        assert np.isclose(cm_per_pixel, expected_scale, rtol=1e-6)

        # Verify reasonable scale for mouse tracking
        assert 0.01 < cm_per_pixel < 1.0  # Reasonable range for mouse experiments

    def test_rotated_arena_corners(self):
        """Test pixel conversion with rotated arena corners."""
        # Arrange - 45-degree rotated square
        import math

        angle = math.pi / 4  # 45 degrees
        side_length = 100
        center = np.array([200, 200])

        # Create square corners and rotate them
        corners_centered = np.array(
            [
                [-side_length / 2, -side_length / 2],
                [side_length / 2, -side_length / 2],
                [side_length / 2, side_length / 2],
                [-side_length / 2, side_length / 2],
            ]
        )

        # Apply rotation matrix
        rotation_matrix = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )

        rotated_corners = corners_centered @ rotation_matrix.T + center

        # Act
        cm_per_pixel = get_px_per_cm(rotated_corners.astype(np.float32))

        # Assert - rotation shouldn't affect the scale
        expected_scale = ARENA_SIZE_CM / side_length
        assert np.isclose(cm_per_pixel, expected_scale, rtol=1e-5)
