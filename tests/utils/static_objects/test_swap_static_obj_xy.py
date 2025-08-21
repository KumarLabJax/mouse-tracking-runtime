"""Unit tests for swap_static_obj_xy function.

This module contains comprehensive tests for the static object coordinate swapping
functionality, ensuring proper handling of HDF5 files with various configurations.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from mouse_tracking.utils.static_objects import swap_static_obj_xy


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file for testing.

    Returns:
        Path to temporary HDF5 file that will be cleaned up automatically.
    """
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        yield tmp_file.name
    # Cleanup
    Path(tmp_file.name).unlink(missing_ok=True)


@pytest.fixture
def sample_coordinates_2d():
    """Create sample 2D coordinate data for testing.

    Returns:
        numpy.ndarray: Sample coordinate data of shape [4, 2] representing
        corners in [y, x] format.
    """
    return np.array(
        [[10.5, 20.3], [15.2, 25.7], [18.1, 30.9], [12.8, 22.4]], dtype=np.float32
    )


@pytest.fixture
def sample_coordinates_3d():
    """Create sample 3D coordinate data for testing.

    Returns:
        numpy.ndarray: Sample coordinate data of shape [10, 4, 2] representing
        multiple frames of corner coordinates in [y, x] format.
    """
    return np.random.rand(10, 4, 2).astype(np.float32) * 100


@pytest.fixture
def sample_attributes():
    """Create sample HDF5 attributes for testing.

    Returns:
        dict: Sample attributes to attach to datasets.
    """
    return {
        "confidence": 0.95,
        "model_version": "v1.2.3",
        "timestamp": "2024-01-01T00:00:00",
    }


def create_h5_dataset_with_data(
    file_path,
    dataset_key,
    data,
    attributes=None,
    compression=None,
    compression_opts=None,
):
    """Create an HDF5 file with a dataset containing the specified data.

    Args:
        file_path (str): Path to the HDF5 file to create.
        dataset_key (str): Key for the dataset within the file.
        data (numpy.ndarray): Data to store in the dataset.
        attributes (dict, optional): Attributes to attach to the dataset.
        compression (str, optional): Compression algorithm to use.
        compression_opts (int, optional): Compression level/options.
    """
    with h5py.File(file_path, "w") as f:
        # Create dataset with appropriate compression settings
        if compression is not None:
            dataset = f.create_dataset(
                dataset_key,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
            )
        else:
            dataset = f.create_dataset(dataset_key, data=data)

        # Add attributes if provided
        if attributes:
            for attr_name, attr_value in attributes.items():
                dataset.attrs[attr_name] = attr_value


def verify_coordinates_swapped(original_data, swapped_data):
    """Verify that coordinates have been properly swapped from [y,x] to [x,y].

    Args:
        original_data (numpy.ndarray): Original coordinate data in [y,x] format.
        swapped_data (numpy.ndarray): Data after swapping operation.

    Returns:
        bool: True if coordinates are properly swapped.
    """
    expected_swapped = np.flip(original_data, axis=-1)
    return np.allclose(swapped_data, expected_swapped)


def verify_attributes_preserved(file_path, dataset_key, expected_attributes):
    """Verify that dataset attributes are preserved after swapping operation.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_key (str): Key for the dataset to check.
        expected_attributes (dict): Expected attributes.

    Returns:
        bool: True if all attributes are preserved.
    """
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_key]
        actual_attributes = dict(dataset.attrs.items())
        return actual_attributes == expected_attributes


class TestSwapStaticObjXySuccessfulCases:
    """Test successful execution paths of swap_static_obj_xy function."""

    def test_swap_coordinates_2d_no_compression_no_attributes(
        self, temp_h5_file, sample_coordinates_2d
    ):
        """Test swapping 2D coordinates without compression or attributes.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_2d: Fixture providing sample coordinate data.
        """
        # Arrange
        dataset_key = "arena_corners"
        create_h5_dataset_with_data(temp_h5_file, dataset_key, sample_coordinates_2d)

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            assert verify_coordinates_swapped(sample_coordinates_2d, swapped_data)
            assert swapped_data.dtype == sample_coordinates_2d.dtype
            assert swapped_data.shape == sample_coordinates_2d.shape

    def test_swap_coordinates_3d_no_compression_no_attributes(
        self, temp_h5_file, sample_coordinates_3d
    ):
        """Test swapping 3D coordinates without compression or attributes.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_3d: Fixture providing sample coordinate data.
        """
        # Arrange
        dataset_key = "multi_frame_corners"
        create_h5_dataset_with_data(temp_h5_file, dataset_key, sample_coordinates_3d)

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            assert verify_coordinates_swapped(sample_coordinates_3d, swapped_data)
            assert swapped_data.dtype == sample_coordinates_3d.dtype
            assert swapped_data.shape == sample_coordinates_3d.shape

    def test_swap_coordinates_with_attributes_preserved(
        self, temp_h5_file, sample_coordinates_2d, sample_attributes
    ):
        """Test that dataset attributes are preserved during coordinate swapping.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_2d: Fixture providing sample coordinate data.
            sample_attributes: Fixture providing sample attributes.
        """
        # Arrange
        dataset_key = "food_hopper"
        create_h5_dataset_with_data(
            temp_h5_file,
            dataset_key,
            sample_coordinates_2d,
            attributes=sample_attributes,
        )

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            assert verify_coordinates_swapped(sample_coordinates_2d, swapped_data)
            assert verify_attributes_preserved(
                temp_h5_file, dataset_key, sample_attributes
            )

    @pytest.mark.parametrize("compression_level", [1, 5, 9])
    def test_swap_coordinates_with_gzip_compression(
        self, temp_h5_file, sample_coordinates_2d, compression_level
    ):
        """Test coordinate swapping with different gzip compression levels.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_2d: Fixture providing sample coordinate data.
            compression_level: Compression level to test.
        """
        # Arrange
        dataset_key = "lixit"
        create_h5_dataset_with_data(
            temp_h5_file,
            dataset_key,
            sample_coordinates_2d,
            compression="gzip",
            compression_opts=compression_level,
        )

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            dataset = f[dataset_key]
            assert verify_coordinates_swapped(sample_coordinates_2d, swapped_data)
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == compression_level

    def test_swap_coordinates_with_compression_and_attributes(
        self, temp_h5_file, sample_coordinates_3d, sample_attributes
    ):
        """Test coordinate swapping with both compression and attributes.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_3d: Fixture providing sample coordinate data.
            sample_attributes: Fixture providing sample attributes.
        """
        # Arrange
        dataset_key = "complex_object"
        create_h5_dataset_with_data(
            temp_h5_file,
            dataset_key,
            sample_coordinates_3d,
            attributes=sample_attributes,
            compression="gzip",
            compression_opts=6,
        )

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            dataset = f[dataset_key]
            assert verify_coordinates_swapped(sample_coordinates_3d, swapped_data)
            assert verify_attributes_preserved(
                temp_h5_file, dataset_key, sample_attributes
            )
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 6


class TestSwapStaticObjXyEdgeCases:
    """Test edge cases and boundary conditions of swap_static_obj_xy function."""

    @patch("builtins.print")
    def test_nonexistent_dataset_key_prints_message(
        self, mock_print, temp_h5_file, sample_coordinates_2d
    ):
        """Test that attempting to swap non-existent dataset prints appropriate message.

        Args:
            mock_print: Mock for the print function.
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_2d: Fixture providing sample coordinate data.
        """
        # Arrange
        existing_key = "existing_data"
        nonexistent_key = "nonexistent_data"
        create_h5_dataset_with_data(temp_h5_file, existing_key, sample_coordinates_2d)

        # Act
        swap_static_obj_xy(temp_h5_file, nonexistent_key)

        # Assert
        mock_print.assert_called_once_with(f"{nonexistent_key} not in {temp_h5_file}.")

        # Verify original data remains unchanged
        with h5py.File(temp_h5_file, "r") as f:
            original_data = f[existing_key][:]
            assert np.array_equal(original_data, sample_coordinates_2d)

    def test_empty_h5_file_with_nonexistent_key(self, temp_h5_file):
        """Test behavior when trying to swap key in empty HDF5 file.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
        """
        # Arrange - create empty HDF5 file
        with h5py.File(temp_h5_file, "w") as _:
            pass  # Create empty file

        # Act & Assert
        with patch("builtins.print") as mock_print:
            swap_static_obj_xy(temp_h5_file, "any_key")
            mock_print.assert_called_once_with(f"any_key not in {temp_h5_file}.")

    def test_single_point_coordinates(self, temp_h5_file):
        """Test swapping with single point coordinate data.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
        """
        # Arrange
        single_point = np.array([[5.5, 10.2]], dtype=np.float32)
        dataset_key = "single_point"
        create_h5_dataset_with_data(temp_h5_file, dataset_key, single_point)

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            assert verify_coordinates_swapped(single_point, swapped_data)

    def test_large_coordinate_dataset(self, temp_h5_file):
        """Test swapping with large coordinate dataset.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
        """
        # Arrange - create large dataset
        large_data = np.random.rand(1000, 10, 2).astype(np.float32) * 1000
        dataset_key = "large_dataset"
        create_h5_dataset_with_data(temp_h5_file, dataset_key, large_data)

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            assert verify_coordinates_swapped(large_data, swapped_data)
            assert swapped_data.shape == large_data.shape

    @pytest.mark.parametrize("data_type", [np.float32, np.float64, np.int32, np.int64])
    def test_different_data_types(self, temp_h5_file, data_type):
        """Test coordinate swapping with different numeric data types.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            data_type: NumPy data type to test.
        """
        # Arrange
        test_data = np.array([[1.5, 2.7], [3.2, 4.8]], dtype=data_type)
        dataset_key = f"data_{data_type.__name__}"
        create_h5_dataset_with_data(temp_h5_file, dataset_key, test_data)

        # Act
        swap_static_obj_xy(temp_h5_file, dataset_key)

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_data = f[dataset_key][:]
            assert verify_coordinates_swapped(test_data, swapped_data)
            assert swapped_data.dtype == data_type


class TestSwapStaticObjXyErrorCases:
    """Test error conditions and exception handling of swap_static_obj_xy function."""

    def test_nonexistent_file_raises_error(self):
        """Test that attempting to open non-existent file raises appropriate error."""
        # Arrange
        nonexistent_file = "/path/to/nonexistent/file.h5"

        # Act & Assert
        with pytest.raises((OSError, IOError)):
            swap_static_obj_xy(nonexistent_file, "any_key")

    def test_invalid_h5_file_raises_error(self, temp_h5_file):
        """Test that attempting to open invalid HDF5 file raises appropriate error.

        Args:
            temp_h5_file: Fixture providing temporary file path.
        """
        # Arrange - create file with invalid HDF5 content
        with open(temp_h5_file, "w") as f:
            f.write("This is not a valid HDF5 file")

        # Act & Assert
        with pytest.raises((OSError, IOError)):
            swap_static_obj_xy(temp_h5_file, "any_key")

    def test_read_only_file_raises_error(self, temp_h5_file, sample_coordinates_2d):
        """Test that attempting to modify read-only file raises appropriate error.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
            sample_coordinates_2d: Fixture providing sample coordinate data.
        """
        # Arrange
        dataset_key = "test_data"
        create_h5_dataset_with_data(temp_h5_file, dataset_key, sample_coordinates_2d)

        # Make file read-only
        import os
        import stat

        os.chmod(temp_h5_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # Act & Assert
        try:
            with pytest.raises(OSError):
                swap_static_obj_xy(temp_h5_file, dataset_key)
        finally:
            # Restore write permissions for cleanup
            os.chmod(
                temp_h5_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            )


class TestSwapStaticObjXyIntegration:
    """Integration tests for swap_static_obj_xy function with realistic scenarios."""

    def test_multiple_datasets_swap_specific_one(self, temp_h5_file):
        """Test swapping coordinates in file with multiple datasets.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
        """
        # Arrange - create file with multiple datasets
        arena_data = np.array(
            [[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.float32
        )
        food_data = np.array([[15, 25], [35, 45]], dtype=np.float32)
        lixit_data = np.array([[5, 15]], dtype=np.float32)

        with h5py.File(temp_h5_file, "w") as f:
            f.create_dataset("arena_corners", data=arena_data)
            f.create_dataset("food_hopper", data=food_data)
            f.create_dataset("lixit", data=lixit_data)

        # Act - swap only one dataset
        swap_static_obj_xy(temp_h5_file, "food_hopper")

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            # Verify target dataset was swapped
            swapped_food = f["food_hopper"][:]
            assert verify_coordinates_swapped(food_data, swapped_food)

            # Verify other datasets remain unchanged
            assert np.array_equal(f["arena_corners"][:], arena_data)
            assert np.array_equal(f["lixit"][:], lixit_data)

    def test_realistic_arena_corner_data(self, temp_h5_file):
        """Test with realistic arena corner coordinate data.

        Args:
            temp_h5_file: Fixture providing temporary HDF5 file path.
        """
        # Arrange - realistic arena corner data in [y, x] format
        arena_corners = np.array(
            [
                [50.2, 100.1],  # Top-left
                [50.3, 600.8],  # Top-right
                [450.7, 600.9],  # Bottom-right
                [450.6, 100.2],  # Bottom-left
            ],
            dtype=np.float32,
        )

        attributes = {
            "confidence": 0.98,
            "model_version": "arena_v2.1",
            "pixel_scale": 0.1034,
        }

        create_h5_dataset_with_data(
            temp_h5_file,
            "arena_corners",
            arena_corners,
            attributes=attributes,
            compression="gzip",
            compression_opts=5,
        )

        # Act
        swap_static_obj_xy(temp_h5_file, "arena_corners")

        # Assert
        with h5py.File(temp_h5_file, "r") as f:
            swapped_corners = f["arena_corners"][:]
            expected_corners = np.array(
                [
                    [100.1, 50.2],  # [x, y] format
                    [600.8, 50.3],
                    [600.9, 450.7],
                    [100.2, 450.6],
                ],
                dtype=np.float32,
            )

            assert np.allclose(swapped_corners, expected_corners)
            assert verify_attributes_preserved(
                temp_h5_file, "arena_corners", attributes
            )
            assert f["arena_corners"].compression == "gzip"
            assert f["arena_corners"].compression_opts == 5
