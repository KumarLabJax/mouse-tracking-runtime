"""Tests for write_pose_clip function."""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from mouse_tracking.utils.writers import write_pose_clip


def test_clips_pose_data_successfully():
    """Test basic clipping of pose data."""
    # Arrange
    clip_indices = [0, 2, 4]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with test data
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            # Create datasets with frame dimension
            points_data = np.random.rand(10, 2, 12, 2).astype(np.float32)
            confidence_data = np.random.rand(10, 2, 12).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.create_dataset("confidence", data=confidence_data)
            poseest.attrs["version"] = [6, 0]
            poseest.attrs["cm_per_pixel"] = 0.1

            # Create static objects
            static_objects = f.create_group("static_objects")
            corners_data = np.array(
                [[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32
            )
            static_objects.create_dataset("corners", data=corners_data)
            static_objects["corners"].attrs["config"] = "corner_config"
            static_objects["corners"].attrs["model"] = "corner_model"

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            # Check that datasets were clipped correctly
            assert "poseest/points" in f
            assert "poseest/confidence" in f
            assert "static_objects/corners" in f

            # Check clipped data shapes
            assert f["poseest/points"].shape == (3, 2, 12, 2)  # 3 frames selected
            assert f["poseest/confidence"].shape == (3, 2, 12)

            # Check that static objects were copied (not clipped)
            assert f["static_objects/corners"].shape == (4, 2)

            # Check that data was actually clipped correctly
            original_points = points_data[clip_indices]
            np.testing.assert_array_equal(f["poseest/points"][:], original_points)

            original_confidence = confidence_data[clip_indices]
            np.testing.assert_array_equal(
                f["poseest/confidence"][:], original_confidence
            )

            # Check that static objects were copied correctly
            np.testing.assert_array_equal(f["static_objects/corners"][:], corners_data)

            # Check that attributes were preserved
            assert f["poseest"].attrs["version"].tolist() == [6, 0]
            assert f["poseest"].attrs["cm_per_pixel"] == 0.1
            assert f["static_objects/corners"].attrs["config"] == "corner_config"
            assert f["static_objects/corners"].attrs["model"] == "corner_model"

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_clips_with_list_indices():
    """Test clipping with list of indices."""
    # Arrange
    clip_indices = [1, 3, 5, 7]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (4, 1, 12, 2)
            expected_data = points_data[clip_indices]
            np.testing.assert_array_equal(f["poseest/points"][:], expected_data)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_clips_with_numpy_array_indices():
    """Test clipping with numpy array indices."""
    # Arrange
    clip_indices = np.array([0, 2, 4, 6])

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(8, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (4, 1, 12, 2)
            expected_data = points_data[clip_indices]
            np.testing.assert_array_equal(f["poseest/points"][:], expected_data)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_clips_with_range_indices():
    """Test clipping with range indices."""
    # Arrange
    clip_indices = range(2, 8)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (6, 1, 12, 2)
            expected_data = points_data[2:8]
            np.testing.assert_array_equal(f["poseest/points"][:], expected_data)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_filters_invalid_frame_indices():
    """Test that invalid frame indices are filtered out without error."""
    # Arrange
    clip_indices = [0, 2, 15, 20, 4]  # 15 and 20 are out of range

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with 10 frames
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert - Only valid indices should be used
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (
                3,
                1,
                12,
                2,
            )  # Only 0, 2, 4 are valid
            expected_data = points_data[[0, 2, 4]]
            np.testing.assert_array_equal(f["poseest/points"][:], expected_data)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_empty_clip_indices():
    """Test handling of empty clip indices."""
    # Arrange
    clip_indices = np.array([], dtype=int)  # Ensure proper dtype for empty array

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (0, 1, 12, 2)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_all_invalid_indices():
    """Test handling when all indices are invalid."""
    # Arrange
    clip_indices = [15, 20, 25]  # All out of range for 10-frame file

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with 10 frames
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (0, 1, 12, 2)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_preserves_compression_settings():
    """Test that compression settings are preserved."""
    # Arrange
    clip_indices = [0, 1, 2]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with compressed data
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset(
                "points", data=points_data, compression="gzip", compression_opts=6
            )

            # Create compressed segmentation data
            seg_data = np.random.rand(10, 1, 2, 10, 2).astype(np.float32)
            poseest.create_dataset(
                "seg_data", data=seg_data, compression="gzip", compression_opts=9
            )

            poseest.attrs["version"] = [6, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            # Check that compression was preserved
            assert f["poseest/points"].compression == "gzip"
            assert f["poseest/points"].compression_opts == 6
            assert f["poseest/seg_data"].compression == "gzip"
            assert f["poseest/seg_data"].compression_opts == 9

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_file_without_static_objects():
    """Test handling of files without static objects."""
    # Arrange
    clip_indices = [0, 1, 2]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file without static objects
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert "poseest/points" in f
            assert "static_objects" not in f
            assert f["poseest/points"].shape == (3, 1, 12, 2)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_different_dataset_shapes():
    """Test handling of datasets with different shapes."""
    # Arrange
    clip_indices = [0, 2, 4]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with various dataset shapes
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            # Frame-based data (should be clipped)
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            confidence_data = np.random.rand(10, 1, 12).astype(np.float32)
            poseest.create_dataset("confidence", data=confidence_data)

            # Non-frame-based data (should be copied as-is)
            centers_data = np.random.rand(5, 64).astype(
                np.float32
            )  # Different first dimension
            poseest.create_dataset("instance_id_center", data=centers_data)

            poseest.attrs["version"] = [4, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            # Frame-based data should be clipped
            assert f["poseest/points"].shape == (3, 1, 12, 2)
            assert f["poseest/confidence"].shape == (3, 1, 12)

            # Non-frame-based data should be copied as-is
            assert f["poseest/instance_id_center"].shape == (5, 64)
            np.testing.assert_array_equal(
                f["poseest/instance_id_center"][:], centers_data
            )

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_preserves_all_attributes():
    """Test that all attributes are preserved correctly."""
    # Arrange
    clip_indices = [0, 1]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with various attributes
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)

            # Set various attributes
            poseest.attrs["version"] = [6, 0]
            poseest.attrs["cm_per_pixel"] = 0.125
            poseest.attrs["cm_per_pixel_source"] = "corner_detection"
            poseest["points"].attrs["config"] = "pose_config"
            poseest["points"].attrs["model"] = "pose_model"

            # Add static objects with attributes
            static_objects = f.create_group("static_objects")
            corners_data = np.random.rand(4, 2).astype(np.float32)
            static_objects.create_dataset("corners", data=corners_data)
            static_objects["corners"].attrs["config"] = "corner_config"
            static_objects["corners"].attrs["model"] = "corner_model"

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            # Check poseest group attributes
            assert f["poseest"].attrs["version"].tolist() == [6, 0]
            assert f["poseest"].attrs["cm_per_pixel"] == 0.125
            assert f["poseest"].attrs["cm_per_pixel_source"] == "corner_detection"

            # Check dataset attributes
            assert f["poseest/points"].attrs["config"] == "pose_config"
            assert f["poseest/points"].attrs["model"] == "pose_model"

            # Check static object attributes
            assert f["static_objects/corners"].attrs["config"] == "corner_config"
            assert f["static_objects/corners"].attrs["model"] == "corner_model"

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_pathlib_paths():
    """Test that function accepts pathlib.Path objects."""
    # Arrange
    clip_indices = [0, 1]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = Path(tmp_in_file.name)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = Path(tmp_out_file.name)

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (2, 1, 12, 2)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if file_path.exists():
                file_path.unlink()


def test_propagates_input_file_exceptions():
    """Test that input file exceptions are propagated correctly."""
    # Arrange
    in_pose_file = "nonexistent_input.h5"
    out_pose_file = "output.h5"
    clip_indices = [0, 1]

    # Act & Assert
    with pytest.raises(OSError):
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)


def test_propagates_output_file_exceptions():
    """Test that output file exceptions are propagated correctly."""
    # Arrange
    clip_indices = [0, 1]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Try to write to invalid output path
        out_pose_file = "/invalid/path/output.h5"

        # Act & Assert
        with pytest.raises(OSError):
            write_pose_clip(in_pose_file, out_pose_file, clip_indices)

    finally:
        if os.path.exists(in_pose_file):
            os.unlink(in_pose_file)


def test_handles_negative_indices():
    """Test handling of negative indices (should be filtered out)."""
    # Arrange
    clip_indices = [-1, 0, 1, 2]  # -1 should be filtered out

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert - Only valid indices should be used
        with h5py.File(out_pose_file, "r") as f:
            assert f["poseest/points"].shape == (
                3,
                1,
                12,
                2,
            )  # Only 0, 1, 2 are valid

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_duplicate_indices():
    """Test that duplicate indices raise an error due to HDF5 limitations."""
    # Arrange
    clip_indices = [0, 1, 1, 2, 2, 2]  # Duplicates not supported by HDF5

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act & Assert - Should raise TypeError due to HDF5 indexing restrictions
        with pytest.raises(TypeError):
            write_pose_clip(in_pose_file, out_pose_file, clip_indices)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_handles_out_of_order_indices():
    """Test that out-of-order indices raise an error due to HDF5 limitations."""
    # Arrange
    clip_indices = [2, 0, 1]  # Out of order not supported by HDF5

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(5, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act & Assert - Should raise TypeError due to HDF5 indexing restrictions
        with pytest.raises(TypeError):
            write_pose_clip(in_pose_file, out_pose_file, clip_indices)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


@pytest.mark.parametrize(
    "clip_indices",
    [
        [0, 1, 2],  # Simple sequence
        [0, 5, 9],  # Sparse selection
        range(0, 10, 2),  # Range with step
        np.array([1, 3, 5, 7]),  # Numpy array
    ],
)
def test_various_index_patterns(clip_indices):
    """Test various patterns of clip indices."""
    # Arrange
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")
            points_data = np.random.rand(10, 1, 12, 2).astype(np.float32)
            poseest.create_dataset("points", data=points_data)
            poseest.attrs["version"] = [3, 0]

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            expected_length = len(clip_indices)
            assert f["poseest/points"].shape == (expected_length, 1, 12, 2)

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_realistic_usage_pattern():
    """Test realistic usage pattern from video clipping workflow."""
    # Arrange - Simulate trimming first hour of a longer recording
    # Create smaller test data (full size would be too large for tests)
    test_frames = 1000
    test_clip_indices = range(0, 500)  # First half

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create input pose file with realistic structure
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")

            points_data = np.random.rand(test_frames, 1, 12, 2).astype(np.uint16)
            confidence_data = np.random.rand(test_frames, 1, 12).astype(np.float32)

            poseest.create_dataset("points", data=points_data)
            poseest.create_dataset("confidence", data=confidence_data)
            poseest.attrs["version"] = [3, 0]
            poseest.attrs["cm_per_pixel"] = 0.1
            poseest.attrs["cm_per_pixel_source"] = "corner_detection"

            # Add static objects
            static_objects = f.create_group("static_objects")
            corners_data = np.array(
                [[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32
            )
            static_objects.create_dataset("corners", data=corners_data)
            static_objects["corners"].attrs["config"] = "corner_detection_v1"
            static_objects["corners"].attrs["model"] = "corner_model_v1"

        # Act
        write_pose_clip(in_pose_file, out_pose_file, test_clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            # Check that clipping worked correctly
            assert f["poseest/points"].shape == (500, 1, 12, 2)
            assert f["poseest/confidence"].shape == (500, 1, 12)

            # Check that static objects were preserved
            assert f["static_objects/corners"].shape == (4, 2)
            np.testing.assert_array_equal(f["static_objects/corners"][:], corners_data)

            # Check that attributes were preserved
            assert f["poseest"].attrs["cm_per_pixel"] == 0.1
            assert f["poseest"].attrs["cm_per_pixel_source"] == "corner_detection"
            assert f["static_objects/corners"].attrs["config"] == "corner_detection_v1"

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


def test_comprehensive_pose_file_structure():
    """Test with comprehensive pose file structure including all possible fields."""
    # Arrange
    clip_indices = [0, 1, 2]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_in_file:
        in_pose_file = tmp_in_file.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out_file:
        out_pose_file = tmp_out_file.name

    try:
        # Create comprehensive pose file
        with h5py.File(in_pose_file, "w") as f:
            poseest = f.create_group("poseest")

            # Version 6 pose data with all fields
            frames = 10
            num_animals = 2

            # Frame-based data (should be clipped)
            poseest.create_dataset(
                "points",
                data=np.random.rand(frames, num_animals, 12, 2).astype(np.uint16),
            )
            poseest.create_dataset(
                "confidence",
                data=np.random.rand(frames, num_animals, 12).astype(np.float32),
            )
            poseest.create_dataset(
                "instance_count",
                data=np.random.randint(0, 3, frames).astype(np.uint8),
            )
            poseest.create_dataset(
                "instance_embedding",
                data=np.random.rand(frames, num_animals, 12).astype(np.float32),
            )
            poseest.create_dataset(
                "instance_track_id",
                data=np.random.randint(0, 10, (frames, num_animals)).astype(np.uint32),
            )
            poseest.create_dataset(
                "id_mask",
                data=np.random.choice([True, False], (frames, num_animals)),
            )
            poseest.create_dataset(
                "instance_embed_id",
                data=np.random.randint(0, 5, (frames, num_animals)).astype(np.uint32),
            )
            poseest.create_dataset(
                "identity_embeds",
                data=np.random.rand(frames, num_animals, 64).astype(np.float32),
            )
            poseest.create_dataset(
                "seg_data",
                data=np.random.rand(frames, num_animals, 2, 10, 2).astype(np.float32),
                compression="gzip",
                compression_opts=9,
            )
            poseest.create_dataset(
                "instance_seg_id",
                data=np.random.randint(0, 10, (frames, num_animals)).astype(np.uint32),
            )
            poseest.create_dataset(
                "longterm_seg_id",
                data=np.random.randint(0, 5, (frames, num_animals)).astype(np.uint32),
            )

            # Non-frame-based data (should be copied as-is)
            poseest.create_dataset(
                "instance_id_center", data=np.random.rand(5, 64).astype(np.float64)
            )

            # Set attributes
            poseest.attrs["version"] = [6, 0]
            poseest.attrs["cm_per_pixel"] = 0.08
            poseest.attrs["cm_per_pixel_source"] = "automated_calibration"

            # Add static objects
            static_objects = f.create_group("static_objects")
            static_objects.create_dataset(
                "corners", data=np.random.rand(4, 2).astype(np.float32)
            )
            static_objects.create_dataset(
                "lixit", data=np.random.rand(1, 2).astype(np.float32)
            )
            static_objects.create_dataset(
                "food_hopper", data=np.random.rand(2, 2).astype(np.float32)
            )

            # Set static object attributes
            static_objects["corners"].attrs["config"] = "corner_config"
            static_objects["corners"].attrs["model"] = "corner_model"
            static_objects["lixit"].attrs["config"] = "lixit_config"
            static_objects["lixit"].attrs["model"] = "lixit_model"

        # Act
        write_pose_clip(in_pose_file, out_pose_file, clip_indices)

        # Assert
        with h5py.File(out_pose_file, "r") as f:
            # Check all frame-based datasets were clipped
            frame_based_datasets = [
                "points",
                "confidence",
                "instance_count",
                "instance_embedding",
                "instance_track_id",
                "id_mask",
                "instance_embed_id",
                "identity_embeds",
                "seg_data",
                "instance_seg_id",
                "longterm_seg_id",
            ]

            for dataset_name in frame_based_datasets:
                dataset = f[f"poseest/{dataset_name}"]
                assert dataset.shape[0] == 3, (
                    f"Dataset {dataset_name} not clipped correctly"
                )

            # Check non-frame-based data was copied as-is
            assert f["poseest/instance_id_center"].shape == (5, 64)

            # Check static objects were copied
            assert f["static_objects/corners"].shape == (4, 2)
            assert f["static_objects/lixit"].shape == (1, 2)
            assert f["static_objects/food_hopper"].shape == (2, 2)

            # Check attributes were preserved
            assert f["poseest"].attrs["version"].tolist() == [6, 0]
            assert f["poseest"].attrs["cm_per_pixel"] == 0.08

            # Check compression was preserved
            assert f["poseest/seg_data"].compression == "gzip"
            assert f["poseest/seg_data"].compression_opts == 9

    finally:
        for file_path in [in_pose_file, out_pose_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)
