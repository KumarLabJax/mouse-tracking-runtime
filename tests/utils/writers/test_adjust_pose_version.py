"""Comprehensive unit tests for the adjust_pose_version function."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from mouse_tracking.utils.writers import adjust_pose_version

from .mock_hdf5 import MockAttrs


class TestAdjustPoseVersionBasicFunctionality:
    """Test basic functionality of adjust_pose_version."""

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_with_promotion(
        self, mock_h5py_file, mock_promote_pose_data
    ):
        """Test adjusting pose version with data promotion enabled."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 4
        current_version = 2

        # Mock HDF5 file reading
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({"version": [current_version, 0]})
        mock_read_context.__getitem__.return_value = mock_poseest_group

        # Mock HDF5 file writing
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        # Setup file context manager behavior
        file_call_count = 0

        def mock_file_side_effect(filename, mode):
            nonlocal file_call_count
            file_call_count += 1
            mock_context = MagicMock()

            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context

            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should read the file to get current version
        assert any(call[0][1] == "r" for call in mock_h5py_file.call_args_list)

        # Should write the new version
        assert any(call[0][1] == "a" for call in mock_h5py_file.call_args_list)

        # Should call promote_pose_data since current_version < new_version
        mock_promote_pose_data.assert_called_once_with(
            pose_file, current_version, new_version
        )

        # Should set the version attribute correctly
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_without_promotion(
        self, mock_h5py_file, mock_promote_pose_data
    ):
        """Test adjusting pose version with data promotion disabled."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 5
        current_version = 3

        # Mock HDF5 file reading
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({"version": [current_version, 0]})
        mock_read_context.__getitem__.return_value = mock_poseest_group

        # Mock HDF5 file writing
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        # Setup file context manager behavior
        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=False)

        # Assert
        # Should NOT call promote_pose_data
        mock_promote_pose_data.assert_not_called()

        # Should still set the version attribute
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_same_version(self, mock_h5py_file):
        """Test adjusting pose version when current version equals new version."""
        # Arrange
        pose_file = "test_pose.h5"
        version = 4

        # Mock HDF5 file reading
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({"version": [version, 0]})
        mock_read_context.__getitem__.return_value = mock_poseest_group

        mock_h5py_file.return_value.__enter__.return_value = mock_read_context

        # Act
        adjust_pose_version(pose_file, version, promote_data=True)

        # Assert
        # Should only read the file once to check version
        mock_h5py_file.assert_called_once_with(pose_file, "r")

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_adjust_pose_version_downgrade_no_operation(self, mock_h5py_file):
        """Test adjusting pose version when current version is higher than new version."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 3
        current_version = 5

        # Mock HDF5 file reading
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({"version": [current_version, 0]})
        mock_read_context.__getitem__.return_value = mock_poseest_group

        mock_h5py_file.return_value.__enter__.return_value = mock_read_context

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should only read the file once to check version
        mock_h5py_file.assert_called_once_with(pose_file, "r")


class TestAdjustPoseVersionErrorHandling:
    """Test error handling in adjust_pose_version."""

    def test_invalid_version_too_low(self):
        """Test that ValueError is raised for version < 2."""
        # Arrange
        pose_file = "test_pose.h5"
        invalid_version = 1

        # Act & Assert
        with pytest.raises(
            ValueError, match="Pose version 1 not allowed. Please select between 2-6."
        ):
            adjust_pose_version(pose_file, invalid_version)

    def test_invalid_version_too_high(self):
        """Test that ValueError is raised for version > 6."""
        # Arrange
        pose_file = "test_pose.h5"
        invalid_version = 7

        # Act & Assert
        with pytest.raises(
            ValueError, match="Pose version 7 not allowed. Please select between 2-6."
        ):
            adjust_pose_version(pose_file, invalid_version)

    @pytest.mark.parametrize(
        "invalid_version",
        [0, 1, 7, 8, -1, 10],
        ids=[
            "version_0",
            "version_1",
            "version_7",
            "version_8",
            "negative_version",
            "version_10",
        ],
    )
    def test_invalid_version_range(self, invalid_version):
        """Test that ValueError is raised for any version outside 2-6 range."""
        # Arrange
        pose_file = "test_pose.h5"

        # Act & Assert
        with pytest.raises(
            ValueError, match=f"Pose version {invalid_version} not allowed"
        ):
            adjust_pose_version(pose_file, invalid_version)

    @pytest.mark.parametrize(
        "valid_version",
        [2, 3, 4, 5, 6],
        ids=["version_2", "version_3", "version_4", "version_5", "version_6"],
    )
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_valid_version_range(self, mock_h5py_file, valid_version):
        """Test that valid versions (2-6) don't raise ValueError."""
        # Arrange
        pose_file = "test_pose.h5"

        # Mock file with same version to avoid upgrade logic
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({"version": [valid_version, 0]})
        mock_read_context.__getitem__.return_value = mock_poseest_group
        mock_h5py_file.return_value.__enter__.return_value = mock_read_context

        # Act & Assert (should not raise)
        adjust_pose_version(pose_file, valid_version, promote_data=True)


class TestAdjustPoseVersionMissingVersion:
    """Test handling of missing version information."""

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_poseest_group(self, mock_h5py_file, mock_promote_pose_data):
        """Test handling when poseest group doesn't exist."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 4

        # Mock file context that raises KeyError for 'poseest'
        mock_read_context = MagicMock()
        mock_read_context.__getitem__.side_effect = KeyError("'poseest'")
        mock_read_context.__contains__.return_value = False
        mock_read_context.create_group = Mock()

        # Mock write context
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should create the poseest group
        mock_read_context.create_group.assert_called_once_with("poseest")

        # Should call promote_pose_data with current_version=-1
        mock_promote_pose_data.assert_called_once_with(pose_file, -1, new_version)

        # Should set version attribute
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_missing_version_attribute(self, mock_h5py_file, mock_promote_pose_data):
        """Test handling when version attribute doesn't exist."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 5

        # Mock poseest group without version attribute
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({})  # No version attribute
        mock_read_context.__getitem__.return_value = mock_poseest_group

        # Mock write context
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should call promote_pose_data with current_version=-1
        mock_promote_pose_data.assert_called_once_with(pose_file, -1, new_version)

        # Should set version attribute
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_malformed_version_attribute(self, mock_h5py_file, mock_promote_pose_data):
        """Test handling when version attribute has wrong shape."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 3

        # Mock poseest group with malformed version attribute (should raise IndexError)
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        # Create a MockAttrs that will raise IndexError when accessing index [0]
        malformed_attrs = MockAttrs({"version": []})  # Empty array causes IndexError
        mock_poseest_group.attrs = malformed_attrs
        mock_read_context.__getitem__.return_value = mock_poseest_group

        # Mock write context
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should call promote_pose_data with current_version=-1
        mock_promote_pose_data.assert_called_once_with(pose_file, -1, new_version)

        # Should set version attribute
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)


class TestAdjustPoseVersionIntegration:
    """Test integration scenarios for adjust_pose_version."""

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_full_version_upgrade_workflow(
        self, mock_h5py_file, mock_promote_pose_data
    ):
        """Test complete workflow of version upgrade from reading to writing."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 2
        new_version = 6

        # Mock read file context
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs(
            {"version": np.array([current_version, 0], dtype=np.uint16)}
        )
        mock_read_context.__getitem__.return_value = mock_poseest_group

        # Mock write file context
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        # Track file operations
        file_operations = []

        def mock_file_side_effect(filename, mode):
            file_operations.append((filename, mode))
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should have read and written to the file
        assert (pose_file, "r") in file_operations
        assert (pose_file, "a") in file_operations

        # Should call promote_pose_data
        mock_promote_pose_data.assert_called_once_with(
            pose_file, current_version, new_version
        )

        # Should set version correctly
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_version_already_current_no_changes(self, mock_h5py_file):
        """Test that no changes are made when version is already current."""
        # Arrange
        pose_file = "test_pose.h5"
        current_version = 4

        # Mock read context
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs({"version": [current_version, 0]})
        mock_read_context.__getitem__.return_value = mock_poseest_group

        mock_h5py_file.return_value.__enter__.return_value = mock_read_context

        # Act
        adjust_pose_version(pose_file, current_version, promote_data=True)

        # Assert
        # Should only read once, no writing should occur
        mock_h5py_file.assert_called_once_with(pose_file, "r")

    @pytest.mark.parametrize(
        "current_version,new_version,promote_data,should_promote",
        [
            (2, 3, True, True),  # Upgrade with promotion
            (2, 3, False, False),  # Upgrade without promotion
            (3, 3, True, False),  # Same version
            (4, 3, True, False),  # Downgrade (no operation)
            (2, 6, True, True),  # Large upgrade
            (5, 6, False, False),  # Small upgrade without promotion
        ],
        ids=[
            "upgrade_with_promotion",
            "upgrade_without_promotion",
            "same_version",
            "downgrade_no_op",
            "large_upgrade",
            "small_upgrade_no_promotion",
        ],
    )
    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_promotion_decision_matrix(
        self,
        mock_h5py_file,
        mock_promote_pose_data,
        current_version,
        new_version,
        promote_data,
        should_promote,
    ):
        """Test that promotion is called only under correct conditions."""
        # Arrange
        pose_file = "test_pose.h5"

        # Mock file context
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs(
            {"version": np.array([current_version, 0], dtype=np.uint16)}
        )
        mock_read_context.__getitem__.return_value = mock_poseest_group

        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=promote_data)

        # Assert
        if should_promote:
            mock_promote_pose_data.assert_called_once_with(
                pose_file, current_version, new_version
            )
        else:
            mock_promote_pose_data.assert_not_called()


class TestAdjustPoseVersionEdgeCases:
    """Test edge cases for adjust_pose_version."""

    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_version_attribute_different_dtype(self, mock_h5py_file):
        """Test handling version attribute with different data types."""
        # Arrange
        pose_file = "test_pose.h5"
        version = 4

        # Mock with version as different data type
        mock_read_context = MagicMock()
        mock_poseest_group = Mock()
        mock_poseest_group.attrs = MockAttrs(
            {"version": np.array([version], dtype=np.int32)}
        )  # Different dtype
        mock_read_context.__getitem__.return_value = mock_poseest_group

        mock_h5py_file.return_value.__enter__.return_value = mock_read_context

        # Act & Assert (should not raise)
        adjust_pose_version(pose_file, version, promote_data=True)

    @patch("mouse_tracking.utils.writers.promote_pose_data")
    @patch("mouse_tracking.utils.writers.h5py.File")
    def test_create_poseest_group_when_missing(
        self, mock_h5py_file, mock_promote_pose_data
    ):
        """Test that poseest group is created when missing."""
        # Arrange
        pose_file = "test_pose.h5"
        new_version = 3

        # Mock read context that raises KeyError and __contains__ returns False
        mock_read_context = MagicMock()
        mock_read_context.__getitem__.side_effect = KeyError("'poseest'")
        mock_read_context.__contains__.return_value = False
        mock_read_context.create_group = Mock()

        # Mock write context
        mock_write_context = MagicMock()
        mock_write_poseest_group = Mock()
        mock_write_poseest_group.attrs = MockAttrs()
        mock_write_context.__getitem__.return_value = mock_write_poseest_group

        def mock_file_side_effect(filename, mode):
            mock_context = MagicMock()
            if mode == "r":
                mock_context.__enter__.return_value = mock_read_context
            elif mode == "a":
                mock_context.__enter__.return_value = mock_write_context
            return mock_context

        mock_h5py_file.side_effect = mock_file_side_effect

        # Act
        adjust_pose_version(pose_file, new_version, promote_data=True)

        # Assert
        # Should create the poseest group
        mock_read_context.create_group.assert_called_once_with("poseest")

        # Should call promote_pose_data with current_version=-1
        mock_promote_pose_data.assert_called_once_with(pose_file, -1, new_version)

        # Should set version attribute
        expected_version_array = np.asarray([new_version, 0], dtype=np.uint16)
        actual_version = mock_write_poseest_group.attrs["version"]
        np.testing.assert_array_equal(actual_version, expected_version_array)
