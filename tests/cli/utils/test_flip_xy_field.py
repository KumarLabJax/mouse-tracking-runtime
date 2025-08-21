"""Unit tests for flip_xy_field CLI command."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.utils import app


@pytest.fixture
def runner():
    """Provide a CliRunner instance for testing."""
    return CliRunner()


@pytest.fixture
def temp_pose_file():
    """Provide a temporary pose file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        yield Path(temp_file.name)


class TestFlipXyField:
    """Test class for flip_xy_field CLI command."""

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_successful_execution(self, mock_swap, temp_pose_file, runner):
        """Test successful execution with required parameters."""
        # Arrange
        mock_swap.return_value = None
        object_key = "arena_corners"

        # Act
        result = runner.invoke(app, ["flip-xy-field", str(temp_pose_file), object_key])

        # Assert
        assert result.exit_code == 0
        mock_swap.assert_called_once_with(temp_pose_file, object_key)

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_path_object_passed_correctly(self, mock_swap, temp_pose_file, runner):
        """Test that Path object is passed correctly to swap_static_obj_xy."""
        # Arrange
        mock_swap.return_value = None
        object_key = "food_hopper"

        # Act
        result = runner.invoke(app, ["flip-xy-field", str(temp_pose_file), object_key])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_swap.call_args
        assert isinstance(args[0], Path)
        assert args[0] == temp_pose_file
        assert args[1] == object_key

    def test_missing_required_arguments(self, runner):
        """Test behavior when required arguments are missing."""
        # Test missing both arguments
        result = runner.invoke(app, ["flip-xy-field"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    def test_missing_object_key_argument(self, temp_pose_file, runner):
        """Test behavior when object_key argument is missing."""
        # Arrange & Act
        result = runner.invoke(app, ["flip-xy-field", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_various_object_keys(self, mock_swap, temp_pose_file, runner):
        """Test with various object key names."""
        # Arrange
        mock_swap.return_value = None
        object_keys = [
            "arena_corners",
            "food_hopper",
            "lixit",
            "water_bottle",
            "custom_object",
            "object_with_underscores",
            "object123",
        ]

        for object_key in object_keys:
            mock_swap.reset_mock()

            # Act
            result = runner.invoke(
                app, ["flip-xy-field", str(temp_pose_file), object_key]
            )

            # Assert
            assert result.exit_code == 0
            mock_swap.assert_called_once_with(temp_pose_file, object_key)

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_swap_static_obj_xy_exception_handling(
        self, mock_swap, temp_pose_file, runner
    ):
        """Test handling of exceptions from swap_static_obj_xy."""
        # Arrange
        mock_swap.side_effect = OSError("Permission denied")
        object_key = "arena_corners"

        # Act
        result = runner.invoke(app, ["flip-xy-field", str(temp_pose_file), object_key])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, OSError)
        assert "Permission denied" in str(result.exception)

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_file_not_found_exception_handling(self, mock_swap, runner):
        """Test handling of FileNotFoundError from swap_static_obj_xy."""
        # Arrange
        mock_swap.side_effect = FileNotFoundError("No such file or directory")
        nonexistent_file = "/path/to/nonexistent/file.h5"
        object_key = "arena_corners"

        # Act
        result = runner.invoke(app, ["flip-xy-field", nonexistent_file, object_key])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_help_message_content(self, runner):
        """Test that help message contains expected content."""
        # Arrange & Act
        result = runner.invoke(app, ["flip-xy-field", "--help"], env={"TERM": "dumb"})

        # Assert
        assert result.exit_code == 0
        assert "Flip XY field" in result.stdout
        assert "Input HDF5 pose file" in result.stdout
        assert "Data key to swap the sorting" in result.stdout
        assert "[y, x] data to" in result.stdout
        assert "[x, y]" in result.stdout

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_relative_path_handling(self, mock_swap, runner):
        """Test handling of relative paths."""
        # Arrange
        mock_swap.return_value = None
        relative_path = "data/pose_file.h5"
        object_key = "lixit"

        # Act
        result = runner.invoke(app, ["flip-xy-field", relative_path, object_key])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_swap.call_args
        assert isinstance(args[0], Path)
        assert str(args[0]) == relative_path

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_absolute_path_handling(self, mock_swap, runner):
        """Test handling of absolute paths."""
        # Arrange
        mock_swap.return_value = None
        absolute_path = "/tmp/absolute_pose_file.h5"
        object_key = "food_hopper"

        # Act
        result = runner.invoke(app, ["flip-xy-field", absolute_path, object_key])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_swap.call_args
        assert isinstance(args[0], Path)
        assert str(args[0]) == absolute_path

    @pytest.mark.parametrize(
        "file_extension",
        [".h5", ".hdf5", ".HDF5", ""],
        ids=["h5_extension", "hdf5_extension", "uppercase_hdf5", "no_extension"],
    )
    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_different_file_extensions(self, mock_swap, file_extension, runner):
        """Test handling of different file extensions."""
        # Arrange
        mock_swap.return_value = None
        filename = f"test_pose{file_extension}"
        object_key = "arena_corners"

        # Act
        result = runner.invoke(app, ["flip-xy-field", filename, object_key])

        # Assert
        assert result.exit_code == 0
        mock_swap.assert_called_once()

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_special_characters_in_filename(self, mock_swap, runner):
        """Test handling of special characters in filename."""
        # Arrange
        mock_swap.return_value = None
        special_filename = "test-pose_file with spaces & symbols!.h5"
        object_key = "arena_corners"

        # Act
        result = runner.invoke(app, ["flip-xy-field", special_filename, object_key])

        # Assert
        assert result.exit_code == 0
        mock_swap.assert_called_once()

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_special_characters_in_object_key(self, mock_swap, temp_pose_file, runner):
        """Test handling of special characters in object key."""
        # Arrange
        mock_swap.return_value = None
        special_object_keys = [
            "object-with-dashes",
            "object_with_underscores",
            "object.with.dots",
            "object123",
            "UPPERCASE_OBJECT",
            "mixedCase_Object",
        ]

        for object_key in special_object_keys:
            mock_swap.reset_mock()

            # Act
            result = runner.invoke(
                app, ["flip-xy-field", str(temp_pose_file), object_key]
            )

            # Assert
            assert result.exit_code == 0
            mock_swap.assert_called_once_with(temp_pose_file, object_key)

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_nonexistent_object_key_no_error(self, mock_swap, temp_pose_file, runner):
        """Test that nonexistent object key doesn't cause CLI error (handled by swap function)."""
        # Arrange
        # The swap function prints a message but doesn't raise an exception for missing keys
        mock_swap.return_value = None  # Function returns None even for missing keys
        nonexistent_key = "nonexistent_object"

        # Act
        result = runner.invoke(
            app, ["flip-xy-field", str(temp_pose_file), nonexistent_key]
        )

        # Assert
        assert result.exit_code == 0  # CLI should still succeed
        mock_swap.assert_called_once_with(temp_pose_file, nonexistent_key)

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_function_called_with_correct_signature(
        self, mock_swap, temp_pose_file, runner
    ):
        """Test that swap_static_obj_xy is called with the correct signature."""
        # Arrange
        mock_swap.return_value = None
        object_key = "test_object"

        # Act
        result = runner.invoke(app, ["flip-xy-field", str(temp_pose_file), object_key])

        # Assert
        assert result.exit_code == 0
        # Verify it's called with Path object and string
        args, kwargs = mock_swap.call_args
        assert len(args) == 2
        assert isinstance(args[0], Path)
        assert isinstance(args[1], str)
        assert args[0] == temp_pose_file
        assert args[1] == object_key

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_no_output_on_success(self, mock_swap, temp_pose_file, runner):
        """Test that successful execution produces no output."""
        # Arrange
        mock_swap.return_value = None
        object_key = "arena_corners"

        # Act
        result = runner.invoke(app, ["flip-xy-field", str(temp_pose_file), object_key])

        # Assert
        assert result.exit_code == 0
        assert result.stdout.strip() == ""  # No output expected

    @pytest.mark.parametrize(
        "invalid_args",
        [
            [],  # No arguments
            ["only_filename.h5"],  # Missing object key
            [],  # Empty arguments list
        ],
        ids=["no_args", "missing_object_key", "empty_args"],
    )
    def test_invalid_argument_combinations(self, invalid_args, runner):
        """Test various invalid argument combinations."""
        # Arrange & Act
        result = runner.invoke(app, ["flip-xy-field", *invalid_args])

        # Assert
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_empty_object_key_string(self, mock_swap, temp_pose_file, runner):
        """Test handling of empty object key string."""
        # Arrange
        mock_swap.return_value = None
        empty_object_key = ""

        # Act
        result = runner.invoke(
            app, ["flip-xy-field", str(temp_pose_file), empty_object_key]
        )

        # Assert
        assert result.exit_code == 0
        mock_swap.assert_called_once_with(temp_pose_file, empty_object_key)

    @patch("mouse_tracking.cli.utils.static_objects.swap_static_obj_xy")
    def test_long_object_key_string(self, mock_swap, temp_pose_file, runner):
        """Test handling of very long object key string."""
        # Arrange
        mock_swap.return_value = None
        long_object_key = "very_long_object_key_" * 20  # 400 characters

        # Act
        result = runner.invoke(
            app, ["flip-xy-field", str(temp_pose_file), long_object_key]
        )

        # Assert
        assert result.exit_code == 0
        mock_swap.assert_called_once_with(temp_pose_file, long_object_key)
