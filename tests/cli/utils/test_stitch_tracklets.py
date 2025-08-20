"""Unit tests for stitch_tracklets CLI command."""

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


class TestStitchTracklets:
    """Test class for stitch_tracklets CLI command."""

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_successful_execution(self, mock_match, temp_pose_file, runner):
        """Test successful execution with required parameter."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        mock_match.assert_called_once_with(temp_pose_file)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_path_object_passed_correctly(self, mock_match, temp_pose_file, runner):
        """Test that Path object is passed correctly to match_predictions."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_match.call_args
        assert len(args) == 1
        assert isinstance(args[0], Path)
        assert args[0] == temp_pose_file

    def test_missing_required_argument(self, runner):
        """Test behavior when required pose file argument is missing."""
        # Arrange & Act
        result = runner.invoke(app, ["stitch-tracklets"])

        # Assert
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_match_predictions_exception_handling(
        self, mock_match, temp_pose_file, runner
    ):
        """Test handling of exceptions from match_predictions."""
        # Arrange
        mock_match.side_effect = ValueError("Invalid pose file format")

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "Invalid pose file format" in str(result.exception)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_file_not_found_exception_handling(self, mock_match, runner):
        """Test handling of FileNotFoundError from match_predictions."""
        # Arrange
        mock_match.side_effect = FileNotFoundError("No such file or directory")
        nonexistent_file = "/path/to/nonexistent/file.h5"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", nonexistent_file])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    def test_help_message_content(self, runner):
        """Test that help message contains expected content."""
        # Arrange & Act
        result = runner.invoke(
            app, ["stitch-tracklets", "--help"], env={"TERM": "dumb"}
        )

        # Assert
        assert result.exit_code == 0
        assert "Stitch tracklets" in result.stdout
        assert "Input HDF5 pose file" in result.stdout

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_relative_path_handling(self, mock_match, runner):
        """Test handling of relative paths."""
        # Arrange
        mock_match.return_value = None
        relative_path = "data/pose_file.h5"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", relative_path])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_match.call_args
        assert isinstance(args[0], Path)
        assert str(args[0]) == relative_path

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_absolute_path_handling(self, mock_match, runner):
        """Test handling of absolute paths."""
        # Arrange
        mock_match.return_value = None
        absolute_path = "/tmp/absolute_pose_file.h5"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", absolute_path])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_match.call_args
        assert isinstance(args[0], Path)
        assert str(args[0]) == absolute_path

    @pytest.mark.parametrize(
        "file_extension",
        [".h5", ".hdf5", ".HDF5", ""],
        ids=["h5_extension", "hdf5_extension", "uppercase_hdf5", "no_extension"],
    )
    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_different_file_extensions(self, mock_match, file_extension, runner):
        """Test handling of different file extensions."""
        # Arrange
        mock_match.return_value = None
        filename = f"test_pose{file_extension}"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", filename])

        # Assert
        assert result.exit_code == 0
        mock_match.assert_called_once()

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_special_characters_in_filename(self, mock_match, runner):
        """Test handling of special characters in filename."""
        # Arrange
        mock_match.return_value = None
        special_filename = "test-pose_file with spaces & symbols!.h5"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", special_filename])

        # Assert
        assert result.exit_code == 0
        mock_match.assert_called_once()

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_function_called_with_correct_signature(
        self, mock_match, temp_pose_file, runner
    ):
        """Test that match_predictions is called with the correct signature."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        # Verify it's called with one Path argument
        args, kwargs = mock_match.call_args
        assert len(args) == 1
        assert len(kwargs) == 0
        assert isinstance(args[0], Path)
        assert args[0] == temp_pose_file

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_no_output_on_success(self, mock_match, temp_pose_file, runner):
        """Test that successful execution produces no output."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        assert result.stdout.strip() == ""  # No output expected

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_pose_file_in_place_modification(self, mock_match, temp_pose_file, runner):
        """Test that the CLI correctly passes the pose file for in-place modification."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        # The function should be called with the pose file for in-place modification
        mock_match.assert_called_once_with(temp_pose_file)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_tracklet_processing_exception_handling(
        self, mock_match, temp_pose_file, runner
    ):
        """Test handling of tracklet processing exceptions."""
        # Arrange
        mock_match.side_effect = RuntimeError("Failed to process tracklets")

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
        assert "Failed to process tracklets" in str(result.exception)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_h5py_exception_handling(self, mock_match, temp_pose_file, runner):
        """Test handling of HDF5-related exceptions."""
        # Arrange
        mock_match.side_effect = OSError("Unable to open file")

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, OSError)
        assert "Unable to open file" in str(result.exception)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_memory_error_handling(self, mock_match, temp_pose_file, runner):
        """Test handling of memory errors during processing."""
        # Arrange
        mock_match.side_effect = MemoryError("Not enough memory")

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, MemoryError)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_large_file_path(self, mock_match, runner):
        """Test handling of very long file paths."""
        # Arrange
        mock_match.return_value = None
        long_path_component = "very_long_path_component_" * 10  # 260 characters
        long_path = f"/tmp/{long_path_component}.h5"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", long_path])

        # Assert
        assert result.exit_code == 0
        mock_match.assert_called_once()
        args, kwargs = mock_match.call_args
        assert str(args[0]) == long_path

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_unicode_filename(self, mock_match, runner):
        """Test handling of Unicode characters in filename."""
        # Arrange
        mock_match.return_value = None
        unicode_filename = "pose_测试_файл_🐁.h5"

        # Act
        result = runner.invoke(app, ["stitch-tracklets", unicode_filename])

        # Assert
        assert result.exit_code == 0
        mock_match.assert_called_once()

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_empty_filename_handling(self, mock_match, runner):
        """Test handling of empty filename."""
        # Arrange
        mock_match.return_value = None
        empty_filename = ""

        # Act
        result = runner.invoke(app, ["stitch-tracklets", empty_filename])

        # Assert
        assert result.exit_code == 0
        mock_match.assert_called_once()

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_pose_file_version_compatibility(self, mock_match, temp_pose_file, runner):
        """Test that the CLI handles different pose file versions through the function."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        # The match_predictions function should handle version compatibility
        mock_match.assert_called_once_with(temp_pose_file)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_concurrent_access_simulation(self, mock_match, temp_pose_file, runner):
        """Test behavior when file might be accessed concurrently."""
        # Arrange
        mock_match.side_effect = [OSError("Resource temporarily unavailable"), None]

        # Act - First call should fail, but test the interface
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, OSError)

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_no_options_available(self, mock_match, temp_pose_file, runner):
        """Test that stitch-tracklets command has no options (only required argument)."""
        # Arrange
        mock_match.return_value = None

        # Act
        result = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        # Verify no keyword arguments are passed
        args, kwargs = mock_match.call_args
        assert len(kwargs) == 0

    @patch("mouse_tracking.cli.utils.match_predictions")
    def test_command_idempotency(self, mock_match, temp_pose_file, runner):
        """Test that the command can be run multiple times on the same file."""
        # Arrange
        mock_match.return_value = None

        # Act - Run the command twice
        result1 = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])
        result2 = runner.invoke(app, ["stitch-tracklets", str(temp_pose_file)])

        # Assert
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert mock_match.call_count == 2
        # Both calls should use the same file
        for call in mock_match.call_args_list:
            args, kwargs = call
            assert args[0] == temp_pose_file
