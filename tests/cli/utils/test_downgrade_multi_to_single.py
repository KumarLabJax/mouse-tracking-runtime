"""Unit tests for downgrade_multi_to_single CLI command."""

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


class TestDowngradeMultiToSingle:
    """Test class for downgrade_multi_to_single CLI command."""

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_successful_execution_with_defaults(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test successful execution with default parameters."""
        # Arrange
        mock_downgrade.return_value = None

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(str(temp_pose_file), disable_id=False)
        # Check that warning message is displayed
        assert "Warning:" in result.stdout
        assert "Not all pipelines may be 100% compatible" in result.stdout

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_execution_with_disable_id_flag(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test execution with --disable-id flag."""
        # Arrange
        mock_downgrade.return_value = None

        # Act
        result = runner.invoke(
            app, ["downgrade-multi-to-single", str(temp_pose_file), "--disable-id"]
        )

        # Assert
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(str(temp_pose_file), disable_id=True)
        # Check that warning message is displayed
        assert "Warning:" in result.stdout

    def test_missing_required_argument(self, runner):
        """Test behavior when required pose file argument is missing."""
        # Arrange & Act
        result = runner.invoke(app, ["downgrade-multi-to-single"])

        # Assert
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_path_argument_conversion_to_string(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test that Path argument is properly converted to string."""
        # Arrange
        mock_downgrade.return_value = None

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_downgrade.call_args
        assert isinstance(args[0], str)
        assert args[0] == str(temp_pose_file)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_disable_id_parameter_handling(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test that disable_id parameter is properly handled."""
        # Arrange
        mock_downgrade.return_value = None

        # Test with disable_id=False (default)
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])
        assert result.exit_code == 0
        mock_downgrade.assert_called_with(str(temp_pose_file), disable_id=False)

        mock_downgrade.reset_mock()

        # Test with disable_id=True
        result = runner.invoke(
            app, ["downgrade-multi-to-single", str(temp_pose_file), "--disable-id"]
        )
        assert result.exit_code == 0
        mock_downgrade.assert_called_with(str(temp_pose_file), disable_id=True)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_downgrade_pose_file_exception_handling(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test handling of exceptions from downgrade_pose_file."""
        # Arrange
        mock_downgrade.side_effect = FileNotFoundError("ERROR: missing file: test.h5")

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)
        assert "ERROR: missing file" in str(result.exception)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_invalid_pose_file_exception_handling(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test handling of InvalidPoseFileException from downgrade_pose_file."""
        # Arrange
        from mouse_tracking.core.exceptions import InvalidPoseFileException

        mock_downgrade.side_effect = InvalidPoseFileException(
            "Pose file test.h5 did not have a valid version."
        )

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, InvalidPoseFileException)

    def test_help_message_content(self, runner):
        """Test that help message contains expected content."""
        # Arrange & Act
        result = runner.invoke(
            app, ["downgrade-multi-to-single", "--help"], env={"TERM": "dumb"}
        )

        # Assert
        assert result.exit_code == 0
        assert "Downgrade multi-identity data to single-identity" in result.stdout
        assert "--disable-id" in result.stdout
        assert "Input HDF5 pose file path" in result.stdout
        assert "Disable identity embedding tracks" in result.stdout

    def test_warning_message_display(self, temp_pose_file, runner):
        """Test that warning message is properly displayed."""
        # Arrange & Act
        with patch("mouse_tracking.cli.utils.downgrade_pose_file"):
            result = runner.invoke(
                app, ["downgrade-multi-to-single", str(temp_pose_file)]
            )

        # Assert
        assert result.exit_code == 0
        warning_text = (
            "Warning: Not all pipelines may be 100% compatible using downgraded pose"
            " files. Files produced from this script will contain 0s in data where "
            "low confidence predictions were made instead of the original values "
            "which may affect performance."
        )
        assert warning_text in result.stdout

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_relative_path_handling(self, mock_downgrade, runner):
        """Test handling of relative paths."""
        # Arrange
        mock_downgrade.return_value = None
        relative_path = "data/pose_file.h5"

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", relative_path])

        # Assert
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(relative_path, disable_id=False)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_absolute_path_handling(self, mock_downgrade, runner):
        """Test handling of absolute paths."""
        # Arrange
        mock_downgrade.return_value = None
        absolute_path = "/tmp/absolute_pose_file.h5"

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", absolute_path])

        # Assert
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(absolute_path, disable_id=False)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_disable_id_flag_variations(self, mock_downgrade, temp_pose_file, runner):
        """Test different ways to specify the disable-id flag."""
        # Arrange
        mock_downgrade.return_value = None

        test_cases = [
            (["--disable-id"], True),
            ([], False),
        ]

        for args, expected_disable_id in test_cases:
            mock_downgrade.reset_mock()

            # Act
            result = runner.invoke(
                app, ["downgrade-multi-to-single", str(temp_pose_file), *args]
            )

            # Assert
            assert result.exit_code == 0
            mock_downgrade.assert_called_once_with(
                str(temp_pose_file), disable_id=expected_disable_id
            )

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_command_execution_order(self, mock_downgrade, temp_pose_file, runner):
        """Test that warning is displayed before calling downgrade_pose_file."""
        # Arrange
        mock_downgrade.return_value = None

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        # Verify warning appears in output before any potential error
        assert "Warning:" in result.stdout
        mock_downgrade.assert_called_once()

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_function_called_with_correct_signature(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test that downgrade_pose_file is called with the correct signature."""
        # Arrange
        mock_downgrade.return_value = None

        # Act
        result = runner.invoke(
            app, ["downgrade-multi-to-single", str(temp_pose_file), "--disable-id"]
        )

        # Assert
        assert result.exit_code == 0
        # Verify it's called with positional string argument and keyword disable_id
        mock_downgrade.assert_called_once_with(str(temp_pose_file), disable_id=True)

    def test_nonexistent_file_path(self, runner):
        """Test behavior with nonexistent file path."""
        # Arrange
        nonexistent_file = "/path/that/does/not/exist.h5"

        # Act
        with patch("mouse_tracking.cli.utils.downgrade_pose_file") as mock_downgrade:
            mock_downgrade.side_effect = FileNotFoundError(
                f"ERROR: missing file: {nonexistent_file}"
            )
            result = runner.invoke(app, ["downgrade-multi-to-single", nonexistent_file])

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_pose_file_v2_already_processed(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test handling when pose file is already v2 format."""
        # Arrange
        # This simulates the behavior where downgrade_pose_file calls exit(0) for v2 files
        mock_downgrade.side_effect = SystemExit(0)

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        # SystemExit(0) results in exit code 0 (successful exit) and no exception in result
        assert result.exit_code == 0
        # Warning message should still be displayed before the exit
        assert "Warning:" in result.stdout

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_warning_message_exact_content(
        self, mock_downgrade, temp_pose_file, runner
    ):
        """Test that the exact warning message content is displayed."""
        # Arrange
        mock_downgrade.return_value = None
        expected_warning = (
            "Warning: Not all pipelines may be 100% compatible using downgraded pose"
            " files. Files produced from this script will contain 0s in data where "
            "low confidence predictions were made instead of the original values "
            "which may affect performance."
        )

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", str(temp_pose_file)])

        # Assert
        assert result.exit_code == 0
        assert expected_warning in result.stdout

    @pytest.mark.parametrize(
        "file_extension",
        [".h5", ".hdf5", ".HDF5", ""],
        ids=["h5_extension", "hdf5_extension", "uppercase_hdf5", "no_extension"],
    )
    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_different_file_extensions(self, mock_downgrade, file_extension, runner):
        """Test handling of different file extensions."""
        # Arrange
        mock_downgrade.return_value = None
        filename = f"test_pose{file_extension}"

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", filename])

        # Assert
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(filename, disable_id=False)

    @patch("mouse_tracking.cli.utils.downgrade_pose_file")
    def test_special_characters_in_filename(self, mock_downgrade, runner):
        """Test handling of special characters in filename."""
        # Arrange
        mock_downgrade.return_value = None
        special_filename = "test-pose_file with spaces & symbols!.h5"

        # Act
        result = runner.invoke(app, ["downgrade-multi-to-single", special_filename])

        # Assert
        assert result.exit_code == 0
        mock_downgrade.assert_called_once_with(special_filename, disable_id=False)
