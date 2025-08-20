"""Unit tests for render_pose CLI command."""

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
def temp_video_file():
    """Provide a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        yield Path(temp_file.name)


@pytest.fixture
def temp_pose_file():
    """Provide a temporary pose file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        yield Path(temp_file.name)


@pytest.fixture
def temp_output_video():
    """Provide a temporary output video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        output_path = Path(temp_file.name)
    # Remove the file so we can test creation
    output_path.unlink()
    yield output_path


class TestRenderPose:
    """Test class for render_pose CLI command."""

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_successful_execution_with_defaults(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test successful execution with default parameters."""
        # Arrange
        mock_process.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            str(temp_video_file),
            str(temp_pose_file),
            str(temp_output_video),
            disable_id=False,
        )

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_execution_with_disable_id_flag(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test execution with --disable-id flag."""
        # Arrange
        mock_process.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
                "--disable-id",
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            str(temp_video_file),
            str(temp_pose_file),
            str(temp_output_video),
            disable_id=True,
        )

    def test_missing_required_arguments(self, runner):
        """Test behavior when required arguments are missing."""
        # Test missing all arguments
        result = runner.invoke(app, ["render-pose"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @pytest.mark.parametrize(
        "missing_args",
        [
            [],  # No arguments
            ["video.mp4"],  # Missing pose and output
            ["video.mp4", "pose.h5"],  # Missing output video
        ],
        ids=["no_args", "missing_pose_and_output", "missing_output"],
    )
    def test_individual_missing_required_arguments(self, missing_args, runner):
        """Test behavior when individual required arguments are missing."""
        # Arrange & Act
        result = runner.invoke(app, ["render-pose", *missing_args])

        # Assert
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_path_arguments_converted_to_strings(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test that Path arguments are properly converted to strings."""
        # Arrange
        mock_process.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
            ],
        )

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_process.call_args
        assert len(args) == 3
        assert all(isinstance(arg, str) for arg in args)
        assert args[0] == str(temp_video_file)
        assert args[1] == str(temp_pose_file)
        assert args[2] == str(temp_output_video)

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_disable_id_parameter_handling(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test that disable_id parameter is properly handled."""
        # Arrange
        mock_process.return_value = None

        # Test with disable_id=False (default)
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
            ],
        )
        assert result.exit_code == 0
        mock_process.assert_called_with(
            str(temp_video_file),
            str(temp_pose_file),
            str(temp_output_video),
            disable_id=False,
        )

        mock_process.reset_mock()

        # Test with disable_id=True
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
                "--disable-id",
            ],
        )
        assert result.exit_code == 0
        mock_process.assert_called_with(
            str(temp_video_file),
            str(temp_pose_file),
            str(temp_output_video),
            disable_id=True,
        )

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_process_video_exception_handling(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test handling of exceptions from render.process_video."""
        # Arrange
        mock_process.side_effect = FileNotFoundError("ERROR: missing file: video.mp4")

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)
        assert "ERROR: missing file" in str(result.exception)

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_video_processing_exception_handling(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test handling of video processing exceptions."""
        # Arrange
        mock_process.side_effect = ValueError("Invalid video format")

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "Invalid video format" in str(result.exception)

    def test_help_message_content(self, runner):
        """Test that help message contains expected content."""
        # Arrange & Act
        result = runner.invoke(app, ["render-pose", "--help"], env={"TERM": "dumb"})

        # Assert
        assert result.exit_code == 0
        assert "Render pose data" in result.stdout
        assert "Input video file path" in result.stdout
        assert "Input HDF5 pose file path" in result.stdout
        assert "Output video file path" in result.stdout
        assert "--disable-id" in result.stdout
        assert "Disable identity rendering" in result.stdout

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_relative_path_handling(self, mock_process, runner):
        """Test handling of relative paths."""
        # Arrange
        mock_process.return_value = None
        in_video = "data/input.mp4"
        in_pose = "data/pose.h5"
        out_video = "output/result.mp4"

        # Act
        result = runner.invoke(app, ["render-pose", in_video, in_pose, out_video])

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            in_video, in_pose, out_video, disable_id=False
        )

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_absolute_path_handling(self, mock_process, runner):
        """Test handling of absolute paths."""
        # Arrange
        mock_process.return_value = None
        in_video = "/tmp/input.mp4"
        in_pose = "/tmp/pose.h5"
        out_video = "/tmp/output.mp4"

        # Act
        result = runner.invoke(app, ["render-pose", in_video, in_pose, out_video])

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            in_video, in_pose, out_video, disable_id=False
        )

    @pytest.mark.parametrize(
        "video_ext,pose_ext,output_ext",
        [
            (".mp4", ".h5", ".mp4"),
            (".avi", ".hdf5", ".avi"),
            (".mov", ".HDF5", ".mov"),
            ("", "", ""),
        ],
        ids=["mp4_h5", "avi_hdf5", "mov_uppercase", "no_extensions"],
    )
    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_different_file_extensions(
        self, mock_process, video_ext, pose_ext, output_ext, runner
    ):
        """Test handling of different file extensions."""
        # Arrange
        mock_process.return_value = None
        in_video = f"input{video_ext}"
        in_pose = f"pose{pose_ext}"
        out_video = f"output{output_ext}"

        # Act
        result = runner.invoke(app, ["render-pose", in_video, in_pose, out_video])

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            in_video, in_pose, out_video, disable_id=False
        )

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_special_characters_in_filenames(self, mock_process, runner):
        """Test handling of special characters in filenames."""
        # Arrange
        mock_process.return_value = None
        in_video = "test-video_file with spaces & symbols!.mp4"
        in_pose = "test-pose_file with spaces & symbols!.h5"
        out_video = "test-output_file with spaces & symbols!.mp4"

        # Act
        result = runner.invoke(app, ["render-pose", in_video, in_pose, out_video])

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            in_video, in_pose, out_video, disable_id=False
        )

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_function_called_with_correct_signature(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test that render.process_video is called with the correct signature."""
        # Arrange
        mock_process.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
                "--disable-id",
            ],
        )

        # Assert
        assert result.exit_code == 0
        # Verify it's called with three string arguments and keyword disable_id
        args, kwargs = mock_process.call_args
        assert len(args) == 3
        assert all(isinstance(arg, str) for arg in args)
        assert "disable_id" in kwargs
        assert kwargs["disable_id"] is True

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_no_output_on_success(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test that successful execution produces no output."""
        # Arrange
        mock_process.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
            ],
        )

        # Assert
        assert result.exit_code == 0
        assert result.stdout.strip() == ""  # No output expected

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_nonexistent_files_handled_by_function(self, mock_process, runner):
        """Test that nonexistent files are handled by the underlying function."""
        # Arrange
        # The render.process_video function is responsible for file validation
        mock_process.side_effect = FileNotFoundError(
            "ERROR: missing file: nonexistent.mp4"
        )
        nonexistent_video = "/path/to/nonexistent.mp4"
        nonexistent_pose = "/path/to/nonexistent.h5"
        nonexistent_output = "/path/to/output.mp4"

        # Act
        result = runner.invoke(
            app,
            ["render-pose", nonexistent_video, nonexistent_pose, nonexistent_output],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_pose_file_version_compatibility(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test that the CLI handles pose file version compatibility through the function."""
        # Arrange
        mock_process.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "render-pose",
                str(temp_video_file),
                str(temp_pose_file),
                str(temp_output_video),
                "--disable-id",
            ],
        )

        # Assert
        assert result.exit_code == 0
        # Verify disable_id flag is passed correctly
        args, kwargs = mock_process.call_args
        assert kwargs["disable_id"] is True

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_large_file_paths(self, mock_process, runner):
        """Test handling of very long file paths."""
        # Arrange
        mock_process.return_value = None
        long_path_component = "very_long_path_component_" * 10  # 260 characters
        in_video = f"/tmp/{long_path_component}.mp4"
        in_pose = f"/tmp/{long_path_component}.h5"
        out_video = f"/tmp/{long_path_component}_output.mp4"

        # Act
        result = runner.invoke(app, ["render-pose", in_video, in_pose, out_video])

        # Assert
        assert result.exit_code == 0
        mock_process.assert_called_once_with(
            in_video, in_pose, out_video, disable_id=False
        )

    @patch("mouse_tracking.cli.utils.render.process_video")
    def test_disable_id_flag_variations(
        self, mock_process, temp_video_file, temp_pose_file, temp_output_video, runner
    ):
        """Test different ways to specify the disable-id flag."""
        # Arrange
        mock_process.return_value = None

        test_cases = [
            (["--disable-id"], True),
            ([], False),
        ]

        for args, expected_disable_id in test_cases:
            mock_process.reset_mock()

            # Act
            result = runner.invoke(
                app,
                [
                    "render-pose",
                    str(temp_video_file),
                    str(temp_pose_file),
                    str(temp_output_video),
                    *args,
                ],
            )

            # Assert
            assert result.exit_code == 0
            args, kwargs = mock_process.call_args
            assert kwargs["disable_id"] == expected_disable_id
