"""Unit tests for manual CLI command (clip video)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.utils import app, clip_video_app


@pytest.fixture
def runner():
    """Provide a CliRunner instance for testing."""
    return CliRunner()


@pytest.fixture
def temp_input_video():
    """Provide a temporary input video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        yield Path(temp_file.name)


@pytest.fixture
def temp_input_pose():
    """Provide a temporary input pose file for testing."""
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


@pytest.fixture
def temp_output_pose():
    """Provide a temporary output pose file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        output_path = Path(temp_file.name)
    # Remove the file so we can test creation
    output_path.unlink()
    yield output_path


class TestClipVideoManual:
    """Test class for manual CLI command within clip-video-to-start."""

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_successful_execution_with_defaults(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test successful execution with default parameters."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_clip_video.assert_called_once_with(
            str(temp_input_video),
            str(temp_input_pose),
            str(temp_output_video),
            str(temp_output_pose),
            1000,  # frame_start
            observation_duration=108000,  # 30 * 60 * 60
        )

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_execution_with_custom_parameters(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test execution with custom parameters."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "500",
                "--observation-duration",
                "54000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_clip_video.assert_called_once_with(
            str(temp_input_video),
            str(temp_input_pose),
            str(temp_output_video),
            str(temp_output_pose),
            500,  # frame_start
            observation_duration=54000,
        )

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_execution_with_allow_overwrite(
        self, mock_clip_video, temp_input_video, temp_input_pose, runner
    ):
        """Test execution with allow_overwrite when output files exist."""
        # Arrange
        mock_clip_video.return_value = None

        # Create existing output files
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            existing_output_video = Path(temp_video.name)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_pose:
            existing_output_pose = Path(temp_pose.name)

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(existing_output_video),
                "--out-pose",
                str(existing_output_pose),
                "--frame-start",
                "750",
                "--allow-overwrite",
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_clip_video.assert_called_once()

    def test_file_exists_error_without_allow_overwrite_video(
        self, temp_input_video, temp_input_pose, temp_output_pose, runner
    ):
        """Test FileExistsError when output video file exists and allow_overwrite is False."""
        # Arrange - Create existing output video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            existing_output_video = Path(temp_video.name)

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(existing_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "300",
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileExistsError)
        assert (
            "exists. If you wish to overwrite, please include --allow-overwrite"
            in str(result.exception)
        )

    def test_file_exists_error_without_allow_overwrite_pose(
        self, temp_input_video, temp_input_pose, temp_output_video, runner
    ):
        """Test FileExistsError when output pose file exists and allow_overwrite is False."""
        # Arrange - Create existing output pose file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_pose:
            existing_output_pose = Path(temp_pose.name)

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(existing_output_pose),
                "--frame-start",
                "600",
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileExistsError)
        assert (
            "exists. If you wish to overwrite, please include --allow-overwrite"
            in str(result.exception)
        )

    def test_missing_required_arguments(self, runner):
        """Test behavior when required arguments are missing."""
        # Arrange & Act
        result = runner.invoke(app, ["clip-video-to-start", "manual"])

        # Assert
        assert result.exit_code != 0
        assert "Missing option" in result.stdout

    def test_missing_frame_start_argument(
        self,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test behavior when required frame-start argument is missing."""
        # Arrange & Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert "Missing option" in result.stdout

    @pytest.mark.parametrize(
        "missing_option",
        ["--in-video", "--in-pose", "--out-video", "--out-pose"],
        ids=[
            "missing_in_video",
            "missing_in_pose",
            "missing_out_video",
            "missing_out_pose",
        ],
    )
    def test_individual_missing_required_arguments(
        self,
        missing_option,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test behavior when individual required arguments are missing."""
        # Arrange
        args = [
            "clip-video-to-start",
            "manual",
            "--in-video",
            str(temp_input_video),
            "--in-pose",
            str(temp_input_pose),
            "--out-video",
            str(temp_output_video),
            "--out-pose",
            str(temp_output_pose),
            "--frame-start",
            "1000",
        ]

        # Remove the missing option and its value
        option_index = args.index(missing_option)
        args.pop(option_index)  # Remove option
        args.pop(option_index)  # Remove value

        # Act
        result = runner.invoke(app, args)

        # Assert
        assert result.exit_code != 0
        assert "Missing option" in result.stdout

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_parameter_type_conversion(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test that parameters are properly converted to correct types."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "2500",
                "--observation-duration",
                "72000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_clip_video.call_args
        assert args[4] == 2500  # frame_start should be int
        assert kwargs["observation_duration"] == 72000  # Should be int

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_clip_video_manual_exception_handling(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test handling of exceptions from clip_video_manual."""
        # Arrange
        mock_clip_video.side_effect = ValueError("Invalid frame start")

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "Invalid frame start"

    def test_help_message_content(self, runner):
        """Test that help message contains expected content."""
        # Arrange & Act
        result = runner.invoke(
            app, ["clip-video-to-start", "manual", "--help"], env={"TERM": "dumb"}
        )

        # Assert
        assert result.exit_code == 0
        assert "Manually set the first frame" in result.stdout
        assert "--in-video" in result.stdout
        assert "--in-pose" in result.stdout
        assert "--out-video" in result.stdout
        assert "--out-pose" in result.stdout
        assert "--allow-overwrite" in result.stdout
        assert "--observation-duration" in result.stdout
        assert "--frame-start" in result.stdout

    @pytest.mark.parametrize(
        "frame_start,observation_duration,expected_frame_start,expected_duration",
        [
            ("0", "0", 0, 0),
            ("5000", "216000", 5000, 216000),
            ("-100", "54000", -100, 54000),  # Edge case: negative frame start
        ],
        ids=["zero_values", "large_values", "negative_frame_start"],
    )
    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_parameter_edge_cases(
        self,
        mock_clip_video,
        frame_start,
        observation_duration,
        expected_frame_start,
        expected_duration,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test edge cases for various parameters."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                frame_start,
                "--observation-duration",
                observation_duration,
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_clip_video.assert_called_once_with(
            str(temp_input_video),
            str(temp_input_pose),
            str(temp_output_video),
            str(temp_output_pose),
            expected_frame_start,
            observation_duration=expected_duration,
        )

    @pytest.mark.parametrize(
        "invalid_value,parameter",
        [
            ("invalid", "--frame-start"),
            ("1.5", "--observation-duration"),
            ("abc", "--frame-start"),
            ("not_an_int", "--observation-duration"),
        ],
        ids=[
            "invalid_frame_start",
            "float_observation_duration",
            "alphabetic_frame_start",
            "invalid_observation_duration",
        ],
    )
    def test_invalid_parameter_values(
        self,
        invalid_value,
        parameter,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test behavior with invalid parameter values."""
        # Arrange & Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000" if parameter != "--frame-start" else invalid_value,
                parameter,
                invalid_value,
            ]
            if parameter != "--frame-start"
            else [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                invalid_value,
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "invalid literal" in result.stdout

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_string_arguments_passed_correctly(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test that file paths are passed as strings to clip_video_manual."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_clip_video.call_args
        assert isinstance(args[0], str)  # in_video
        assert isinstance(args[1], str)  # in_pose
        assert isinstance(args[2], str)  # out_video
        assert isinstance(args[3], str)  # out_pose

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_allow_overwrite_false_by_default(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test that allow_overwrite defaults to False."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        # Verify that no file existence checks failed (which would happen if files existed and allow_overwrite was False)
        mock_clip_video.assert_called_once()

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_command_within_clip_video_app(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
    ):
        """Test that manual command can be called directly on clip_video_app."""
        # Arrange
        mock_clip_video.return_value = None
        runner = CliRunner()

        # Act
        result = runner.invoke(
            clip_video_app,
            [
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_clip_video.assert_called_once()

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_path_object_handling(self, mock_clip_video, runner):
        """Test that Path objects are properly handled in file existence checks."""
        # Arrange
        mock_clip_video.return_value = None

        # Create temp files that exist
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            in_video = Path(temp_video.name)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_pose:
            in_pose = Path(temp_pose.name)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_out_video:
            out_video = Path(temp_out_video.name)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_out_pose:
            out_pose = Path(temp_out_pose.name)

        # Act - This should trigger FileExistsError since output files exist and allow_overwrite is False
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(in_video),
                "--in-pose",
                str(in_pose),
                "--out-video",
                str(out_video),
                "--out-pose",
                str(out_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert isinstance(result.exception, FileExistsError)

    @patch("mouse_tracking.cli.utils.clip_video_manual")
    def test_observation_duration_default_value(
        self,
        mock_clip_video,
        temp_input_video,
        temp_input_pose,
        temp_output_video,
        temp_output_pose,
        runner,
    ):
        """Test that observation_duration uses correct default value."""
        # Arrange
        mock_clip_video.return_value = None

        # Act
        result = runner.invoke(
            app,
            [
                "clip-video-to-start",
                "manual",
                "--in-video",
                str(temp_input_video),
                "--in-pose",
                str(temp_input_pose),
                "--out-video",
                str(temp_output_video),
                "--out-pose",
                str(temp_output_pose),
                "--frame-start",
                "1000",
            ],
        )

        # Assert
        assert result.exit_code == 0
        args, kwargs = mock_clip_video.call_args
        assert kwargs["observation_duration"] == 108000  # 30 * 60 * 60
