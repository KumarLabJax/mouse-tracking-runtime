"""Unit tests for food hopper Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.infer import app


class TestFoodHopperImplementation:
    """Test suite for food hopper Typer implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.runner = CliRunner()
        self.test_video_path = Path("/tmp/test_video.mp4")
        self.test_frame_path = Path("/tmp/test_frame.jpg")
        self.test_output_path = Path("/tmp/output.json")

    @pytest.mark.parametrize(
        "video_arg,frame_arg,expected_success",
        [
            ("--video", None, True),
            (None, "--frame", True),
            ("--video", "--frame", False),  # Both specified
            (None, None, False),  # Neither specified
        ],
        ids=[
            "video_only_success",
            "frame_only_success",
            "both_specified_error",
            "neither_specified_error",
        ],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_input_validation(
        self, mock_infer, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for food hopper implementation.

        Args:
            mock_infer: Mock for the inference function
            video_arg: Video argument flag or None
            frame_arg: Frame argument flag or None
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["food-hopper"]

        # Mock file existence for successful cases
        with patch("pathlib.Path.exists", return_value=True):
            if video_arg:
                cmd_args.extend([video_arg, str(self.test_video_path)])
            if frame_arg:
                cmd_args.extend([frame_arg, str(self.test_frame_path)])

            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            if expected_success:
                assert result.exit_code == 0
                mock_infer.assert_called_once()
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout
                mock_infer.assert_not_called()

    @pytest.mark.parametrize(
        "model_choice,runtime_choice,expected_success",
        [
            ("social-2022-pipeline", "tfs", True),
            ("invalid-model", "tfs", False),
            ("social-2022-pipeline", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_choice_validation(
        self, mock_infer, model_choice, runtime_choice, expected_success
    ):
        """
        Test model and runtime choice validation.

        Args:
            mock_infer: Mock for the inference function
            model_choice: Model choice to test
            runtime_choice: Runtime choice to test
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--model",
            model_choice,
            "--runtime",
            runtime_choice,
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            if expected_success:
                assert result.exit_code == 0
                mock_infer.assert_called_once()
                # Verify the args object passed to the inference function
                args = mock_infer.call_args[0][0]
                assert args.model == model_choice
                assert args.runtime == runtime_choice
            else:
                assert result.exit_code != 0
                mock_infer.assert_not_called()

    @pytest.mark.parametrize(
        "file_exists,expected_success",
        [
            (True, True),
            (False, False),
        ],
        ids=["file_exists", "file_not_exists"],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_file_existence_validation(
        self, mock_infer, file_exists, expected_success
    ):
        """
        Test file existence validation.

        Args:
            mock_infer: Mock for the inference function
            file_exists: Whether the input file should exist
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["food-hopper", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=file_exists):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            if expected_success:
                assert result.exit_code == 0
                mock_infer.assert_called_once()
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout
                mock_infer.assert_not_called()

    @pytest.mark.parametrize(
        "out_file,out_image,out_video",
        [
            (None, None, None),
            ("output.json", None, None),
            (None, "output.png", None),
            (None, None, "output.mp4"),
            ("output.json", "output.png", "output.mp4"),
        ],
        ids=[
            "no_outputs",
            "file_output_only",
            "image_output_only",
            "video_output_only",
            "all_outputs",
        ],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_output_options(
        self, mock_infer, out_file, out_image, out_video
    ):
        """
        Test output options functionality.

        Args:
            mock_infer: Mock for the inference function
            out_file: Output file path or None
            out_image: Output image path or None
            out_video: Output video path or None
        """
        # Arrange
        cmd_args = ["food-hopper", "--video", str(self.test_video_path)]

        if out_file:
            cmd_args.extend(["--out-file", out_file])
        if out_image:
            cmd_args.extend(["--out-image", out_image])
        if out_video:
            cmd_args.extend(["--out-video", out_video])

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify the args object contains the correct output paths
            args = mock_infer.call_args[0][0]
            assert args.out_file == out_file
            assert args.out_image == out_image
            assert args.out_video == out_video

    @pytest.mark.parametrize(
        "num_frames,frame_interval",
        [
            (100, 100),  # defaults
            (50, 10),  # custom values
            (1, 1),  # minimal values
            (1000, 500),  # large values
        ],
        ids=["default_values", "custom_values", "minimal_values", "large_values"],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_frame_options(self, mock_infer, num_frames, frame_interval):
        """
        Test frame number and interval options.

        Args:
            mock_infer: Mock for the inference function
            num_frames: Number of frames to process
            frame_interval: Frame interval
        """
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--num-frames",
            str(num_frames),
            "--frame-interval",
            str(frame_interval),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify the args object contains the correct frame options
            args = mock_infer.call_args[0][0]
            assert args.num_frames == num_frames
            assert args.frame_interval == frame_interval

    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_default_values(self, mock_infer):
        """Test that food hopper uses the correct default values."""
        # Arrange
        cmd_args = ["food-hopper", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "social-2022-pipeline"
            assert args.runtime == "tfs"
            assert args.num_frames == 100
            assert args.frame_interval == 100
            assert args.out_file is None
            assert args.out_image is None
            assert args.out_video is None

    def test_food_hopper_help_text(self):
        """Test that the food hopper command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(
            app, ["food-hopper", "--help"], env={"TERM": "dumb"}
        )

        # Assert
        assert result.exit_code == 0
        assert "Run food hopper inference" in result.stdout
        assert "Exactly one of --video or --frame must be specified" in result.stdout

    def test_food_hopper_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test case 1: Both video and frame specified
        result = self.runner.invoke(
            app,
            [
                "food-hopper",
                "--video",
                str(self.test_video_path),
                "--frame",
                str(self.test_frame_path),
            ],
        )
        assert result.exit_code == 1
        assert "Cannot specify both --video and --frame" in result.stdout

        # Test case 2: Neither video nor frame specified
        result = self.runner.invoke(app, ["food-hopper"])
        assert result.exit_code == 1
        assert "Must specify either --video or --frame" in result.stdout

        # Test case 3: File doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(
                app, ["food-hopper", "--video", str(self.test_video_path)]
            )
            assert result.exit_code == 1
            assert "does not exist" in result.stdout

    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_integration_flow(self, mock_infer):
        """Test the complete integration flow of food hopper inference."""
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--model",
            "social-2022-pipeline",
            "--runtime",
            "tfs",
            "--out-file",
            "output.json",
            "--out-image",
            "output.png",
            "--out-video",
            "output.mp4",
            "--num-frames",
            "25",
            "--frame-interval",
            "5",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify the args object has all the expected values
            args = mock_infer.call_args[0][0]
            assert args.model == "social-2022-pipeline"
            assert args.runtime == "tfs"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == "output.json"
            assert args.out_image == "output.png"
            assert args.out_video == "output.mp4"
            assert args.num_frames == 25
            assert args.frame_interval == 5

    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_video_input_processing(self, mock_infer):
        """Test food hopper specifically with video input."""
        # Arrange
        cmd_args = ["food-hopper", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == str(self.test_video_path)
            assert args.frame is None

    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_frame_input_processing(self, mock_infer):
        """Test food hopper specifically with frame input."""
        # Arrange
        cmd_args = ["food-hopper", "--frame", str(self.test_frame_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video is None
            assert args.frame == str(self.test_frame_path)

    @pytest.mark.parametrize(
        "edge_case_path",
        [
            "/path/with spaces/video.mp4",
            "/path/with-dashes/video.mp4",
            "/path/with_underscores/video.mp4",
            "/path/with.dots/video.mp4",
            "relative/path/video.mp4",
        ],
        ids=[
            "path_with_spaces",
            "path_with_dashes",
            "path_with_underscores",
            "path_with_dots",
            "relative_path",
        ],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_edge_case_paths(self, mock_infer, edge_case_path):
        """
        Test food hopper with edge case file paths.

        Args:
            mock_infer: Mock for the inference function
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, ["food-hopper", "--video", edge_case_path])

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == edge_case_path

    @pytest.mark.parametrize(
        "num_frames",
        [1, 10, 100, 1000, 10000],
        ids=[
            "minimal_frames",
            "small_frames",
            "default_frames",
            "large_frames",
            "huge_frames",
        ],
    )
    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_frame_count_edge_cases(self, mock_infer, num_frames):
        """Test food hopper with edge case frame counts."""
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--num-frames",
            str(num_frames),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.num_frames == num_frames

    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_parameter_independence(self, mock_infer):
        """Test that num_frames and frame_interval work independently."""
        # Arrange - only num_frames changed
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--num-frames",
            "200",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.num_frames == 200
            assert args.frame_interval == 100  # should be default

    @patch("mouse_tracking.cli.infer.infer_food_hopper_model")
    def test_food_hopper_args_compatibility_object(self, mock_infer):
        """Test that the InferenceArgs compatibility object is properly structured."""
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--out-file",
            "test.json",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify that the args object has all expected attributes
            args = mock_infer.call_args[0][0]
            assert hasattr(args, "model")
            assert hasattr(args, "runtime")
            assert hasattr(args, "video")
            assert hasattr(args, "frame")
            assert hasattr(args, "out_file")
            assert hasattr(args, "out_image")
            assert hasattr(args, "out_video")
            assert hasattr(args, "num_frames")
            assert hasattr(args, "frame_interval")
