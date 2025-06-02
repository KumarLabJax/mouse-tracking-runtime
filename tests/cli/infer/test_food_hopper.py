"""Unit tests for food hopper Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking_runtime.cli.infer import app


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
    def test_food_hopper_input_validation(self, video_arg, frame_arg, expected_success):
        """
        Test input validation for food hopper implementation.

        Args:
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
                assert "Running TFS inference" in result.stdout
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout

    @pytest.mark.parametrize(
        "model_choice,runtime_choice,expected_success",
        [
            ("social-2022-pipeline", "tfs", True),
            ("invalid-model", "tfs", False),
            ("social-2022-pipeline", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    def test_food_hopper_choice_validation(
        self, model_choice, runtime_choice, expected_success
    ):
        """
        Test model and runtime choice validation.

        Args:
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
                assert f"Model: {model_choice}" in result.stdout
            else:
                assert result.exit_code != 0

    @pytest.mark.parametrize(
        "file_exists,expected_success",
        [
            (True, True),
            (False, False),
        ],
        ids=["file_exists", "file_not_exists"],
    )
    def test_food_hopper_file_existence_validation(self, file_exists, expected_success):
        """
        Test file existence validation.

        Args:
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
                assert "Running TFS inference" in result.stdout
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout

    @pytest.mark.parametrize(
        "out_file,out_image,out_video,expected_outputs",
        [
            (None, None, None, []),
            ("output.json", None, None, ["Output file: output.json"]),
            (None, "output.png", None, ["Output image: output.png"]),
            (None, None, "output.mp4", ["Output video: output.mp4"]),
            (
                "output.json",
                "output.png",
                "output.mp4",
                [
                    "Output file: output.json",
                    "Output image: output.png",
                    "Output video: output.mp4",
                ],
            ),
        ],
        ids=[
            "no_outputs",
            "file_output_only",
            "image_output_only",
            "video_output_only",
            "all_outputs",
        ],
    )
    def test_food_hopper_output_options(
        self, out_file, out_image, out_video, expected_outputs
    ):
        """
        Test output options functionality.

        Args:
            out_file: Output file path or None
            out_image: Output image path or None
            out_video: Output video path or None
            expected_outputs: Expected output messages
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
            for expected_output in expected_outputs:
                assert expected_output in result.stdout

    @pytest.mark.parametrize(
        "num_frames,frame_interval,expected_in_output",
        [
            (100, 100, "Frames: 100, Interval: 100"),  # defaults
            (50, 10, "Frames: 50, Interval: 10"),  # custom values
            (1, 1, "Frames: 1, Interval: 1"),  # minimal values
            (1000, 500, "Frames: 1000, Interval: 500"),  # large values
        ],
        ids=["default_values", "custom_values", "minimal_values", "large_values"],
    )
    def test_food_hopper_frame_options(
        self, num_frames, frame_interval, expected_in_output
    ):
        """
        Test frame number and interval options.

        Args:
            num_frames: Number of frames to process
            frame_interval: Frame interval
            expected_in_output: Expected output message containing frame info
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
            assert expected_in_output in result.stdout

    def test_food_hopper_default_values(self):
        """Test that food hopper uses the correct default values."""
        # Arrange
        cmd_args = ["food-hopper", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Model: social-2022-pipeline" in result.stdout
            assert "Frames: 100, Interval: 100" in result.stdout
            assert "Running TFS inference" in result.stdout

    def test_food_hopper_help_text(self):
        """Test that the food hopper command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["food-hopper", "--help"])

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

    def test_food_hopper_integration_flow(self):
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

            # Verify all expected outputs are in the result
            expected_messages = [
                "Running TFS inference on video",
                "Model: social-2022-pipeline",
                "Frames: 25, Interval: 5",
                "Output file: output.json",
                "Output image: output.png",
                "Output video: output.mp4",
            ]

            for message in expected_messages:
                assert message in result.stdout

    def test_food_hopper_video_input_processing(self):
        """Test food hopper specifically with video input."""
        # Arrange
        cmd_args = ["food-hopper", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running TFS inference on video" in result.stdout
            assert str(self.test_video_path) in result.stdout

    def test_food_hopper_frame_input_processing(self):
        """Test food hopper specifically with frame input."""
        # Arrange
        cmd_args = ["food-hopper", "--frame", str(self.test_frame_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running TFS inference on frame" in result.stdout
            assert str(self.test_frame_path) in result.stdout

    def test_food_hopper_args_compatibility_object(self):
        """Test that the InferenceArgs compatibility object is properly structured."""
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--out-file",
            "test.json",
            "--num-frames",
            "75",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Verify that the output indicates proper args object creation
            assert "Running TFS inference on video" in result.stdout
            assert "Output file: test.json" in result.stdout
            assert "Frames: 75, Interval: 100" in result.stdout

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
    def test_food_hopper_edge_case_paths(self, edge_case_path):
        """
        Test food hopper with edge case file paths.

        Args:
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, ["food-hopper", "--video", edge_case_path])

            # Assert
            assert result.exit_code == 0
            assert "Running TFS inference" in result.stdout

    def test_food_hopper_frame_count_edge_cases(self):
        """Test food hopper with edge case frame counts."""
        # Arrange & Act - very small frame count
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(
                app,
                [
                    "food-hopper",
                    "--video",
                    str(self.test_video_path),
                    "--num-frames",
                    "1",
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Frames: 1, Interval: 100" in result.stdout

        # Arrange & Act - large frame count
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(
                app,
                [
                    "food-hopper",
                    "--video",
                    str(self.test_video_path),
                    "--num-frames",
                    "10000",
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Frames: 10000, Interval: 100" in result.stdout

    def test_food_hopper_comparison_with_arena_corner(self):
        """Test that food hopper has same parameter structure as arena corner."""
        # This test ensures consistency between similar commands
        # Arrange
        cmd_args = [
            "food-hopper",
            "--video",
            str(self.test_video_path),
            "--model",
            "social-2022-pipeline",
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Should use same model and runtime as arena_corner
            assert "Model: social-2022-pipeline" in result.stdout
            assert "Running TFS inference" in result.stdout

    def test_food_hopper_parameter_independence(self):
        """Test that num_frames and frame_interval work independently."""
        # Arrange & Act - only num_frames changed
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(
                app,
                [
                    "food-hopper",
                    "--video",
                    str(self.test_video_path),
                    "--num-frames",
                    "200",
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Frames: 200, Interval: 100" in result.stdout

        # Arrange & Act - only frame_interval changed
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(
                app,
                [
                    "food-hopper",
                    "--video",
                    str(self.test_video_path),
                    "--frame-interval",
                    "50",
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Frames: 100, Interval: 50" in result.stdout
