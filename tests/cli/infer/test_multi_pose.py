"""Unit tests for multi-pose Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.infer import app


class TestMultiPoseImplementation:
    """Test suite for multi-pose Typer implementation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.runner = CliRunner()
        self.test_video_path = Path("/tmp/test_video.mp4")
        self.test_frame_path = Path("/tmp/test_frame.jpg")
        self.test_output_path = Path("/tmp/output.json")
        self.test_video_output_path = Path("/tmp/output_video.mp4")

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
    def test_multi_pose_input_validation(self, video_arg, frame_arg, expected_success):
        """
        Test input validation for multi-pose implementation.

        Args:
            video_arg: Video argument flag or None
            frame_arg: Frame argument flag or None
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["multi-pose", "--out-file", str(self.test_output_path)]

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
                assert "Running PyTorch inference" in result.stdout
                assert "Multi-pose inference completed" in result.stdout
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout

    @pytest.mark.parametrize(
        "model_choice,runtime_choice,expected_success",
        [
            ("social-paper-topdown", "pytorch", True),
            ("invalid-model", "pytorch", False),
            ("social-paper-topdown", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    def test_multi_pose_choice_validation(
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
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
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
    def test_multi_pose_file_existence_validation(self, file_exists, expected_success):
        """
        Test file existence validation.

        Args:
            file_exists: Whether the input file should exist
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
        ]

        with patch("pathlib.Path.exists", return_value=file_exists):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            if expected_success:
                assert result.exit_code == 0
                assert "Running PyTorch inference" in result.stdout
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout

    def test_multi_pose_required_out_file(self):
        """Test that out-file parameter is required."""
        # Arrange
        cmd_args = ["multi-pose", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code != 0
            # Should fail because --out-file is missing

    @pytest.mark.parametrize(
        "out_video,expected_output",
        [
            (None, []),
            ("output_render.mp4", ["Output video: output_render.mp4"]),
        ],
        ids=["no_video_output", "with_video_output"],
    )
    def test_multi_pose_video_output_option(self, out_video, expected_output):
        """
        Test video output option functionality.

        Args:
            out_video: Output video path or None
            expected_output: Expected output messages
        """
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
        ]

        if out_video:
            cmd_args.extend(["--out-video", out_video])

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            for expected in expected_output:
                assert expected in result.stdout

    @pytest.mark.parametrize(
        "batch_size,expected_in_output",
        [
            (1, "Batch size: 1"),  # default
            (2, "Batch size: 2"),  # custom value
            (8, "Batch size: 8"),  # larger batch
            (16, "Batch size: 16"),  # even larger batch
        ],
        ids=["default_batch", "small_batch", "medium_batch", "large_batch"],
    )
    def test_multi_pose_batch_size_option(self, batch_size, expected_in_output):
        """
        Test batch size option.

        Args:
            batch_size: Batch size to test
            expected_in_output: Expected output message containing batch size
        """
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--batch-size",
            str(batch_size),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert expected_in_output in result.stdout

    def test_multi_pose_default_values(self):
        """Test that multi-pose uses the correct default values."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Model: social-paper-topdown" in result.stdout
            assert "Batch size: 1" in result.stdout
            assert "Running PyTorch inference" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_multi_pose_help_text(self):
        """Test that the multi-pose command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["multi-pose", "--help"])

        # Assert
        assert result.exit_code == 0
        assert "Run multi-pose inference" in result.stdout
        assert "Exactly one of --video or --frame must be specified" in result.stdout

    def test_multi_pose_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test case 1: Both video and frame specified
        result = self.runner.invoke(
            app,
            [
                "multi-pose",
                "--out-file",
                str(self.test_output_path),
                "--video",
                str(self.test_video_path),
                "--frame",
                str(self.test_frame_path),
            ],
        )
        assert result.exit_code == 1
        assert "Cannot specify both --video and --frame" in result.stdout

        # Test case 2: Neither video nor frame specified
        result = self.runner.invoke(
            app, ["multi-pose", "--out-file", str(self.test_output_path)]
        )
        assert result.exit_code == 1
        assert "Must specify either --video or --frame" in result.stdout

        # Test case 3: File doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(
                app,
                [
                    "multi-pose",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    str(self.test_video_path),
                ],
            )
            assert result.exit_code == 1
            assert "does not exist" in result.stdout

    def test_multi_pose_integration_flow(self):
        """Test the complete integration flow of multi-pose inference."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            "social-paper-topdown",
            "--runtime",
            "pytorch",
            "--out-video",
            str(self.test_video_output_path),
            "--batch-size",
            "4",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0

            # Verify all expected outputs are in the result
            expected_messages = [
                "Running PyTorch inference on video",
                "Model: social-paper-topdown",
                "Batch size: 4",
                f"Output file: {self.test_output_path}",
                f"Output video: {self.test_video_output_path}",
                "Multi-pose inference completed",
            ]

            for message in expected_messages:
                assert message in result.stdout

    def test_multi_pose_video_input_processing(self):
        """Test multi-pose specifically with video input."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference on video" in result.stdout
            assert str(self.test_video_path) in result.stdout

    def test_multi_pose_frame_input_processing(self):
        """Test multi-pose specifically with frame input."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--frame",
            str(self.test_frame_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference on frame" in result.stdout
            assert str(self.test_frame_path) in result.stdout

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
    def test_multi_pose_edge_case_paths(self, edge_case_path):
        """
        Test multi-pose with edge case file paths.

        Args:
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(
                app,
                [
                    "multi-pose",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    edge_case_path,
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference" in result.stdout

    def test_multi_pose_batch_size_edge_cases(self):
        """Test multi-pose with edge case batch sizes."""
        # Arrange & Act - very small batch size
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(
                app,
                [
                    "multi-pose",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    str(self.test_video_path),
                    "--batch-size",
                    "0",
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Batch size: 0" in result.stdout

        # Arrange & Act - large batch size
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(
                app,
                [
                    "multi-pose",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    str(self.test_video_path),
                    "--batch-size",
                    "64",
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Batch size: 64" in result.stdout

    def test_multi_pose_pytorch_runtime_specific(self):
        """Test multi-pose-specific functionality for PyTorch runtime."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            "multi_mouse_poses.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "social-paper-topdown",
            "--runtime",
            "pytorch",
            "--batch-size",
            "8",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference on video" in result.stdout
            assert "Model: social-paper-topdown" in result.stdout
            assert "Batch size: 8" in result.stdout
            assert "Output file: multi_mouse_poses.json" in result.stdout
            assert "Multi-pose inference completed" in result.stdout

    def test_multi_pose_minimal_configuration(self):
        """Test multi-pose with minimal required configuration."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--frame",
            str(self.test_frame_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference on frame" in result.stdout
            assert "Model: social-paper-topdown" in result.stdout  # default model
            assert "Batch size: 1" in result.stdout  # default batch size
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_multi_pose_maximum_configuration(self):
        """Test multi-pose with all possible options specified."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            "complete_pose_output.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "social-paper-topdown",
            "--runtime",
            "pytorch",
            "--out-video",
            "pose_visualization.mp4",
            "--batch-size",
            "16",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0

            # Verify all options are processed correctly
            expected_in_output = [
                "Running PyTorch inference on video",
                "Model: social-paper-topdown",
                "Batch size: 16",
                "Output file: complete_pose_output.json",
                "Output video: pose_visualization.mp4",
                "Multi-pose inference completed",
            ]

            for expected in expected_in_output:
                assert expected in result.stdout

    def test_multi_pose_topdown_model_specific(self):
        """Test multi-pose with the social-paper-topdown model specifically."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            "social-paper-topdown",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Model: social-paper-topdown" in result.stdout
            assert "Running PyTorch inference" in result.stdout
            assert "Multi-pose inference completed" in result.stdout

    def test_multi_pose_comparison_with_fecal_boli_batch_size(self):
        """Test that multi-pose batch-size works like fecal_boli but with different defaults."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--batch-size",
            "5",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Should have batch size like fecal_boli but different runtime
            assert "Batch size: 5" in result.stdout
            assert (
                "Running PyTorch inference" in result.stdout
            )  # pytorch, not pytorch like fecal_boli

    def test_multi_pose_simplified_output_options(self):
        """Test that multi-pose has simplified output options compared to other commands."""
        # This test ensures that multi-pose doesn't have the extra output options
        # that some other inference commands have

        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0

            # Verify it doesn't have frame count, interval, or image output
            assert "Frames:" not in result.stdout
            assert "Interval:" not in result.stdout
            assert "Output image:" not in result.stdout

            # But should have the basic functionality
            assert "Running PyTorch inference" in result.stdout
            assert "Model: social-paper-topdown" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout
            assert "Batch size: 1" in result.stdout
