"""Unit tests for single-segmentation Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.infer import app


class TestSingleSegmentationImplementation:
    """Test suite for single-segmentation Typer implementation."""

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
    def test_single_segmentation_input_validation(
        self, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for single-segmentation implementation.

        Args:
            video_arg: Video argument flag or None
            frame_arg: Frame argument flag or None
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["single-segmentation", "--out-file", str(self.test_output_path)]

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
                assert "Single-segmentation inference completed" in result.stdout
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout

    @pytest.mark.parametrize(
        "model_choice,runtime_choice,expected_success",
        [
            ("tracking-paper", "tfs", True),
            ("invalid-model", "tfs", False),
            ("tracking-paper", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    def test_single_segmentation_choice_validation(
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
            "single-segmentation",
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
    def test_single_segmentation_file_existence_validation(
        self, file_exists, expected_success
    ):
        """
        Test file existence validation.

        Args:
            file_exists: Whether the input file should exist
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = [
            "single-segmentation",
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
                assert "Running TFS inference" in result.stdout
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout

    def test_single_segmentation_required_out_file(self):
        """Test that out-file parameter is required."""
        # Arrange
        cmd_args = ["single-segmentation", "--video", str(self.test_video_path)]

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
    def test_single_segmentation_video_output_option(self, out_video, expected_output):
        """
        Test video output option functionality.

        Args:
            out_video: Output video path or None
            expected_output: Expected output messages
        """
        # Arrange
        cmd_args = [
            "single-segmentation",
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

    def test_single_segmentation_default_values(self):
        """Test that single-segmentation uses the correct default values."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            assert "Model: tracking-paper" in result.stdout
            assert "Running TFS inference" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_single_segmentation_help_text(self):
        """Test that the single-segmentation command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["single-segmentation", "--help"])

        # Assert
        assert result.exit_code == 0
        assert "Run single-segmentation inference" in result.stdout
        assert "Exactly one of --video or --frame must be specified" in result.stdout

    def test_single_segmentation_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test case 1: Both video and frame specified
        result = self.runner.invoke(
            app,
            [
                "single-segmentation",
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
            app, ["single-segmentation", "--out-file", str(self.test_output_path)]
        )
        assert result.exit_code == 1
        assert "Must specify either --video or --frame" in result.stdout

        # Test case 3: File doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(
                app,
                [
                    "single-segmentation",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    str(self.test_video_path),
                ],
            )
            assert result.exit_code == 1
            assert "does not exist" in result.stdout

    def test_single_segmentation_integration_flow(self):
        """Test the complete integration flow of single-segmentation inference."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            "tracking-paper",
            "--runtime",
            "tfs",
            "--out-video",
            str(self.test_video_output_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0

            # Verify all expected outputs are in the result
            expected_messages = [
                "Running TFS inference on video",
                "Model: tracking-paper",
                f"Output file: {self.test_output_path}",
                f"Output video: {self.test_video_output_path}",
                "Single-segmentation inference completed",
            ]

            for message in expected_messages:
                assert message in result.stdout

    def test_single_segmentation_video_input_processing(self):
        """Test single-segmentation specifically with video input."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            assert "Running TFS inference on video" in result.stdout
            assert str(self.test_video_path) in result.stdout

    def test_single_segmentation_frame_input_processing(self):
        """Test single-segmentation specifically with frame input."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            assert "Running TFS inference on frame" in result.stdout
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
    def test_single_segmentation_edge_case_paths(self, edge_case_path):
        """
        Test single-segmentation with edge case file paths.

        Args:
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(
                app,
                [
                    "single-segmentation",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    edge_case_path,
                ],
            )

            # Assert
            assert result.exit_code == 0
            assert "Running TFS inference" in result.stdout

    def test_single_segmentation_tracking_paper_model_specific(self):
        """Test single-segmentation with the tracking-paper model specifically."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            "mouse_segmentation.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "tracking-paper",
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            assert "Running TFS inference on video" in result.stdout
            assert "Model: tracking-paper" in result.stdout
            assert "Output file: mouse_segmentation.json" in result.stdout
            assert "Single-segmentation inference completed" in result.stdout

    def test_single_segmentation_minimal_configuration(self):
        """Test single-segmentation with minimal required configuration."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            assert "Running TFS inference on frame" in result.stdout
            assert "Model: tracking-paper" in result.stdout  # default model
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_single_segmentation_maximum_configuration(self):
        """Test single-segmentation with all possible options specified."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            "complete_segmentation_output.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "tracking-paper",
            "--runtime",
            "tfs",
            "--out-video",
            "segmentation_visualization.mp4",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0

            # Verify all options are processed correctly
            expected_in_output = [
                "Running TFS inference on video",
                "Model: tracking-paper",
                "Output file: complete_segmentation_output.json",
                "Output video: segmentation_visualization.mp4",
                "Single-segmentation inference completed",
            ]

            for expected in expected_in_output:
                assert expected in result.stdout

    def test_single_segmentation_tfs_runtime_specific(self):
        """Test single-segmentation with TFS runtime specifically."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            "tracking-paper",
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Should use TFS runtime (different from pytorch-based commands)
            assert "Running TFS inference" in result.stdout
            assert "Model: tracking-paper" in result.stdout

    def test_single_segmentation_simplified_output_options(self):
        """Test that single-segmentation has simplified output options compared to some other commands."""
        # This test ensures that single-segmentation doesn't have the extra output options
        # that some other inference commands have

        # Arrange
        cmd_args = [
            "single-segmentation",
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

            # Verify it doesn't have frame count, interval, batch size, or image output
            assert "Frames:" not in result.stdout
            assert "Interval:" not in result.stdout
            assert "Batch size:" not in result.stdout
            assert "Output image:" not in result.stdout

            # But should have the basic functionality
            assert "Running TFS inference" in result.stdout
            assert "Model: tracking-paper" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_single_segmentation_tracking_vs_gait_models(self):
        """Test that single-segmentation uses tracking-paper model (different from single-pose gait-paper)."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            # Should use tracking-paper model (different from single-pose's gait-paper)
            assert "Model: tracking-paper" in result.stdout
            assert "gait-paper" not in result.stdout  # should not be single-pose model
            assert "Single-segmentation inference completed" in result.stdout

    def test_single_segmentation_tfs_vs_pytorch_runtime(self):
        """Test that single-segmentation uses TFS runtime (different from pose models using pytorch)."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Should use TFS runtime (different from pytorch-based pose commands)
            assert "Running TFS inference" in result.stdout
            assert "pytorch" not in result.stdout.lower()  # should not be pytorch
            assert "Model: tracking-paper" in result.stdout

    def test_single_segmentation_no_batch_size_parameter(self):
        """Test that single-segmentation doesn't have batch-size parameter like pose commands."""
        # Arrange - try to use batch-size option (should not be available)
        cmd_args = [
            "single-segmentation",
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
            # Should not have batch size functionality
            assert "Batch size" not in result.stdout
            assert "batch-size" not in result.stdout

            # But should have normal segmentation functionality
            assert "Running TFS inference" in result.stdout
            assert "Model: tracking-paper" in result.stdout

    def test_single_segmentation_no_frame_parameters(self):
        """Test that single-segmentation doesn't have frame count/interval parameters."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            # Should not have frame parameters
            assert "num-frames" not in result.stdout
            assert "frame-interval" not in result.stdout
            assert "Frames:" not in result.stdout
            assert "Interval:" not in result.stdout

            # But should have normal segmentation functionality
            assert "Running TFS inference" in result.stdout
            assert "Model: tracking-paper" in result.stdout

    def test_single_segmentation_comparison_with_multi_identity(self):
        """Test that single-segmentation has similar structure to multi-identity (required out-file, optional out-video)."""
        # Arrange
        cmd_args = [
            "single-segmentation",
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
            # Should have similar structure to multi-identity
            assert "Running TFS inference" in result.stdout
            assert "Model: tracking-paper" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout
            assert "Single-segmentation inference completed" in result.stdout

    def test_single_segmentation_segmentation_vs_pose_functionality(self):
        """Test that single-segmentation is clearly for segmentation (not pose detection)."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            "mouse_segments.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "tracking-paper",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Should be clearly for segmentation, not pose
            assert "Single-segmentation inference completed" in result.stdout
            assert "Model: tracking-paper" in result.stdout
            assert "Output file: mouse_segments.json" in result.stdout

            # Should not have pose-specific terminology
            assert "pose" not in result.stdout.lower()
            assert "keypoint" not in result.stdout.lower()
