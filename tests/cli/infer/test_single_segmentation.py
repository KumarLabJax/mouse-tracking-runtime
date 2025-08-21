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
    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_input_validation(
        self, mock_infer, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for single-segmentation implementation.

        Args:
            mock_infer: Mock for the inference function
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
                mock_infer.assert_called_once()
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout
                mock_infer.assert_not_called()

    @pytest.mark.parametrize(
        "model_choice,runtime_choice,expected_success",
        [
            ("tracking-paper", "tfs", True),
            ("invalid-model", "tfs", False),
            ("tracking-paper", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_choice_validation(
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
    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_file_existence_validation(
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
                mock_infer.assert_called_once()
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout
                mock_infer.assert_not_called()

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
        "out_video",
        [None, "output_render.mp4"],
        ids=["no_video_output", "with_video_output"],
    )
    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_video_output_option(self, mock_infer, out_video):
        """
        Test video output option functionality.

        Args:
            mock_infer: Mock for the inference function
            out_video: Output video path or None
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            if out_video:
                assert args.out_video == out_video
            else:
                assert args.out_video is None

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_default_values(self, mock_infer):
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"
            assert args.runtime == "tfs"
            assert args.out_video is None

    def test_single_segmentation_help_text(self):
        """Test that the single-segmentation command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(
            app, ["single-segmentation", "--help"], env={"TERM": "dumb"}
        )

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

        # Test case 3: Input file doesn't exist
        def mock_exists_input_missing(path_self):
            return str(path_self) != str(self.test_video_path)  # Input doesn't exist

        with patch.object(Path, "exists", mock_exists_input_missing):
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

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_integration_flow(self, mock_infer):
        """Test complete integration flow with typical parameters."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"
            assert args.runtime == "tfs"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == str(self.test_output_path)
            assert args.out_video == str(self.test_video_output_path)

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_video_input_processing(self, mock_infer):
        """Test video input processing."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == str(self.test_video_path)
            assert args.frame is None

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_frame_input_processing(self, mock_infer):
        """Test frame input processing."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.frame == str(self.test_frame_path)
            assert args.video is None

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
    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_edge_case_paths(self, mock_infer, edge_case_path):
        """
        Test handling of edge case file paths.

        Args:
            mock_infer: Mock for the inference function
            edge_case_path: Path with special characters to test
        """
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
            "--video",
            edge_case_path,
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == edge_case_path

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_tracking_paper_model_specific(self, mock_infer):
        """Test tracking-paper model specific functionality."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_minimal_configuration(self, mock_infer):
        """Test minimal valid configuration."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"
            assert args.runtime == "tfs"
            assert args.out_video is None

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_maximum_configuration(self, mock_infer):
        """Test maximum configuration with all parameters."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"
            assert args.runtime == "tfs"
            assert args.video == str(self.test_video_path)
            assert args.out_file == str(self.test_output_path)
            assert args.out_video == str(self.test_video_output_path)

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_tfs_runtime_specific(self, mock_infer):
        """Test TFS runtime specific functionality."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.runtime == "tfs"

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_simplified_output_options(self, mock_infer):
        """Test simplified output options compared to other commands."""
        # Arrange - single-segmentation only has out-video, no out-image, no batch-size
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--out-video",
            str(self.test_video_output_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.out_video == str(self.test_video_output_path)
            # single-segmentation doesn't have out_image or batch_size parameters
            assert not hasattr(args, "out_image")
            assert not hasattr(args, "batch_size")

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_tracking_vs_gait_models(self, mock_infer):
        """Test that single-segmentation uses tracking-paper vs single-pose gait-paper model."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"
            # Different from single-pose which uses "gait-paper"

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_tfs_vs_pytorch_runtime(self, mock_infer):
        """Test that single-segmentation uses TFS vs pose functions that use PyTorch."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.runtime == "tfs"
            # Different from pose functions which use "pytorch"

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_no_batch_size_parameter(self, mock_infer):
        """Test that single-segmentation doesn't have batch-size parameter."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            # Verify batch_size parameter doesn't exist
            assert not hasattr(args, "batch_size")

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_no_frame_parameters(self, mock_infer):
        """Test that single-segmentation doesn't have frame-related parameters."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            # Verify frame-related parameters don't exist
            assert not hasattr(args, "num_frames")
            assert not hasattr(args, "frame_interval")

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_comparison_with_multi_identity(self, mock_infer):
        """Test that single-segmentation has similar structure to multi_identity but different models."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "tracking-paper"
            assert args.runtime == "tfs"
            # Both use TFS runtime but different models

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_segmentation_vs_pose_functionality(self, mock_infer):
        """Test that single-segmentation is different from pose functionality."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            # Segmentation uses TFS, pose uses PyTorch
            assert args.runtime == "tfs"
            # Segmentation uses tracking-paper, pose uses gait-paper or social-paper-topdown
            assert args.model == "tracking-paper"

    @patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs")
    def test_single_segmentation_args_compatibility_object(self, mock_infer):
        """Test that the args object has all required attributes for compatibility."""
        # Arrange
        cmd_args = [
            "single-segmentation",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--out-video",
            str(self.test_video_output_path),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            # Verify all expected attributes exist
            assert hasattr(args, "model")
            assert hasattr(args, "runtime")
            assert hasattr(args, "video")
            assert hasattr(args, "frame")
            assert hasattr(args, "out_file")
            assert hasattr(args, "out_video")

            # Verify values are correct
            assert args.model == "tracking-paper"
            assert args.runtime == "tfs"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == str(self.test_output_path)
            assert args.out_video == str(self.test_video_output_path)
