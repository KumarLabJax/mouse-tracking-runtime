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
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_input_validation(
        self, mock_infer, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for multi-pose implementation.

        Args:
            mock_infer: Mock for the inference function
            video_arg: Video argument flag or None
            frame_arg: Frame argument flag or None
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["multi-pose", "--out-file", str(self.test_output_path)]

        # Mock file existence for successful cases (input and out-file must exist)
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
            ("social-paper-topdown", "pytorch", True),
            ("invalid-model", "pytorch", False),
            ("social-paper-topdown", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_choice_validation(
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
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_file_existence_validation(
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
                mock_infer.assert_called_once()
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout
                mock_infer.assert_not_called()

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

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_out_file_must_exist(self, mock_infer):
        """Test that out-file must already exist (contains segmentation data)."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
        ]

        def mock_exists(path_self):
            # Input video exists, but out-file doesn't exist
            return str(path_self) == str(self.test_video_path)

        with patch.object(Path, "exists", mock_exists):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 1
            assert "Pose file containing segmentation data is required" in result.stdout
            mock_infer.assert_not_called()

    @pytest.mark.parametrize(
        "out_video,batch_size",
        [
            (None, 1),  # No video output, default batch
            ("output_render.mp4", 1),  # With video output, default batch
            (None, 4),  # No video output, custom batch
            ("output_render.mp4", 8),  # With video output, custom batch
        ],
        ids=[
            "no_video_default_batch",
            "with_video_default_batch",
            "no_video_custom_batch",
            "with_video_custom_batch",
        ],
    )
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_optional_parameters(self, mock_infer, out_video, batch_size):
        """
        Test optional parameters functionality.

        Args:
            mock_infer: Mock for the inference function
            out_video: Output video path or None
            batch_size: Batch size to test
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
        if batch_size != 1:
            cmd_args.extend(["--batch-size", str(batch_size)])

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.batch_size == batch_size
            if out_video:
                assert args.out_video == out_video
            else:
                assert args.out_video is None

    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 8, 16],
        ids=["batch_1", "batch_2", "batch_8", "batch_16"],
    )
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_batch_size_validation(self, mock_infer, batch_size):
        """
        Test batch size validation.

        Args:
            mock_infer: Mock for the inference function
            batch_size: Batch size to test
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
            mock_infer.assert_called_once()
            args = mock_infer.call_args[0][0]
            assert args.batch_size == batch_size

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_default_values(self, mock_infer):
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "social-paper-topdown"
            assert args.runtime == "pytorch"
            assert args.batch_size == 1
            assert args.out_video is None

    def test_multi_pose_help_text(self):
        """Test that the multi-pose command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["multi-pose", "--help"], env={"TERM": "dumb"})

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

        # Test case 3: Input file doesn't exist
        def mock_exists_input_missing(path_self):
            return str(path_self) != str(self.test_video_path)  # Input doesn't exist

        with patch.object(Path, "exists", mock_exists_input_missing):
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

        # Test case 4: Out-file doesn't exist (special validation for multi-pose)
        def mock_exists_outfile_missing(path_self):
            return str(path_self) == str(self.test_video_path)  # Only input exists

        with patch.object(Path, "exists", mock_exists_outfile_missing):
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
            assert "Pose file containing segmentation data is required" in result.stdout

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_integration_flow(self, mock_infer):
        """Test complete integration flow with typical parameters."""
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
            "--batch-size",
            "4",
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
            assert args.model == "social-paper-topdown"
            assert args.runtime == "pytorch"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == str(self.test_output_path)
            assert args.out_video == str(self.test_video_output_path)
            assert args.batch_size == 4

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_video_input_processing(self, mock_infer):
        """Test video input processing."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == str(self.test_video_path)
            assert args.frame is None

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_frame_input_processing(self, mock_infer):
        """Test frame input processing."""
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
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_edge_case_paths(self, mock_infer, edge_case_path):
        """
        Test handling of edge case file paths.

        Args:
            mock_infer: Mock for the inference function
            edge_case_path: Path with special characters to test
        """
        # Arrange
        cmd_args = [
            "multi-pose",
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

    @pytest.mark.parametrize(
        "batch_size",
        [1, 2, 4, 8, 16, 32],
        ids=["batch_1", "batch_2", "batch_4", "batch_8", "batch_16", "batch_32"],
    )
    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_batch_size_edge_cases(self, mock_infer, batch_size):
        """
        Test various batch sizes including edge cases.

        Args:
            mock_infer: Mock for the inference function
            batch_size: Batch size to test
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.batch_size == batch_size

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_pytorch_runtime_specific(self, mock_infer):
        """Test PyTorch runtime specific functionality."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.runtime == "pytorch"
            assert args.batch_size == 8

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_minimal_configuration(self, mock_infer):
        """Test minimal valid configuration."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "social-paper-topdown"
            assert args.runtime == "pytorch"
            assert args.batch_size == 1
            assert args.out_video is None

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_maximum_configuration(self, mock_infer):
        """Test maximum configuration with all parameters."""
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
            "--batch-size",
            "16",
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
            assert args.model == "social-paper-topdown"
            assert args.runtime == "pytorch"
            assert args.video == str(self.test_video_path)
            assert args.out_file == str(self.test_output_path)
            assert args.out_video == str(self.test_video_output_path)
            assert args.batch_size == 16

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_topdown_model_specific(self, mock_infer):
        """Test social-paper-topdown model specific functionality."""
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
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "social-paper-topdown"

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_comparison_with_single_pose_batch_size(self, mock_infer):
        """Test that multi-pose can use same batch sizes as single-pose."""
        # Arrange - Test that multi-pose supports similar batch sizes to single-pose
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--batch-size",
            "4",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.batch_size == 4

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_simplified_output_options(self, mock_infer):
        """Test simplified output options compared to other commands."""
        # Arrange - multi-pose only has out-video, no out-image
        cmd_args = [
            "multi-pose",
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
            # multi-pose doesn't have out_image parameter
            assert not hasattr(args, "out_image")

    @patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch")
    def test_multi_pose_args_compatibility_object(self, mock_infer):
        """Test that the args object has all required attributes for compatibility."""
        # Arrange
        cmd_args = [
            "multi-pose",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--batch-size",
            "2",
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
            assert hasattr(args, "batch_size")

            # Verify values are correct
            assert args.model == "social-paper-topdown"
            assert args.runtime == "pytorch"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == str(self.test_output_path)
            assert args.out_video == str(self.test_video_output_path)
            assert args.batch_size == 2
