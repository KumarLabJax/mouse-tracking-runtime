"""Unit tests for fecal boli Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.infer import app


class TestFecalBoliImplementation:
    """Test suite for fecal boli Typer implementation."""

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
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_input_validation(
        self, mock_infer, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for fecal boli implementation.

        Args:
            mock_infer: Mock for the inference function
            video_arg: Video argument flag or None
            frame_arg: Frame argument flag or None
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["fecal-boli"]

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
            ("fecal-boli", "pytorch", True),
            ("invalid-model", "pytorch", False),
            ("fecal-boli", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_choice_validation(
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
            "fecal-boli",
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
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_file_existence_validation(
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
        cmd_args = ["fecal-boli", "--video", str(self.test_video_path)]

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
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_output_options(
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
        cmd_args = ["fecal-boli", "--video", str(self.test_video_path)]

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
        "frame_interval,batch_size",
        [
            (1800, 1),  # defaults
            (3600, 2),  # custom values
            (1, 1),  # minimal values
            (7200, 10),  # large values
        ],
        ids=["default_values", "custom_values", "minimal_values", "large_values"],
    )
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_frame_interval_and_batch_size_options(
        self, mock_infer, frame_interval, batch_size
    ):
        """
        Test frame interval and batch size options.

        Args:
            mock_infer: Mock for the inference function
            frame_interval: Frame interval to test
            batch_size: Batch size to test
        """
        # Arrange
        cmd_args = [
            "fecal-boli",
            "--video",
            str(self.test_video_path),
            "--frame-interval",
            str(frame_interval),
            "--batch-size",
            str(batch_size),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify the args object contains the correct values
            args = mock_infer.call_args[0][0]
            assert args.frame_interval == frame_interval
            assert args.batch_size == batch_size

    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_default_values(self, mock_infer):
        """Test that fecal boli uses the correct default values."""
        # Arrange
        cmd_args = ["fecal-boli", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == "fecal-boli"
            assert args.runtime == "pytorch"
            assert args.frame_interval == 1800
            assert args.batch_size == 1
            assert args.out_file is None
            assert args.out_image is None
            assert args.out_video is None

    def test_fecal_boli_help_text(self):
        """Test that the fecal boli command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["fecal-boli", "--help"], env={"TERM": "dumb"})

        # Assert
        assert result.exit_code == 0
        assert "Run fecal boli inference" in result.stdout
        assert "Exactly one of --video or --frame must be specified" in result.stdout

    def test_fecal_boli_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test case 1: Both video and frame specified
        result = self.runner.invoke(
            app,
            [
                "fecal-boli",
                "--video",
                str(self.test_video_path),
                "--frame",
                str(self.test_frame_path),
            ],
        )
        assert result.exit_code == 1
        assert "Cannot specify both --video and --frame" in result.stdout

        # Test case 2: Neither video nor frame specified
        result = self.runner.invoke(app, ["fecal-boli"])
        assert result.exit_code == 1
        assert "Must specify either --video or --frame" in result.stdout

        # Test case 3: File doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(
                app, ["fecal-boli", "--video", str(self.test_video_path)]
            )
            assert result.exit_code == 1
            assert "does not exist" in result.stdout

    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_integration_flow(self, mock_infer):
        """Test the complete integration flow of fecal boli inference."""
        # Arrange
        cmd_args = [
            "fecal-boli",
            "--video",
            str(self.test_video_path),
            "--model",
            "fecal-boli",
            "--runtime",
            "pytorch",
            "--out-file",
            "output.json",
            "--out-image",
            "output.png",
            "--out-video",
            "output.mp4",
            "--frame-interval",
            "3600",
            "--batch-size",
            "4",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify the args object has all the expected values
            args = mock_infer.call_args[0][0]
            assert args.model == "fecal-boli"
            assert args.runtime == "pytorch"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == "output.json"
            assert args.out_image == "output.png"
            assert args.out_video == "output.mp4"
            assert args.frame_interval == 3600
            assert args.batch_size == 4

    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_video_input_processing(self, mock_infer):
        """Test fecal boli specifically with video input."""
        # Arrange
        cmd_args = ["fecal-boli", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == str(self.test_video_path)
            assert args.frame is None

    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_frame_input_processing(self, mock_infer):
        """Test fecal boli specifically with frame input."""
        # Arrange
        cmd_args = ["fecal-boli", "--frame", str(self.test_frame_path)]

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
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_edge_case_paths(self, mock_infer, edge_case_path):
        """
        Test fecal boli with edge case file paths.

        Args:
            mock_infer: Mock for the inference function
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, ["fecal-boli", "--video", edge_case_path])

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == edge_case_path

    @pytest.mark.parametrize(
        "batch_size",
        [0, 1, 2, 10, 100],
        ids=[
            "zero_batch",
            "minimal_batch",
            "small_batch",
            "medium_batch",
            "large_batch",
        ],
    )
    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_batch_size_edge_cases(self, mock_infer, batch_size):
        """Test fecal boli with edge case batch sizes."""
        # Arrange
        cmd_args = [
            "fecal-boli",
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

    @patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch")
    def test_fecal_boli_args_compatibility_object(self, mock_infer):
        """Test that the InferenceArgs compatibility object is properly structured."""
        # Arrange
        cmd_args = [
            "fecal-boli",
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
            assert hasattr(args, "frame_interval")
            assert hasattr(args, "batch_size")
