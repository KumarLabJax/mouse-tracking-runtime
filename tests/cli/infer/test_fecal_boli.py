"""Unit tests for fecal boli Typer implementation."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch

from mouse_tracking_runtime.cli.infer import app


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
    def test_fecal_boli_input_validation(
        self, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for fecal boli implementation.
        
        Args:
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
                assert "Running PyTorch inference" in result.stdout
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout

    @pytest.mark.parametrize(
        "model_choice,runtime_choice,expected_success",
        [
            ("fecal-boli", "pytorch", True),
            ("invalid-model", "pytorch", False),
            ("fecal-boli", "invalid-runtime", False),
        ],
        ids=["valid_choices", "invalid_model", "invalid_runtime"],
    )
    def test_fecal_boli_choice_validation(
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
            "fecal-boli",
            "--video", str(self.test_video_path),
            "--model", model_choice,
            "--runtime", runtime_choice,
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
    def test_fecal_boli_file_existence_validation(self, file_exists, expected_success):
        """
        Test file existence validation.
        
        Args:
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
                assert "Running PyTorch inference" in result.stdout
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
                    "Output video: output.mp4"
                ]
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
    def test_fecal_boli_output_options(
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
            for expected_output in expected_outputs:
                assert expected_output in result.stdout

    @pytest.mark.parametrize(
        "frame_interval,batch_size,expected_in_output",
        [
            (1800, 1, "Frame interval: 1800, Batch size: 1"),  # defaults
            (3600, 2, "Frame interval: 3600, Batch size: 2"),  # custom values
            (1, 1, "Frame interval: 1, Batch size: 1"),  # minimal values
            (7200, 10, "Frame interval: 7200, Batch size: 10"),  # large values
        ],
        ids=["default_values", "custom_values", "minimal_values", "large_values"],
    )
    def test_fecal_boli_frame_interval_and_batch_size_options(
        self, frame_interval, batch_size, expected_in_output
    ):
        """
        Test frame interval and batch size options.
        
        Args:
            frame_interval: Frame interval to test
            batch_size: Batch size to test
            expected_in_output: Expected output message containing these values
        """
        # Arrange
        cmd_args = [
            "fecal-boli",
            "--video", str(self.test_video_path),
            "--frame-interval", str(frame_interval),
            "--batch-size", str(batch_size),
        ]
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)
            
            # Assert
            assert result.exit_code == 0
            assert expected_in_output in result.stdout

    def test_fecal_boli_default_values(self):
        """Test that fecal boli uses the correct default values."""
        # Arrange
        cmd_args = ["fecal-boli", "--video", str(self.test_video_path)]
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)
            
            # Assert
            assert result.exit_code == 0
            assert "Model: fecal-boli" in result.stdout
            assert "Frame interval: 1800, Batch size: 1" in result.stdout
            assert "Running PyTorch inference" in result.stdout

    def test_fecal_boli_help_text(self):
        """Test that the fecal boli command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["fecal-boli", "--help"])
        
        # Assert
        assert result.exit_code == 0
        assert "Run fecal boli inference" in result.stdout
        assert "Exactly one of --video or --frame must be specified" in result.stdout

    def test_fecal_boli_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test case 1: Both video and frame specified
        result = self.runner.invoke(app, [
            "fecal-boli",
            "--video", str(self.test_video_path),
            "--frame", str(self.test_frame_path)
        ])
        assert result.exit_code == 1
        assert "Cannot specify both --video and --frame" in result.stdout
        
        # Test case 2: Neither video nor frame specified
        result = self.runner.invoke(app, ["fecal-boli"])
        assert result.exit_code == 1
        assert "Must specify either --video or --frame" in result.stdout
        
        # Test case 3: File doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(app, [
                "fecal-boli",
                "--video", str(self.test_video_path)
            ])
            assert result.exit_code == 1
            assert "does not exist" in result.stdout

    def test_fecal_boli_integration_flow(self):
        """Test the complete integration flow of fecal boli inference."""
        # Arrange
        cmd_args = [
            "fecal-boli",
            "--video", str(self.test_video_path),
            "--model", "fecal-boli",
            "--runtime", "pytorch",
            "--out-file", "output.json",
            "--out-image", "output.png",
            "--out-video", "output.mp4",
            "--frame-interval", "3600",
            "--batch-size", "4",
        ]
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)
            
            # Assert
            assert result.exit_code == 0
            
            # Verify all expected outputs are in the result
            expected_messages = [
                "Running PyTorch inference on video",
                "Model: fecal-boli",
                "Frame interval: 3600, Batch size: 4",
                "Output file: output.json",
                "Output image: output.png",
                "Output video: output.mp4",
            ]
            
            for message in expected_messages:
                assert message in result.stdout

    def test_fecal_boli_video_input_processing(self):
        """Test fecal boli specifically with video input."""
        # Arrange
        cmd_args = ["fecal-boli", "--video", str(self.test_video_path)]
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)
            
            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference on video" in result.stdout
            assert str(self.test_video_path) in result.stdout

    def test_fecal_boli_frame_input_processing(self):
        """Test fecal boli specifically with frame input."""
        # Arrange
        cmd_args = ["fecal-boli", "--frame", str(self.test_frame_path)]
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)
            
            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference on frame" in result.stdout
            assert str(self.test_frame_path) in result.stdout

    def test_fecal_boli_args_compatibility_object(self):
        """Test that the InferenceArgs compatibility object is properly structured."""
        # Arrange
        cmd_args = [
            "fecal-boli",
            "--video", str(self.test_video_path),
            "--out-file", "test.json",
            "--batch-size", "3",
        ]
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)
            
            # Assert
            assert result.exit_code == 0
            # Verify that the output indicates proper args object creation
            assert "Running PyTorch inference on video" in result.stdout
            assert "Output file: test.json" in result.stdout
            assert "Frame interval: 1800, Batch size: 3" in result.stdout

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
    def test_fecal_boli_edge_case_paths(self, edge_case_path):
        """
        Test fecal boli with edge case file paths.
        
        Args:
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, [
                "fecal-boli",
                "--video", edge_case_path
            ])
            
            # Assert
            assert result.exit_code == 0
            assert "Running PyTorch inference" in result.stdout

    def test_fecal_boli_batch_size_edge_cases(self):
        """Test fecal boli with edge case batch sizes."""
        # Arrange & Act - very small batch size
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(app, [
                "fecal-boli",
                "--video", str(self.test_video_path),
                "--batch-size", "0"
            ])
            
            # Assert
            assert result.exit_code == 0
            assert "Batch size: 0" in result.stdout
        
        # Arrange & Act - large batch size
        with patch("pathlib.Path.exists", return_value=True):
            result = self.runner.invoke(app, [
                "fecal-boli",
                "--video", str(self.test_video_path),
                "--batch-size", "100"
            ])
            
            # Assert
            assert result.exit_code == 0
            assert "Batch size: 100" in result.stdout 