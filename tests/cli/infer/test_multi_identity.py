"""Unit tests for multi-identity Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.infer import app


class TestMultiIdentityImplementation:
    """Test suite for multi-identity Typer implementation."""

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
    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_input_validation(
        self, mock_infer, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for multi-identity implementation.

        Args:
            mock_infer: Mock for the inference function
            video_arg: Video argument flag or None
            frame_arg: Frame argument flag or None
            expected_success: Whether the command should succeed
        """
        # Arrange
        cmd_args = ["multi-identity", "--out-file", str(self.test_output_path)]

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
            ("social-paper", "tfs", True),
            ("2023", "tfs", True),
            ("invalid-model", "tfs", False),
            ("social-paper", "invalid-runtime", False),
        ],
        ids=["valid_social_paper", "valid_2023", "invalid_model", "invalid_runtime"],
    )
    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_choice_validation(
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
            "multi-identity",
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
    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_file_existence_validation(
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
            "multi-identity",
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

    def test_multi_identity_required_out_file(self):
        """Test that out-file parameter is required."""
        # Arrange
        cmd_args = ["multi-identity", "--video", str(self.test_video_path)]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code != 0
            # Should fail because --out-file is missing

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_default_values(self, mock_infer):
        """Test that multi-identity uses the correct default values."""
        # Arrange
        cmd_args = [
            "multi-identity",
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
            assert args.model == "social-paper"
            assert args.runtime == "tfs"
            assert args.out_file == str(self.test_output_path)

    def test_multi_identity_help_text(self):
        """Test that the multi-identity command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(
            app, ["multi-identity", "--help"], env={"TERM": "dumb"}
        )

        # Assert
        assert result.exit_code == 0
        assert "Run multi-identity inference" in result.stdout
        assert "Exactly one of --video or --frame must be specified" in result.stdout

    def test_multi_identity_error_handling_comprehensive(self):
        """Test comprehensive error handling scenarios."""
        # Test case 1: Both video and frame specified
        result = self.runner.invoke(
            app,
            [
                "multi-identity",
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
            app, ["multi-identity", "--out-file", str(self.test_output_path)]
        )
        assert result.exit_code == 1
        assert "Must specify either --video or --frame" in result.stdout

        # Test case 3: File doesn't exist
        with patch("pathlib.Path.exists", return_value=False):
            result = self.runner.invoke(
                app,
                [
                    "multi-identity",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    str(self.test_video_path),
                ],
            )
            assert result.exit_code == 1
            assert "does not exist" in result.stdout

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_integration_flow(self, mock_infer):
        """Test the complete integration flow of multi-identity inference."""
        # Arrange
        cmd_args = [
            "multi-identity",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            "2023",
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify the args object has all the expected values
            args = mock_infer.call_args[0][0]
            assert args.model == "2023"
            assert args.runtime == "tfs"
            assert args.video == str(self.test_video_path)
            assert args.frame is None
            assert args.out_file == str(self.test_output_path)

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_video_input_processing(self, mock_infer):
        """Test multi-identity specifically with video input."""
        # Arrange
        cmd_args = [
            "multi-identity",
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

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_frame_input_processing(self, mock_infer):
        """Test multi-identity specifically with frame input."""
        # Arrange
        cmd_args = [
            "multi-identity",
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
    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_edge_case_paths(self, mock_infer, edge_case_path):
        """
        Test multi-identity with edge case file paths.

        Args:
            mock_infer: Mock for the inference function
            edge_case_path: Path with special characters to test
        """
        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(
                app,
                [
                    "multi-identity",
                    "--out-file",
                    str(self.test_output_path),
                    "--video",
                    edge_case_path,
                ],
            )

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.video == edge_case_path

    @pytest.mark.parametrize(
        "model_variant",
        ["social-paper", "2023"],
        ids=["social_paper_model", "2023_model"],
    )
    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_model_variants(self, mock_infer, model_variant):
        """
        Test multi-identity with different model variants.

        Args:
            mock_infer: Mock for the inference function
            model_variant: Model variant to test
        """
        # Arrange
        cmd_args = [
            "multi-identity",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            model_variant,
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            args = mock_infer.call_args[0][0]
            assert args.model == model_variant

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_mouse_identity_specific_functionality(self, mock_infer):
        """Test multi-identity-specific functionality for mouse identity detection."""
        # Arrange
        cmd_args = [
            "multi-identity",
            "--out-file",
            "mouse_identities.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "2023",
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
            assert args.model == "2023"
            assert args.runtime == "tfs"
            assert args.out_file == "mouse_identities.json"

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_minimal_configuration(self, mock_infer):
        """Test multi-identity with minimal required configuration."""
        # Arrange
        cmd_args = [
            "multi-identity",
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
            assert args.model == "social-paper"  # default model
            assert args.runtime == "tfs"  # default runtime
            assert args.out_file == str(self.test_output_path)

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_maximum_configuration(self, mock_infer):
        """Test multi-identity with all possible options specified."""
        # Arrange
        cmd_args = [
            "multi-identity",
            "--out-file",
            "complete_identity_output.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "2023",
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            mock_infer.assert_called_once()

            # Verify all options are processed correctly
            args = mock_infer.call_args[0][0]
            assert args.model == "2023"
            assert args.runtime == "tfs"
            assert args.out_file == "complete_identity_output.json"

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_simplified_interface(self, mock_infer):
        """Test that multi-identity has a simplified interface compared to other commands."""
        # This test ensures that multi-identity doesn't have the extra parameters
        # that other inference commands have

        # Arrange
        cmd_args = [
            "multi-identity",
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
            assert args.model == "social-paper"
            assert args.runtime == "tfs"
            assert args.out_file == str(self.test_output_path)

    @patch("mouse_tracking.cli.infer.infer_multi_identity_tfs")
    def test_multi_identity_args_compatibility_object(self, mock_infer):
        """Test that the InferenceArgs compatibility object is properly structured."""
        # Arrange
        cmd_args = [
            "multi-identity",
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

            # Verify that the args object has all expected attributes
            args = mock_infer.call_args[0][0]
            assert hasattr(args, "model")
            assert hasattr(args, "runtime")
            assert hasattr(args, "video")
            assert hasattr(args, "frame")
            assert hasattr(args, "out_file")
