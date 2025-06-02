"""Unit tests for multi-identity Typer implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking_runtime.cli.infer import app


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
    def test_multi_identity_input_validation(
        self, video_arg, frame_arg, expected_success
    ):
        """
        Test input validation for multi-identity implementation.

        Args:
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
                assert "Running TFS inference" in result.stdout
                assert "Multi-identity inference completed" in result.stdout
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout

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
    def test_multi_identity_choice_validation(
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
    def test_multi_identity_file_existence_validation(
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
                assert "Running TFS inference" in result.stdout
            else:
                assert result.exit_code == 1
                assert "does not exist" in result.stdout

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

    def test_multi_identity_default_values(self):
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
            assert "Model: social-paper" in result.stdout
            assert "Running TFS inference" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_multi_identity_help_text(self):
        """Test that the multi-identity command has proper help text."""
        # Arrange & Act
        result = self.runner.invoke(app, ["multi-identity", "--help"])

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

    def test_multi_identity_integration_flow(self):
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

            # Verify all expected outputs are in the result
            expected_messages = [
                "Running TFS inference on video",
                "Model: 2023",
                f"Output file: {self.test_output_path}",
                "Multi-identity inference completed",
            ]

            for message in expected_messages:
                assert message in result.stdout

    def test_multi_identity_video_input_processing(self):
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
            assert "Running TFS inference on video" in result.stdout
            assert str(self.test_video_path) in result.stdout

    def test_multi_identity_frame_input_processing(self):
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
            assert "Running TFS inference on frame" in result.stdout
            assert str(self.test_frame_path) in result.stdout

    def test_multi_identity_args_compatibility_object(self):
        """Test that the InferenceArgs compatibility object is properly structured."""
        # Arrange
        cmd_args = [
            "multi-identity",
            "--out-file",
            "test_identity.json",
            "--video",
            str(self.test_video_path),
            "--model",
            "social-paper",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Verify that the output indicates proper args object creation
            assert "Running TFS inference on video" in result.stdout
            assert "Output file: test_identity.json" in result.stdout
            assert "Model: social-paper" in result.stdout

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
    def test_multi_identity_edge_case_paths(self, edge_case_path):
        """
        Test multi-identity with edge case file paths.

        Args:
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
            assert "Running TFS inference" in result.stdout

    @pytest.mark.parametrize(
        "model_variant",
        ["social-paper", "2023"],
        ids=["social_paper_model", "2023_model"],
    )
    def test_multi_identity_model_variants(self, model_variant):
        """
        Test multi-identity with different model variants.

        Args:
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
            assert f"Model: {model_variant}" in result.stdout
            assert "Multi-identity inference completed" in result.stdout

    def test_multi_identity_mouse_identity_specific_functionality(self):
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
            assert "Running TFS inference on video" in result.stdout
            assert "Model: 2023" in result.stdout
            assert "Output file: mouse_identities.json" in result.stdout
            assert "Multi-identity inference completed" in result.stdout

    def test_multi_identity_minimal_configuration(self):
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
            assert "Running TFS inference on frame" in result.stdout
            assert "Model: social-paper" in result.stdout  # default model
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_multi_identity_maximum_configuration(self):
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

            # Verify all options are processed correctly
            expected_in_output = [
                "Running TFS inference on video",
                "Model: 2023",
                "Output file: complete_identity_output.json",
                "Multi-identity inference completed",
            ]

            for expected in expected_in_output:
                assert expected in result.stdout

    def test_multi_identity_simplified_interface(self):
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

            # Verify it's simpler - no frame count, interval, image/video outputs
            assert "Frames:" not in result.stdout
            assert "Interval:" not in result.stdout
            assert "Output image:" not in result.stdout
            assert "Output video:" not in result.stdout

            # But should have the basic functionality
            assert "Running TFS inference" in result.stdout
            assert "Model: social-paper" in result.stdout
            assert f"Output file: {self.test_output_path}" in result.stdout

    def test_multi_identity_comparison_with_other_commands(self):
        """Test that multi-identity maintains consistency with other inference commands."""
        # Arrange
        cmd_args = [
            "multi-identity",
            "--out-file",
            str(self.test_output_path),
            "--video",
            str(self.test_video_path),
            "--model",
            "social-paper",
            "--runtime",
            "tfs",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            # Act
            result = self.runner.invoke(app, cmd_args)

            # Assert
            assert result.exit_code == 0
            # Should use consistent patterns with other commands
            assert "Running TFS inference on video" in result.stdout
            assert "Model: social-paper" in result.stdout
