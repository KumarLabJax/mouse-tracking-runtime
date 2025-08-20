"""Tests for inference command registration and basic functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.infer import app


def test_infer_app_is_typer_instance():
    """Test that the infer app is a proper Typer instance."""
    # Arrange & Act
    import typer

    # Assert
    assert isinstance(app, typer.Typer)


def test_infer_app_has_commands():
    """Test that the infer app has registered commands."""
    # Arrange & Act
    commands = app.registered_commands

    # Assert
    assert len(commands) > 0
    assert isinstance(commands, list)


@pytest.mark.parametrize(
    "command_name,expected_docstring",
    [
        ("arena-corner", "Infer arena corner detection model."),
        ("fecal-boli", "Run fecal boli inference."),
        ("food-hopper", "Run food hopper inference."),
        ("lixit", "Run lixit inference."),
        ("multi-identity", "Run multi-identity inference."),
        ("multi-pose", "Run multi-pose inference."),
        ("single-pose", "Run single-pose inference."),
        ("single-segmentation", "Run single-segmentation inference."),
        ("multi-segmentation", "Run multi-segmentation inference."),
    ],
    ids=[
        "arena_corner_command",
        "fecal_boli_command",
        "food_hopper_command",
        "lixit_command",
        "multi_identity_command",
        "multi_pose_command",
        "single_pose_command",
        "single_segmentation_command",
        "multi_segmentation_command",
    ],
)
def test_infer_commands_registered(command_name, expected_docstring):
    """Test that all expected inference commands are registered with correct docstrings."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert expected_docstring in result.stdout


def test_infer_commands_list():
    """Test that all expected inference commands are registered."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--help"])

    # Assert
    assert result.exit_code == 0
    expected_commands = [
        "arena-corner",
        "fecal-boli",
        "food-hopper",
        "lixit",
        "multi-identity",
        "multi-pose",
        "single-pose",
        "single-segmentation",
        "multi-segmentation",
    ]

    for command in expected_commands:
        assert command in result.stdout


def test_infer_commands_help_structure():
    """Test that inference commands have consistent help structure."""
    # Arrange
    runner = CliRunner()
    commands = [
        "arena-corner",
        "fecal-boli",
        "food-hopper",
        "lixit",
        "multi-identity",
        "multi-pose",
        "single-pose",
        "single-segmentation",
        "multi-segmentation",
    ]

    # Act & Assert
    for command in commands:
        result = runner.invoke(app, [command, "--help"], env={"TERM": "dumb"})
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--help" in result.stdout


def test_infer_invalid_command():
    """Test that invalid inference commands show appropriate error."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["invalid-command"])

    # Assert
    assert result.exit_code != 0
    assert "No such command" in result.stdout or "Usage:" in result.stdout


def test_infer_app_without_arguments():
    """Test infer app behavior when called without arguments."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [])

    # Assert
    # When no command is provided, typer shows help and exits with code 0
    # 2 is also acceptable for missing required command
    assert result.exit_code == 0 or result.exit_code == 2
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "command_function_name",
    [
        "arena_corner",
        "fecal_boli",
        "food_hopper",
        "lixit",
        "multi_identity",
        "multi_pose",
        "single_pose",
        "single_segmentation",
        "multi_segmentation",
    ],
    ids=[
        "arena_corner_function",
        "fecal_boli_function",
        "food_hopper_function",
        "lixit_function",
        "multi_identity_function",
        "multi_pose_function",
        "single_pose_function",
        "single_segmentation_function",
        "multi_segmentation_function",
    ],
)
def test_infer_command_functions_exist(command_function_name):
    """Test that all inference command functions exist in the module."""
    # Arrange & Act
    from mouse_tracking.cli import infer

    # Assert
    assert hasattr(infer, command_function_name)
    assert callable(getattr(infer, command_function_name))


@pytest.mark.parametrize(
    "command_function_name,expected_docstring_content",
    [
        ("arena_corner", "arena corner detection"),
        ("fecal_boli", "fecal boli inference"),
        ("food_hopper", "food hopper inference"),
        ("lixit", "lixit inference"),
        ("multi_identity", "multi-identity inference"),
        ("multi_pose", "multi-pose inference"),
        ("single_pose", "single-pose inference"),
        ("single_segmentation", "single-segmentation inference"),
        ("multi_segmentation", "multi-segmentation inference"),
    ],
    ids=[
        "arena_corner_docstring",
        "fecal_boli_docstring",
        "food_hopper_docstring",
        "lixit_docstring",
        "multi_identity_docstring",
        "multi_pose_docstring",
        "single_pose_docstring",
        "single_segmentation_docstring",
        "multi_segmentation_docstring",
    ],
)
def test_infer_command_function_docstrings(
    command_function_name, expected_docstring_content
):
    """Test that inference command functions have appropriate docstrings."""
    # Arrange
    from mouse_tracking.cli import infer

    # Act
    command_function = getattr(infer, command_function_name)
    docstring = command_function.__doc__

    # Assert
    assert docstring is not None
    assert expected_docstring_content.lower() in docstring.lower()


@pytest.mark.parametrize(
    "command_name",
    [
        "arena-corner",
        "fecal-boli",
        "food-hopper",
        "lixit",
        "multi-identity",
        "multi-pose",
        "single-pose",
        "single-segmentation",
        "multi-segmentation",
    ],
    ids=[
        "arena_corner_help",
        "fecal_boli_help",
        "food_hopper_help",
        "lixit_help",
        "multi_identity_help",
        "multi_pose_help",
        "single_pose_help",
        "single_segmentation_help",
        "multi_segmentation_help",
    ],
)
def test_infer_command_help_format(command_name):
    """Test that each inference command has properly formatted help output."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"], env={"TERM": "dumb"})

    # Assert
    assert result.exit_code == 0
    assert f"Usage: root {command_name}" in result.stdout or "Usage:" in result.stdout
    # Options section might be styled differently (e.g., with rich formatting)
    assert "Options" in result.stdout or "--help" in result.stdout


def test_infer_command_name_conventions():
    """Test that command names follow expected conventions (kebab-case)."""
    # Arrange
    expected_names = [
        "arena_corner",
        "fecal_boli",
        "food_hopper",
        "lixit",
        "multi_identity",
        "multi_pose",
        "single_pose",
        "single_segmentation",
        "multi_segmentation",
    ]

    # Act
    registered_commands = app.registered_commands
    actual_names = [cmd.callback.__name__ for cmd in registered_commands]

    # Assert
    for name in expected_names:
        assert name in actual_names
        # Check that names use snake_case for function names (typer converts to kebab-case)
        assert "-" not in name  # Function names should use underscores


def test_infer_commands_require_input_validation():
    """Test that all inference commands properly validate required inputs."""
    # Arrange
    runner = CliRunner()
    commands_requiring_video_or_frame = [
        "arena-corner",
        "fecal-boli",
        "food-hopper",
        "lixit",
        "multi-identity",
        "multi-pose",
        "single-pose",
        "single-segmentation",
    ]

    # Act & Assert
    for command in commands_requiring_video_or_frame:
        # Test without required inputs - should fail
        result = runner.invoke(app, [command])
        assert result.exit_code != 0  # Should fail due to missing required parameters


def test_infer_commands_with_minimal_valid_inputs():
    """Test that inference commands work with minimal valid inputs."""
    # Arrange
    runner = CliRunner()
    test_video = Path("/tmp/test.mp4")
    test_output = Path("/tmp/output.json")

    commands_with_optional_outfile = [
        "arena-corner",
        "fecal-boli",
        "food-hopper",
        "lixit",
    ]

    commands_with_required_outfile = [
        "multi-identity",
        "multi-pose",
        "single-pose",
        "single-segmentation",
        "multi-segmentation",
    ]

    # Mock all the inference functions and file existence
    with (
        patch.object(Path, "exists", return_value=True),
        patch("mouse_tracking.cli.infer.infer_arena_corner_model"),
        patch("mouse_tracking.cli.infer.infer_fecal_boli_pytorch"),
        patch("mouse_tracking.cli.infer.infer_food_hopper_model"),
        patch("mouse_tracking.cli.infer.infer_lixit_model"),
        patch("mouse_tracking.cli.infer.infer_multi_identity_tfs"),
        patch("mouse_tracking.cli.infer.infer_multi_pose_pytorch"),
        patch("mouse_tracking.cli.infer.infer_single_pose_pytorch"),
        patch("mouse_tracking.cli.infer.infer_single_segmentation_tfs"),
        patch("mouse_tracking.cli.infer.infer_multi_segmentation_tfs"),
    ):
        # Test commands with optional out-file
        for command in commands_with_optional_outfile:
            result = runner.invoke(app, [command, "--video", str(test_video)])
            assert result.exit_code == 0

        # Test commands with required out-file
        for command in commands_with_required_outfile:
            result = runner.invoke(
                app,
                [command, "--out-file", str(test_output), "--video", str(test_video)],
            )
            assert result.exit_code == 0


def test_infer_commands_mutually_exclusive_validation():
    """Test that inference commands properly validate mutually exclusive video/frame options."""
    # Arrange
    runner = CliRunner()
    test_video = Path("/tmp/test.mp4")
    test_frame = Path("/tmp/test.jpg")
    test_output = Path("/tmp/output.json")

    commands = [
        "arena-corner",
        "fecal-boli",
        "food-hopper",
        "lixit",
        ("multi-identity", ["--out-file", str(test_output)]),
        ("multi-pose", ["--out-file", str(test_output)]),
        ("single-pose", ["--out-file", str(test_output)]),
        ("single-segmentation", ["--out-file", str(test_output)]),
        ("multi-segmentation", ["--out-file", str(test_output)]),
    ]

    with patch("pathlib.Path.exists", return_value=True):
        for command_info in commands:
            if isinstance(command_info, tuple):
                command, extra_args = command_info
            else:
                command, extra_args = command_info, []

            # Test both video and frame specified - should fail
            cmd_args = [
                command,
                "--video",
                str(test_video),
                "--frame",
                str(test_frame),
                *extra_args,
            ]
            result = runner.invoke(app, cmd_args)
            assert result.exit_code == 1
            assert "Cannot specify both --video and --frame" in result.stdout
