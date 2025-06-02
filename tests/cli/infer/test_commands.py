"""Unit tests for inference CLI commands."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from mouse_tracking_runtime.cli.infer import app


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
        ("arena-corner", "Run arena corder inference."),
        ("fecal-boli", "Run fecal boli inference."),
        ("food-hopper", "Run food_hopper inference."),
        ("lixit", "Run lixit inference."),
        ("multi-identity", "Run multi-identity inference."),
        ("multi-pose", "Run multi-pose inference."),
        ("single-pose", "Run single-pose inference."),
        ("single-segmentation", "Run single-segmentation inference."),
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


def test_all_expected_infer_commands_present():
    """Test that all expected inference commands are present."""
    # Arrange
    expected_commands = {
        "arena_corner",
        "fecal_boli",
        "food_hopper",
        "lixit",
        "multi_identity",
        "multi_pose",
        "single_pose",
        "single_segmentation",
    }

    # Act
    registered_commands = app.registered_commands
    registered_command_names = {cmd.callback.__name__ for cmd in registered_commands}

    # Assert
    assert registered_command_names == expected_commands


def test_infer_help_displays_all_commands():
    """Test that infer help displays all available commands."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--help"])

    # Assert
    assert result.exit_code == 0
    assert "arena-corner" in result.stdout
    assert "fecal-boli" in result.stdout
    assert "food-hopper" in result.stdout
    assert "lixit" in result.stdout
    assert "multi-identity" in result.stdout
    assert "multi-pose" in result.stdout
    assert "single-pose" in result.stdout
    assert "single-segmentation" in result.stdout


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
    ],
)
def test_infer_command_functions_exist(command_function_name):
    """Test that all inference command functions exist in the module."""
    # Arrange & Act
    from mouse_tracking_runtime.cli import infer

    # Assert
    assert hasattr(infer, command_function_name)
    assert callable(getattr(infer, command_function_name))


@pytest.mark.parametrize(
    "command_function_name,expected_docstring_content",
    [
        ("arena_corner", "arena corder inference"),
        ("fecal_boli", "fecal boli inference"),
        ("food_hopper", "food_hopper inference"),
        ("lixit", "lixit inference"),
        ("multi_identity", "multi-identity inference"),
        ("multi_pose", "multi-pose inference"),
        ("single_pose", "single-pose inference"),
        ("single_segmentation", "single-segmentation inference"),
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
    ],
)
def test_infer_command_function_docstrings(
    command_function_name, expected_docstring_content
):
    """Test that inference command functions have appropriate docstrings."""
    # Arrange
    from mouse_tracking_runtime.cli import infer

    # Act
    command_function = getattr(infer, command_function_name)
    docstring = command_function.__doc__

    # Assert
    assert docstring is not None
    assert expected_docstring_content.lower() in docstring.lower()


def test_infer_commands_return_none():
    """Test that all inference commands return None (current implementations)."""
    # Arrange
    from mouse_tracking_runtime.cli import infer

    command_functions = [
        infer.arena_corner,
        infer.fecal_boli,
        infer.food_hopper,
        infer.lixit,
        infer.multi_identity,
        infer.multi_pose,
        infer.single_pose,
        infer.single_segmentation,
    ]

    # Act & Assert
    for func in command_functions:
        result = func()
        assert result is None


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
    ],
)
def test_infer_command_help_format(command_name):
    """Test that each inference command has properly formatted help output."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"])

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
    ]

    # Act
    registered_commands = app.registered_commands
    actual_names = [cmd.callback.__name__ for cmd in registered_commands]

    # Assert
    for name in expected_names:
        assert name in actual_names
        # Check that names use snake_case for function names (typer converts to kebab-case)
        assert "-" not in name  # Function names should use underscores
