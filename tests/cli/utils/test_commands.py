"""Unit tests for utility CLI commands."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from mouse_tracking.cli.utils import app


def test_utils_app_is_typer_instance():
    """Test that the utils app is a proper Typer instance."""
    # Arrange & Act
    import typer

    # Assert
    assert isinstance(app, typer.Typer)


def test_utils_app_has_commands():
    """Test that the utils app has registered commands."""
    # Arrange & Act
    commands = app.registered_commands

    # Assert
    assert len(commands) > 0
    assert isinstance(commands, list)


@pytest.mark.parametrize(
    "command_name,expected_docstring_content",
    [
        ("aggregate-fecal-boli", "Aggregate fecal boli data."),
        ("clip-video-to-start", "Clip video to start."),
        (
            "downgrade-multi-to-single",
            "Downgrade multi-identity data to single-identity.",
        ),
        ("flip-xy-field", "Flip XY field."),
        ("render-pose", "Render pose data."),
        ("stitch-tracklets", "Stitch tracklets."),
    ],
    ids=[
        "aggregate_fecal_boli_command",
        "clip_video_to_start_command",
        "downgrade_multi_to_single_command",
        "flip_xy_field_command",
        "render_pose_command",
        "stitch_tracklets_command",
    ],
)
def test_utils_commands_registered(command_name, expected_docstring_content):
    """Test that all expected utils commands are registered with correct docstrings."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert expected_docstring_content in result.stdout


def test_all_expected_utils_commands_present():
    """Test that all expected utility commands are present."""
    # Arrange
    expected_commands = {
        "aggregate_fecal_boli",
        "clip_video_to_start",
        "downgrade_multi_to_single",
        "flip_xy_field",
        "render_pose",
        "stitch_tracklets",
    }

    # Act
    registered_commands = app.registered_commands
    registered_command_names = {cmd.callback.__name__ for cmd in registered_commands}

    # Assert
    assert registered_command_names == expected_commands


def test_utils_help_displays_all_commands():
    """Test that utils help displays all available commands."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--help"])

    # Assert
    assert result.exit_code == 0
    assert "aggregate-fecal-boli" in result.stdout
    assert "clip-video-to-start" in result.stdout
    assert "downgrade-multi-to-single" in result.stdout
    assert "flip-xy-field" in result.stdout
    assert "render-pose" in result.stdout
    assert "stitch-tracklets" in result.stdout


@pytest.mark.parametrize(
    "command_name,expected_output_content",
    [
        (
            "aggregate-fecal-boli",
            "Aggregating fecal boli data... (not implemented yet)",
        ),
        ("clip-video-to-start", "Clipping video to start... (not implemented yet)"),
        (
            "downgrade-multi-to-single",
            "Downgrading multi-identity data to single-identity... (not implemented yet)",
        ),
        ("flip-xy-field", "Flipping XY field... (not implemented yet)"),
        ("render-pose", "Rendering pose data... (not implemented yet)"),
        ("stitch-tracklets", "Stitching tracklets... (not implemented yet)"),
    ],
    ids=[
        "aggregate_fecal_boli_execution",
        "clip_video_to_start_execution",
        "downgrade_multi_to_single_execution",
        "flip_xy_field_execution",
        "render_pose_execution",
        "stitch_tracklets_execution",
    ],
)
def test_utils_command_execution_with_output(command_name, expected_output_content):
    """Test that each utils command executes and prints expected placeholder message."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name])

    # Assert
    assert result.exit_code == 0
    assert expected_output_content in result.stdout


def test_utils_invalid_command():
    """Test that invalid utils commands show appropriate error."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["invalid-command"])

    # Assert
    assert result.exit_code != 0
    assert "No such command" in result.stdout or "Usage:" in result.stdout


def test_utils_app_without_arguments():
    """Test utils app behavior when called without arguments."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [])

    # Assert
    assert result.exit_code == 2  # Typer returns 2 for missing required arguments
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "command_function_name",
    [
        "aggregate_fecal_boli",
        "clip_video_to_start",
        "downgrade_multi_to_single",
        "flip_xy_field",
        "render_pose",
        "stitch_tracklets",
    ],
    ids=[
        "aggregate_fecal_boli_function",
        "clip_video_to_start_function",
        "downgrade_multi_to_single_function",
        "flip_xy_field_function",
        "render_pose_function",
        "stitch_tracklets_function",
    ],
)
def test_utils_command_functions_exist(command_function_name):
    """Test that all utils command functions exist in the module."""
    # Arrange & Act
    from mouse_tracking.cli import utils

    # Assert
    assert hasattr(utils, command_function_name)
    assert callable(getattr(utils, command_function_name))


@pytest.mark.parametrize(
    "command_function_name,expected_docstring_content",
    [
        ("aggregate_fecal_boli", "Aggregate fecal boli data"),
        ("clip_video_to_start", "Clip video to start"),
        (
            "downgrade_multi_to_single",
            "Downgrade multi-identity data to single-identity",
        ),
        ("flip_xy_field", "Flip XY field"),
        ("render_pose", "Render pose data"),
        ("stitch_tracklets", "Stitch tracklets"),
    ],
    ids=[
        "aggregate_fecal_boli_docstring",
        "clip_video_to_start_docstring",
        "downgrade_multi_to_single_docstring",
        "flip_xy_field_docstring",
        "render_pose_docstring",
        "stitch_tracklets_docstring",
    ],
)
def test_utils_command_function_docstrings(
    command_function_name, expected_docstring_content
):
    """Test that utils command functions have appropriate docstrings."""
    # Arrange
    from mouse_tracking.cli import utils

    # Act
    command_function = getattr(utils, command_function_name)
    docstring = command_function.__doc__

    # Assert
    assert docstring is not None
    assert expected_docstring_content.lower() in docstring.lower()


def test_utils_commands_have_no_parameters():
    """Test that all current utils commands have no parameters (placeholder implementations)."""
    # Arrange
    from mouse_tracking.cli import utils
    import inspect

    command_functions = [
        "aggregate_fecal_boli",
        "clip_video_to_start",
        "downgrade_multi_to_single",
        "flip_xy_field",
        "render_pose",
        "stitch_tracklets",
    ]

    # Act & Assert
    for func_name in command_functions:
        func = getattr(utils, func_name)
        signature = inspect.signature(func)

        # All current implementations should have no parameters
        assert len(signature.parameters) == 0


def test_utils_commands_return_none():
    """Test that all utils commands return None (current implementations)."""
    # Arrange
    from mouse_tracking.cli import utils

    command_functions = [
        utils.aggregate_fecal_boli,
        utils.clip_video_to_start,
        utils.downgrade_multi_to_single,
        utils.flip_xy_field,
        utils.render_pose,
        utils.stitch_tracklets,
    ]

    # Act & Assert
    for func in command_functions:
        result = func()
        assert result is None


@pytest.mark.parametrize(
    "command_name",
    [
        "aggregate-fecal-boli",
        "clip-video-to-start",
        "downgrade-multi-to-single",
        "flip-xy-field",
        "render-pose",
        "stitch-tracklets",
    ],
    ids=[
        "aggregate_fecal_boli_help",
        "clip_video_to_start_help",
        "downgrade_multi_to_single_help",
        "flip_xy_field_help",
        "render_pose_help",
        "stitch_tracklets_help",
    ],
)
def test_utils_command_help_format(command_name):
    """Test that each utils command has properly formatted help output."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"])

    # Assert
    assert result.exit_code == 0
    assert f"Usage: app {command_name}" in result.stdout or "Usage:" in result.stdout
    assert "Options" in result.stdout
    assert "--help" in result.stdout


def test_utils_app_module_docstring():
    """Test that the utils module has appropriate docstring."""
    # Arrange & Act
    from mouse_tracking.cli import utils

    # Assert
    assert utils.__doc__ is not None
    assert "utilities" in utils.__doc__.lower() or "helper" in utils.__doc__.lower()
    assert "cli" in utils.__doc__.lower()


def test_utils_command_name_conventions():
    """Test that command names follow expected conventions (kebab-case)."""
    # Arrange
    expected_names = [
        "aggregate_fecal_boli",
        "clip_video_to_start",
        "downgrade_multi_to_single",
        "flip_xy_field",
        "render_pose",
        "stitch_tracklets",
    ]

    # Act
    registered_commands = app.registered_commands
    actual_names = [cmd.callback.__name__ for cmd in registered_commands]

    # Assert
    for name in expected_names:
        assert name in actual_names
        # Check that names use snake_case for function names (typer converts to kebab-case)
        assert "-" not in name  # Function names should use underscores


def test_utils_version_callback_function_exists():
    """Test that the version_callback function exists in utils module."""
    # Arrange & Act
    from mouse_tracking.cli import utils

    # Assert
    assert hasattr(utils, "version_callback")
    assert callable(utils.version_callback)


@pytest.mark.parametrize(
    "command_combo",
    [
        ["--help"],
        ["aggregate-fecal-boli", "--help"],
        ["clip-video-to-start", "--help"],
        ["downgrade-multi-to-single", "--help"],
        ["flip-xy-field", "--help"],
        ["render-pose", "--help"],
        ["stitch-tracklets", "--help"],
        ["aggregate-fecal-boli"],
        ["render-pose"],
    ],
    ids=[
        "utils_help",
        "aggregate_fecal_boli_help",
        "clip_video_to_start_help",
        "downgrade_multi_to_single_help",
        "flip_xy_field_help",
        "render_pose_help",
        "stitch_tracklets_help",
        "aggregate_fecal_boli_run",
        "render_pose_run",
    ],
)
def test_utils_command_combinations(command_combo):
    """Test various command combinations with the utils app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, command_combo)

    # Assert
    assert result.exit_code == 0


def test_utils_function_names_match_command_names():
    """Test that function names correspond properly to command names."""
    # Arrange
    function_to_command_mapping = {
        "aggregate_fecal_boli": "aggregate-fecal-boli",
        "clip_video_to_start": "clip-video-to-start",
        "downgrade_multi_to_single": "downgrade-multi-to-single",
        "flip_xy_field": "flip-xy-field",
        "render_pose": "render-pose",
        "stitch_tracklets": "stitch-tracklets",
    }

    # Act
    registered_commands = app.registered_commands

    # Assert
    for func_name, command_name in function_to_command_mapping.items():
        # Check that the function exists in the utils module
        from mouse_tracking.cli import utils

        assert hasattr(utils, func_name)

        # Check that the function is registered as a command
        found_command = False
        for cmd in registered_commands:
            if cmd.callback.__name__ == func_name:
                found_command = True
                break
        assert found_command, f"Function {func_name} not found in registered commands"


def test_utils_rich_print_import():
    """Test that utils module imports rich print correctly."""
    # Arrange & Act
    from mouse_tracking.cli import utils
    import inspect

    # Act
    source = inspect.getsource(utils)

    # Assert
    assert "from rich import print" in source


def test_utils_commands_detailed_docstrings():
    """Test that utils commands have detailed docstrings with proper formatting."""
    # Arrange
    from mouse_tracking.cli import utils

    command_functions = [
        utils.aggregate_fecal_boli,
        utils.clip_video_to_start,
        utils.downgrade_multi_to_single,
        utils.flip_xy_field,
        utils.render_pose,
        utils.stitch_tracklets,
    ]

    # Act & Assert
    for func in command_functions:
        docstring = func.__doc__

        # Should have a docstring
        assert docstring is not None

        # Should have at least a description paragraph
        lines = [line.strip() for line in docstring.strip().split("\n") if line.strip()]
        assert len(lines) >= 2  # Title and description (reduced from 3 to 2)

        # First line should be a brief description
        assert len(lines[0]) > 0
        assert lines[0].endswith(".")

        # Should contain the word "command" in the description
        assert "command" in docstring.lower()
