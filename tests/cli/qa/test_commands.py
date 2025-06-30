"""Unit tests for QA CLI commands."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from mouse_tracking.cli.qa import app


def test_qa_app_is_typer_instance():
    """Test that the qa app is a proper Typer instance."""
    # Arrange & Act
    import typer

    # Assert
    assert isinstance(app, typer.Typer)


def test_qa_app_has_commands():
    """Test that the qa app has registered commands."""
    # Arrange & Act
    commands = app.registered_commands

    # Assert
    assert len(commands) > 0
    assert isinstance(commands, list)


@pytest.mark.parametrize(
    "command_name,expected_docstring",
    [
        ("single-pose", "Run single pose quality assurance."),
        (
            "multi-pose",
            "Run multi pose quality assurance.",
        ),
    ],
    ids=["single_pose_command", "multi_pose_command"],
)
def test_qa_commands_registered(command_name, expected_docstring):
    """Test that all expected QA commands are registered with correct docstrings."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert expected_docstring in result.stdout


def test_all_expected_qa_commands_present():
    """Test that all expected QA commands are present."""
    # Arrange
    expected_commands = {"single_pose", "multi_pose"}

    # Act
    registered_commands = app.registered_commands
    registered_command_names = {cmd.callback.__name__ for cmd in registered_commands}

    # Assert
    assert registered_command_names == expected_commands


def test_qa_help_displays_all_commands():
    """Test that qa help displays all available commands."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--help"])

    # Assert
    assert result.exit_code == 0
    assert "single-pose" in result.stdout
    assert "multi-pose" in result.stdout


@pytest.mark.parametrize(
    "command_name",
    ["single-pose", "multi-pose"],
    ids=["single_pose_execution", "multi_pose_execution"],
)
def test_qa_command_execution(command_name):
    """Test that each QA command can be executed without arguments."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name])

    # Assert
    # All current commands have empty implementations, so they should succeed
    assert result.exit_code == 0


def test_qa_invalid_command():
    """Test that invalid QA commands show appropriate error."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["invalid-command"])

    # Assert
    assert result.exit_code != 0
    assert "No such command" in result.stdout or "Usage:" in result.stdout


def test_qa_app_without_arguments():
    """Test qa app behavior when called without arguments."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [])

    # Assert
    assert result.exit_code == 2  # Typer returns 2 for missing required arguments
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "command_function_name",
    ["single_pose", "multi_pose"],
    ids=["single_pose_function", "multi_pose_function"],
)
def test_qa_command_functions_exist(command_function_name):
    """Test that all QA command functions exist in the module."""
    # Arrange & Act
    from mouse_tracking.cli import qa

    # Assert
    assert hasattr(qa, command_function_name)
    assert callable(getattr(qa, command_function_name))


@pytest.mark.parametrize(
    "command_function_name,expected_docstring_content",
    [
        ("single_pose", "single pose quality assurance"),
        (
            "multi_pose",
            "multi pose quality assurance",
        ),
    ],
    ids=["single_pose_docstring", "multi_pose_docstring"],
)
def test_qa_command_function_docstrings(
    command_function_name, expected_docstring_content
):
    """Test that QA command functions have appropriate docstrings."""
    # Arrange
    from mouse_tracking.cli import qa

    # Act
    command_function = getattr(qa, command_function_name)
    docstring = command_function.__doc__

    # Assert
    assert docstring is not None
    assert expected_docstring_content.lower() in docstring.lower()


def test_qa_commands_have_no_parameters():
    """Test that all current QA commands have no parameters (empty implementations)."""
    # Arrange
    from mouse_tracking.cli import qa
    import inspect

    command_functions = ["single_pose", "multi_pose"]

    # Act & Assert
    for func_name in command_functions:
        func = getattr(qa, func_name)
        signature = inspect.signature(func)

        # All current implementations should have no parameters
        assert len(signature.parameters) == 0


def test_qa_commands_return_none():
    """Test that all QA commands return None (current implementations)."""
    # Arrange
    from mouse_tracking.cli import qa

    command_functions = [qa.single_pose, qa.multi_pose]

    # Act & Assert
    for func in command_functions:
        result = func()
        assert result is None


@pytest.mark.parametrize(
    "command_name",
    ["single-pose", "multi-pose"],
    ids=["single_pose_help", "multi_pose_help"],
)
def test_qa_command_help_format(command_name):
    """Test that each QA command has properly formatted help output."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name, "--help"])

    # Assert
    assert result.exit_code == 0
    assert f"Usage: app {command_name}" in result.stdout or "Usage:" in result.stdout
    assert (
        "Options" in result.stdout
    )  # Rich formatting uses "╭─ Options ─" instead of "Options:"
    assert "--help" in result.stdout


def test_qa_app_module_docstring():
    """Test that the qa module has appropriate docstring."""
    # Arrange & Act
    from mouse_tracking.cli import qa

    # Assert
    assert qa.__doc__ is not None
    assert "qa" in qa.__doc__.lower() or "quality assurance" in qa.__doc__.lower()
    assert "cli" in qa.__doc__.lower()


def test_qa_command_name_conventions():
    """Test that command names follow expected conventions (kebab-case)."""
    # Arrange
    expected_names = ["single_pose", "multi_pose"]

    # Act
    registered_commands = app.registered_commands
    actual_names = [cmd.callback.__name__ for cmd in registered_commands]

    # Assert
    for name in expected_names:
        assert name in actual_names
        # Check that names use snake_case for function names (typer converts to kebab-case)
        assert "-" not in name  # Function names should use underscores


def test_qa_commands_are_properly_decorated():
    """Test that QA commands are properly decorated as typer commands."""
    # Arrange
    from mouse_tracking.cli import qa

    # Act
    single_pose_func = qa.single_pose
    multi_pose_func = qa.multi_pose

    # Assert
    # Typer decorates functions, so they should have certain attributes
    assert callable(single_pose_func)
    assert callable(multi_pose_func)


@pytest.mark.parametrize(
    "command_combo",
    [
        ["--help"],
        ["single-pose", "--help"],
        ["multi-pose", "--help"],
        ["single-pose"],
        ["multi-pose"],
    ],
    ids=[
        "qa_help",
        "single_pose_help",
        "multi_pose_help",
        "single_pose_run",
        "multi_pose_run",
    ],
)
def test_qa_command_combinations(command_combo):
    """Test various command combinations with the qa app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, command_combo)

    # Assert
    assert result.exit_code == 0


def test_qa_function_names_match_command_names():
    """Test that function names correspond properly to command names."""
    # Arrange
    function_to_command_mapping = {
        "single_pose": "single-pose",
        "multi_pose": "multi-pose",
    }

    # Act
    registered_commands = app.registered_commands

    # Assert
    for func_name, command_name in function_to_command_mapping.items():
        # Check that the function exists in the qa module
        from mouse_tracking.cli import qa

        assert hasattr(qa, func_name)

        # Check that the function is registered as a command
        found_command = False
        for cmd in registered_commands:
            if cmd.callback.__name__ == func_name:
                found_command = True
                break
        assert found_command, f"Function {func_name} not found in registered commands"
