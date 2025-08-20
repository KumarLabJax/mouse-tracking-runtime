"""Unit tests for QA CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

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
    "command_name,expected_exit_code",
    [
        ("single-pose", 2),  # Missing required pose argument
        ("multi-pose", 0),  # Empty implementation, no arguments required
    ],
    ids=["single_pose_execution", "multi_pose_execution"],
)
def test_qa_command_execution_without_args(command_name, expected_exit_code):
    """Test QA command execution without arguments shows appropriate behavior."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [command_name])

    # Assert
    assert result.exit_code == expected_exit_code


def test_qa_single_pose_execution_with_mock_file():
    """Test that single-pose command can be executed with proper arguments."""
    # Arrange
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        pose_file = Path(tmp_file.name)

    # Mock the inspect_pose_v6 function to avoid actual file processing
    # and mock to_csv to avoid creating real CSV files
    with (
        patch("mouse_tracking.cli.qa.inspect_pose_v6") as mock_inspect,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
    ):
        mock_inspect.return_value = {"metric1": 0.5, "metric2": 0.8}

        # Act
        result = runner.invoke(app, ["single-pose", str(pose_file)])

        # Assert
        assert result.exit_code == 0
        mock_inspect.assert_called_once()
        mock_to_csv.assert_called_once()

    # Cleanup
    if pose_file.exists():
        pose_file.unlink()


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


def test_qa_single_pose_has_parameters():
    """Test that single_pose command has the expected parameters."""
    # Arrange
    import inspect

    from mouse_tracking.cli import qa

    # Act
    func = qa.single_pose
    signature = inspect.signature(func)

    # Assert
    expected_params = {"pose", "output", "pad", "duration"}
    actual_params = set(signature.parameters.keys())
    assert actual_params == expected_params


def test_qa_multi_pose_has_no_parameters():
    """Test that multi_pose command has no parameters (empty implementation)."""
    # Arrange
    import inspect

    from mouse_tracking.cli import qa

    # Act
    func = qa.multi_pose
    signature = inspect.signature(func)

    # Assert
    assert len(signature.parameters) == 0


def test_qa_multi_pose_returns_none():
    """Test that multi_pose command returns None (current implementation)."""
    # Arrange
    from mouse_tracking.cli import qa

    # Act
    with pytest.raises(typer.Exit):
        # This will raise SystemExit due to the typer Exit call in multi_pose
        qa.multi_pose()


def test_qa_single_pose_execution_with_mocked_dependencies():
    """Test single_pose function execution with mocked dependencies."""
    # Arrange
    from pathlib import Path

    from mouse_tracking.cli import qa

    mock_pose_path = Path("/fake/pose.h5")
    mock_result = {"metric1": 0.5, "metric2": 0.8}

    with (
        patch("mouse_tracking.cli.qa.inspect_pose_v6") as mock_inspect,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
        patch("pandas.Timestamp.now") as mock_timestamp,
    ):
        mock_inspect.return_value = mock_result
        mock_timestamp.return_value.strftime.return_value = "20231201_120000"

        # Act
        result = qa.single_pose(
            pose=mock_pose_path, output=None, pad=150, duration=108000
        )

        # Assert
        assert result is None
        mock_inspect.assert_called_once_with(mock_pose_path, pad=150, duration=108000)
        mock_to_csv.assert_called_once()


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
    result = runner.invoke(app, [command_name, "--help"], env={"TERM": "dumb"})

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
    "command_combo,expected_exit_code",
    [
        (["--help"], 0),
        (["single-pose", "--help"], 0),
        (["multi-pose", "--help"], 0),
        (["multi-pose"], 0),  # Empty implementation, no args required
    ],
    ids=[
        "qa_help",
        "single_pose_help",
        "multi_pose_help",
        "multi_pose_run",
    ],
)
def test_qa_command_combinations(command_combo, expected_exit_code):
    """Test various command combinations with the qa app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, command_combo)

    # Assert
    assert result.exit_code == expected_exit_code


def test_qa_single_pose_requires_arguments():
    """Test that single-pose command requires pose argument."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["single-pose"])

    # Assert
    assert result.exit_code == 2  # Missing required argument
    assert "Missing argument" in result.stdout or "Usage:" in result.stdout


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
    for func_name, _command_name in function_to_command_mapping.items():
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
