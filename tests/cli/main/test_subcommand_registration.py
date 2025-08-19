"""Unit tests for typer subcommand registration in main CLI app."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mouse_tracking.cli import infer, qa, utils
from mouse_tracking.cli.main import app


def test_main_app_is_typer_instance():
    """Test that the main app is a proper Typer instance."""
    # Arrange & Act
    import typer

    # Assert
    assert isinstance(app, typer.Typer)


def test_main_app_has_callback():
    """Test that the main app has a callback function registered."""
    # Arrange & Act
    callback_info = app.registered_callback

    # Assert
    assert callback_info is not None
    assert callback_info.callback is not None
    assert callable(callback_info.callback)


@pytest.mark.parametrize(
    "subcommand_name,expected_module",
    [
        ("infer", infer),
        ("qa", qa),
        ("utils", utils),
    ],
    ids=["infer_subcommand", "qa_subcommand", "utils_subcommand"],
)
def test_subcommands_are_registered(subcommand_name, expected_module):
    """Test that each subcommand is properly registered with the main app."""
    # Arrange & Act
    registered_groups = app.registered_groups

    # Assert
    assert len(registered_groups) >= 3  # Should have at least our 3 subcommands

    # Check that the expected module's app is in the registered groups
    found_subcommand = False
    for group_info in registered_groups:
        if group_info.typer_instance == expected_module.app:
            found_subcommand = True
            break

    assert found_subcommand, (
        f"Subcommand {subcommand_name} not found in registered groups"
    )


def test_all_expected_subcommands_registered():
    """Test that all expected subcommands are registered and no unexpected ones."""
    # Arrange
    expected_modules = {infer.app, qa.app, utils.app}

    # Act
    registered_groups = app.registered_groups
    registered_apps = {group.typer_instance for group in registered_groups}

    # Assert
    assert expected_modules.issubset(registered_apps)


def test_subcommand_help_text():
    """Test that subcommands have appropriate help text."""
    # Arrange
    expected_help_texts = {
        "infer": "Inference commands for mouse tracking runtime",
        "qa": "Quality assurance commands for mouse tracking runtime",
        "utils": "Utility commands for mouse tracking runtime",
    }

    # Act & Assert
    for subcommand_name, expected_help in expected_help_texts.items():
        # Use CLI runner to get help text
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        # Check that the subcommand and its help text appear in the output
        assert subcommand_name in result.stdout
        assert expected_help in result.stdout


def test_main_app_help_displays_subcommands():
    """Test that main app help displays all subcommands."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--help"])

    # Assert
    assert result.exit_code == 0
    assert "infer" in result.stdout
    assert "qa" in result.stdout
    assert "utils" in result.stdout


@pytest.mark.parametrize(
    "subcommand", ["infer", "qa", "utils"], ids=["infer_help", "qa_help", "utils_help"]
)
def test_subcommand_help_accessible(subcommand):
    """Test that help for each subcommand is accessible."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [subcommand, "--help"])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_main_app_docstring():
    """Test that the main app has the correct docstring from callback."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--help"])

    # Assert
    assert result.exit_code == 0
    assert "Mouse Tracking Runtime CLI" in result.stdout


def test_invalid_subcommand_error():
    """Test that invalid subcommands show appropriate error."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["invalid_command"])

    # Assert
    assert result.exit_code != 0
    assert "No such command" in result.stdout or "Usage:" in result.stdout


@pytest.mark.parametrize(
    "subcommand_module", [infer, qa, utils], ids=["infer_app", "qa_app", "utils_app"]
)
def test_subcommand_modules_have_typer_apps(subcommand_module):
    """Test that each subcommand module has a proper Typer app."""
    # Arrange & Act
    import typer

    # Assert
    assert hasattr(subcommand_module, "app")
    assert isinstance(subcommand_module.app, typer.Typer)


def test_main_app_version_option():
    """Test that the main app has a version option."""
    # Arrange
    runner = CliRunner()

    # Act
    with patch("mouse_tracking.cli.utils.__version__", "1.0.0"):
        result = runner.invoke(app, ["--version"])

    # Assert
    assert result.exit_code == 0
    assert "Mouse Tracking Runtime version" in result.stdout
    assert "1.0.0" in result.stdout


def test_main_app_verbose_option():
    """Test that the main app has a verbose option."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--verbose", "--help"])

    # Assert
    assert result.exit_code == 0
    # The verbose flag should be processed without error


def test_main_app_verbose_option_with_subcommand():
    """Test that verbose option works with subcommands."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, ["--verbose", "utils", "--help"])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "option_combo",
    [
        ["--help"],
        ["--verbose", "--help"],
        ["utils", "--help"],
        ["infer", "--help"],
        ["qa", "--help"],
    ],
    ids=["help_only", "verbose_help", "utils_help", "infer_help", "qa_help"],
)
def test_main_app_option_combinations(option_combo):
    """Test various option combinations with the main app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, option_combo)

    # Assert
    assert result.exit_code == 0


def test_main_app_without_arguments():
    """Test main app behavior when called without arguments."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [])

    # Assert
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_registered_groups_structure():
    """Test that registered groups have the expected structure."""
    # Arrange & Act
    registered_groups = app.registered_groups

    # Assert
    assert len(registered_groups) == 3  # Should have exactly 3 subcommands

    for group_info in registered_groups:
        assert hasattr(group_info, "typer_instance")
        assert hasattr(group_info, "name")
        assert hasattr(group_info, "help")
        assert group_info.name in ["infer", "qa", "utils"]


def test_callback_structure():
    """Test that the registered callback has the expected structure."""
    # Arrange & Act
    callback_info = app.registered_callback

    # Assert
    assert callback_info is not None
    assert hasattr(callback_info, "callback")
    assert hasattr(callback_info, "help")
    assert callback_info.callback.__name__ == "callback"
