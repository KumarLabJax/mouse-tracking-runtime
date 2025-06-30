"""Integration tests for the complete CLI application."""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from mouse_tracking.cli.main import app


def test_full_cli_help_hierarchy():
    """Test the complete help hierarchy from main app through all subcommands."""
    # Arrange
    runner = CliRunner()

    # Act & Assert - Main app help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Mouse Tracking Runtime CLI" in result.stdout
    assert "infer" in result.stdout
    assert "qa" in result.stdout
    assert "utils" in result.stdout

    # Act & Assert - Infer subcommand help
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "arena-corner" in result.stdout
    assert "single-pose" in result.stdout
    assert "multi-pose" in result.stdout

    # Act & Assert - QA subcommand help
    result = runner.invoke(app, ["qa", "--help"])
    assert result.exit_code == 0
    assert "single-pose" in result.stdout
    assert "multi-pose" in result.stdout

    # Act & Assert - Utils subcommand help
    result = runner.invoke(app, ["utils", "--help"])
    assert result.exit_code == 0
    assert "aggregate-fecal-boli" in result.stdout
    assert "render-pose" in result.stdout


@pytest.mark.parametrize(
    "subcommand,command,expected_pattern",
    [
        ("infer", "arena-corner", None),  # Empty implementation
        ("infer", "single-pose", None),  # Empty implementation
        ("infer", "multi-pose", None),  # Empty implementation
        ("qa", "single-pose", None),  # Empty implementation
        ("qa", "multi-pose", None),  # Empty implementation
        ("utils", "aggregate-fecal-boli", "Aggregating fecal boli data"),
        ("utils", "render-pose", "Rendering pose data"),
        ("utils", "stitch-tracklets", "Stitching tracklets"),
    ],
    ids=[
        "infer_arena_corner",
        "infer_single_pose",
        "infer_multi_pose",
        "qa_single_pose",
        "qa_multi_pose",
        "utils_aggregate_fecal_boli",
        "utils_render_pose",
        "utils_stitch_tracklets",
    ],
)
def test_subcommand_execution_through_main_app(subcommand, command, expected_pattern):
    """Test executing subcommands through the main app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [subcommand, command])

    # Assert
    assert result.exit_code == 0
    if expected_pattern:
        assert expected_pattern in result.stdout


def test_main_app_version_option_integration():
    """Test version option integration across the CLI."""
    # Arrange
    runner = CliRunner()

    # Act
    with patch("mouse_tracking.cli.utils.__version__", "2.1.0"):
        result = runner.invoke(app, ["--version"])

    # Assert
    assert result.exit_code == 0
    assert "Mouse Tracking Runtime version" in result.stdout
    assert "2.1.0" in result.stdout


def test_main_app_verbose_option_integration():
    """Test verbose option integration with subcommands."""
    # Arrange
    runner = CliRunner()

    # Act & Assert - Verbose with main help
    result = runner.invoke(app, ["--verbose", "--help"])
    assert result.exit_code == 0

    # Act & Assert - Verbose with subcommand help
    result = runner.invoke(app, ["--verbose", "infer", "--help"])
    assert result.exit_code == 0

    # Act & Assert - Verbose with command execution
    result = runner.invoke(app, ["--verbose", "utils", "render-pose"])
    assert result.exit_code == 0
    assert "Rendering pose data" in result.stdout


@pytest.mark.parametrize(
    "invalid_path",
    [
        ["invalid-subcommand"],
        ["infer", "invalid-command"],
        ["qa", "invalid-command"],
        ["utils", "invalid-command"],
        ["invalid-subcommand", "invalid-command"],
    ],
    ids=[
        "invalid_subcommand",
        "invalid_infer_command",
        "invalid_qa_command",
        "invalid_utils_command",
        "double_invalid",
    ],
)
def test_invalid_command_paths_through_main_app(invalid_path):
    """Test that invalid command paths show appropriate errors."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, invalid_path)

    # Assert
    assert result.exit_code != 0
    assert "No such command" in result.stdout or "Usage:" in result.stdout


def test_complete_command_discovery():
    """Test that all commands are discoverable through the main app."""
    # Arrange
    runner = CliRunner()

    # Expected commands for each subcommand
    expected_commands = {
        "infer": [
            "arena-corner",
            "fecal-boli",
            "food-hopper",
            "lixit",
            "multi-identity",
            "multi-pose",
            "single-pose",
            "single-segmentation",
        ],
        "qa": ["single-pose", "multi-pose"],
        "utils": [
            "aggregate-fecal-boli",
            "clip-video-to-start",
            "downgrade-multi-to-single",
            "flip-xy-field",
            "render-pose",
            "stitch-tracklets",
        ],
    }

    # Act & Assert
    for subcommand, commands in expected_commands.items():
        result = runner.invoke(app, [subcommand, "--help"])
        assert result.exit_code == 0

        for command in commands:
            assert command in result.stdout


def test_help_command_accessibility():
    """Test that help is accessible at all levels of the CLI."""
    # Arrange
    runner = CliRunner()

    help_paths = [
        ["--help"],
        ["infer", "--help"],
        ["qa", "--help"],
        ["utils", "--help"],
        ["infer", "single-pose", "--help"],
        ["qa", "multi-pose", "--help"],
        ["utils", "render-pose", "--help"],
    ]

    # Act & Assert
    for path in help_paths:
        result = runner.invoke(app, path)
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--help" in result.stdout


def test_subcommand_isolation():
    """Test that subcommands are properly isolated from each other."""
    # Arrange
    runner = CliRunner()

    # Act & Assert - Commands with same names in different subcommands
    infer_single_pose = runner.invoke(app, ["infer", "single-pose"])
    qa_single_pose = runner.invoke(app, ["qa", "single-pose"])

    assert infer_single_pose.exit_code == 0
    assert qa_single_pose.exit_code == 0

    # Both should succeed but be different commands
    infer_single_pose_help = runner.invoke(app, ["infer", "single-pose", "--help"])
    qa_single_pose_help = runner.invoke(app, ["qa", "single-pose", "--help"])

    assert infer_single_pose_help.exit_code == 0
    assert qa_single_pose_help.exit_code == 0

    # Should have different help text indicating different purposes
    assert "inference" in infer_single_pose_help.stdout.lower()
    assert "quality assurance" in qa_single_pose_help.stdout.lower()


@pytest.mark.parametrize(
    "command_sequence",
    [
        ["infer", "arena-corner"],
        ["infer", "single-pose"],
        ["qa", "single-pose"],
        ["utils", "aggregate-fecal-boli"],
        ["utils", "render-pose"],
    ],
    ids=[
        "infer_arena_corner_sequence",
        "infer_single_pose_sequence",
        "qa_single_pose_sequence",
        "utils_aggregate_sequence",
        "utils_render_sequence",
    ],
)
def test_command_execution_sequences(command_sequence):
    """Test that command sequences execute properly through the main app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, command_sequence)

    # Assert
    assert result.exit_code == 0


def test_option_flag_combinations():
    """Test various combinations of options and flags."""
    # Arrange
    runner = CliRunner()

    test_combinations = [
        ["--verbose"],
        ["--verbose", "infer"],
        ["--verbose", "utils", "render-pose"],
        ["infer", "--help"],
        ["--verbose", "qa", "--help"],
    ]

    # Act & Assert
    for combo in test_combinations:
        result = runner.invoke(app, combo)
        # Some combinations may fail with exit code 2 (missing arguments)
        # Only help combinations should succeed with exit code 0
        if "--help" in combo:
            assert result.exit_code == 0
        else:
            # Commands without proper arguments may return exit code 2
            assert result.exit_code in [0, 2]


def test_cli_error_handling_consistency():
    """Test that error handling is consistent across all levels of the CLI."""
    # Arrange
    runner = CliRunner()

    error_scenarios = [
        ["nonexistent"],
        ["infer", "nonexistent"],
        ["qa", "nonexistent"],
        ["utils", "nonexistent"],
    ]

    # Act & Assert
    for scenario in error_scenarios:
        result = runner.invoke(app, scenario)
        assert result.exit_code != 0
        # Should contain helpful error information
        assert (
            "No such command" in result.stdout
            or "Usage:" in result.stdout
            or "Error" in result.stdout
        )


def test_complete_workflow_examples():
    """Test complete workflow examples that users might run."""
    # Arrange
    runner = CliRunner()

    workflows = [
        # Check version first
        ["--version"],
        # Explore available commands
        ["--help"],
        ["infer", "--help"],
        # Run specific inference commands
        ["infer", "single-pose"],
        ["infer", "arena-corner"],
        # Run QA commands
        ["qa", "single-pose"],
        # Run utility commands
        ["utils", "render-pose"],
        ["utils", "aggregate-fecal-boli"],
    ]

    # Act & Assert
    for i, workflow_step in enumerate(workflows):
        if workflow_step == ["--version"]:
            with patch("mouse_tracking.cli.utils.__version__", "1.0.0"):
                result = runner.invoke(app, workflow_step)
        else:
            result = runner.invoke(app, workflow_step)

        assert result.exit_code == 0, f"Workflow step {i} failed: {workflow_step}"


def test_subcommand_app_independence():
    """Test that each subcommand app can function independently."""
    # Arrange
    from mouse_tracking.cli import infer, qa, utils

    runner = CliRunner()

    # Act & Assert - Test each subcommand app independently
    # Infer app
    result = runner.invoke(infer.app, ["--help"])
    assert result.exit_code == 0
    assert "arena-corner" in result.stdout

    result = runner.invoke(infer.app, ["single-pose"])
    assert result.exit_code == 0

    # QA app
    result = runner.invoke(qa.app, ["--help"])
    assert result.exit_code == 0
    assert "single-pose" in result.stdout

    result = runner.invoke(qa.app, ["multi-pose"])
    assert result.exit_code == 0

    # Utils app
    result = runner.invoke(utils.app, ["--help"])
    assert result.exit_code == 0
    assert "render-pose" in result.stdout

    result = runner.invoke(utils.app, ["render-pose"])
    assert result.exit_code == 0
    assert "Rendering pose data" in result.stdout


def test_main_app_callback_integration():
    """Test that the main app callback integrates properly with subcommands."""
    # Arrange
    runner = CliRunner()

    # Act & Assert - Test callback options work with subcommands
    result = runner.invoke(app, ["--verbose", "utils", "render-pose"])
    assert result.exit_code == 0

    # Test that version callback overrides subcommand execution
    with patch("mouse_tracking.cli.utils.__version__", "1.0.0"):
        result = runner.invoke(app, ["--version", "utils", "render-pose"])
    assert result.exit_code == 0
    assert "Mouse Tracking Runtime version" in result.stdout
    # Should not execute the render-pose command due to version callback exit


def test_comprehensive_cli_structure():
    """Test the overall structure and organization of the CLI."""
    # Arrange
    runner = CliRunner()

    # Act
    main_help = runner.invoke(app, ["--help"])

    # Assert - Main structure
    assert main_help.exit_code == 0
    assert (
        "Commands" in main_help.stdout
    )  # Rich formatting uses "╭─ Commands ─" instead of "Commands:"

    # Should show all three main subcommands
    assert "infer" in main_help.stdout
    assert "qa" in main_help.stdout
    assert "utils" in main_help.stdout

    # Should show main options
    assert "--version" in main_help.stdout
    assert "--verbose" in main_help.stdout
