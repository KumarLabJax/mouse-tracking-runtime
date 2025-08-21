"""Integration tests for the complete CLI application."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

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
    "subcommand,command,expected_exit_code,expected_pattern",
    [
        ("infer", "arena-corner", 1, None),  # Missing required --video or --frame
        ("infer", "single-pose", 2, None),  # Missing required --out-file
        ("infer", "multi-pose", 2, None),  # Missing required --out-file
        ("qa", "single-pose", 2, None),  # Missing required pose argument
        ("qa", "multi-pose", 0, None),  # Empty implementation
        ("utils", "aggregate-fecal-boli", 2, None),  # Missing required folder argument
        ("utils", "render-pose", 2, None),  # Missing required arguments
        ("utils", "stitch-tracklets", 2, None),  # Missing required pose file argument
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
def test_subcommand_execution_through_main_app(
    subcommand, command, expected_exit_code, expected_pattern
):
    """Test executing subcommands through the main app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, [subcommand, command])

    # Assert
    assert result.exit_code == expected_exit_code
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

    # Act & Assert - Verbose with command execution (should fail due to missing args)
    result = runner.invoke(app, ["--verbose", "utils", "render-pose"])
    assert result.exit_code == 2  # Missing required arguments


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
        result = runner.invoke(app, path, env={"TERM": "dumb"})
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

    # Both should fail with missing arguments, but with different error codes
    assert infer_single_pose.exit_code == 2  # Missing --out-file
    assert qa_single_pose.exit_code == 2  # Missing pose argument

    # Both should succeed with help
    infer_single_pose_help = runner.invoke(app, ["infer", "single-pose", "--help"])
    qa_single_pose_help = runner.invoke(app, ["qa", "single-pose", "--help"])

    assert infer_single_pose_help.exit_code == 0
    assert qa_single_pose_help.exit_code == 0

    # Should have different help text indicating different purposes
    assert "inference" in infer_single_pose_help.stdout.lower()
    assert "quality assurance" in qa_single_pose_help.stdout.lower()


@pytest.mark.parametrize(
    "command_sequence,expected_exit_code",
    [
        (["infer", "arena-corner"], 1),  # Missing required --video or --frame
        (["infer", "single-pose"], 2),  # Missing required --out-file
        (["qa", "single-pose"], 2),  # Missing required pose argument
        (["qa", "multi-pose"], 0),  # Empty implementation
        (["utils", "aggregate-fecal-boli"], 2),  # Missing required folder argument
        (["utils", "render-pose"], 2),  # Missing required arguments
    ],
    ids=[
        "infer_arena_corner_sequence",
        "infer_single_pose_sequence",
        "qa_single_pose_sequence",
        "qa_multi_pose_sequence",
        "utils_aggregate_sequence",
        "utils_render_sequence",
    ],
)
def test_command_execution_sequences(command_sequence, expected_exit_code):
    """Test that command sequences execute properly through the main app."""
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(app, command_sequence)

    # Assert
    assert result.exit_code == expected_exit_code


def test_option_flag_combinations():
    """Test various combinations of options and flags."""
    # Arrange
    runner = CliRunner()

    test_combinations = [
        (["--verbose"], 2),  # Missing subcommand
        (["--verbose", "infer"], 2),  # Missing command
        (["--verbose", "utils", "render-pose"], 2),  # Missing required arguments
        (["infer", "--help"], 0),  # Help always succeeds
        (["--verbose", "qa", "--help"], 0),  # Help with verbose
    ]

    # Act & Assert
    for combo, expected_exit in test_combinations:
        result = runner.invoke(app, combo)
        assert result.exit_code == expected_exit


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
        (["--version"], 0),
        # Explore available commands
        (["--help"], 0),
        (["infer", "--help"], 0),
        # Try to run specific inference commands without args (should fail appropriately)
        (["infer", "single-pose"], 2),  # Missing --out-file
        (["infer", "arena-corner"], 1),  # Missing --video or --frame
        # Try QA commands
        (["qa", "single-pose"], 2),  # Missing pose argument
        (["qa", "multi-pose"], 0),  # Empty implementation
        # Run utility commands (these now require arguments)
        (["utils", "render-pose"], 2),  # Missing required arguments
        (["utils", "aggregate-fecal-boli"], 2),  # Missing required folder argument
    ]

    # Act & Assert
    for i, (workflow_step, expected_exit) in enumerate(workflows):
        if workflow_step == ["--version"]:
            with patch("mouse_tracking.cli.utils.__version__", "1.0.0"):
                result = runner.invoke(app, workflow_step)
        else:
            result = runner.invoke(app, workflow_step)

        assert result.exit_code == expected_exit, (
            f"Workflow step {i} failed: {workflow_step}"
        )


def test_subcommand_app_independence():
    """Test that each subcommand app can function independently."""
    # Arrange
    from mouse_tracking.cli import infer, qa, utils

    runner = CliRunner()

    # Act & Assert - Test each subcommand app independently
    # Infer app help should work
    result = runner.invoke(infer.app, ["--help"])
    assert result.exit_code == 0
    assert "arena-corner" in result.stdout

    # Infer app commands should fail without required arguments
    result = runner.invoke(infer.app, ["single-pose"])
    assert result.exit_code == 2  # Missing --out-file

    # QA app help should work
    result = runner.invoke(qa.app, ["--help"])
    assert result.exit_code == 0
    assert "single-pose" in result.stdout

    # QA multi-pose should work (empty implementation)
    result = runner.invoke(qa.app, ["multi-pose"])
    assert result.exit_code == 0

    # Utils app should work
    result = runner.invoke(utils.app, ["--help"])
    assert result.exit_code == 0
    assert "render-pose" in result.stdout

    # Utils commands now require arguments
    result = runner.invoke(utils.app, ["render-pose"])
    assert result.exit_code == 2  # Missing required arguments


def test_main_app_callback_integration():
    """Test that the main app callback integrates properly with subcommands."""
    # Arrange
    runner = CliRunner()

    # Act & Assert - Test callback options work with subcommands (will fail due to missing args)
    result = runner.invoke(app, ["--verbose", "utils", "render-pose"])
    assert result.exit_code == 2  # Missing required arguments

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
    main_help = runner.invoke(app, ["--help"], env={"TERM": "dumb"})

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


def test_commands_with_proper_arguments():
    """Test that commands work when provided with proper arguments."""
    # Arrange
    runner = CliRunner()

    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        video_path = Path(tmp_video.name)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_pose:
        pose_path = Path(tmp_pose.name)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_out:
        out_path = Path(tmp_out.name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_folder = Path(tmp_dir)

        try:
            # Test infer arena-corner with video (mock the entire inference function)
            with (
                patch(
                    "mouse_tracking.cli.infer.infer_arena_corner_model"
                ) as mock_arena,
                patch.object(Path, "exists", return_value=True),
            ):
                result = runner.invoke(
                    app, ["infer", "arena-corner", "--video", str(video_path)]
                )
                assert result.exit_code == 0
                mock_arena.assert_called_once()

            # Test infer single-pose with proper arguments (mock the entire inference function)
            with (
                patch(
                    "mouse_tracking.cli.infer.infer_single_pose_pytorch"
                ) as mock_pose,
                patch.object(Path, "exists", return_value=True),
            ):
                result = runner.invoke(
                    app,
                    [
                        "infer",
                        "single-pose",
                        "--video",
                        str(video_path),
                        "--out-file",
                        str(out_path),
                    ],
                )
                assert result.exit_code == 0
                mock_pose.assert_called_once()

            # Test qa single-pose with proper arguments (mock the inspect function)
            with (
                patch("mouse_tracking.cli.qa.inspect_pose_v6") as mock_inspect,
                patch("pandas.DataFrame.to_csv") as mock_to_csv,
                patch("pandas.Timestamp.now") as mock_timestamp,
            ):
                mock_inspect.return_value = {"metric1": 0.5}
                mock_timestamp.return_value.strftime.return_value = "20231201_120000"

                result = runner.invoke(app, ["qa", "single-pose", str(pose_path)])
                assert result.exit_code == 0
                mock_to_csv.assert_called_once()

            # Test utils commands with proper arguments
            with patch(
                "mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data"
            ) as mock_aggregate:
                # Mock the DataFrame with a to_csv method
                mock_df = MagicMock()
                mock_aggregate.return_value = mock_df

                result = runner.invoke(
                    app, ["utils", "aggregate-fecal-boli", str(tmp_folder)]
                )
                assert result.exit_code == 0
                mock_aggregate.assert_called_once()

            # Test utils render-pose with mocked function
            with patch("mouse_tracking.cli.utils.render.process_video") as mock_render:
                result = runner.invoke(
                    app,
                    [
                        "utils",
                        "render-pose",
                        str(video_path),
                        str(pose_path),
                        str(out_path),
                    ],
                )
                assert result.exit_code == 0
                mock_render.assert_called_once()

            # Test utils stitch-tracklets with mocked function
            with patch("mouse_tracking.cli.utils.match_predictions") as mock_stitch:
                result = runner.invoke(
                    app, ["utils", "stitch-tracklets", str(pose_path)]
                )
                assert result.exit_code == 0
                mock_stitch.assert_called_once()

        finally:
            # Cleanup
            for path in [video_path, pose_path, out_path]:
                if path.exists():
                    path.unlink()
