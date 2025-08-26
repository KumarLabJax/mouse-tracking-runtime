"""Unit tests for version_callback helper function."""

from unittest.mock import patch

import pytest
import typer

from mouse_tracking.cli.utils import version_callback


@pytest.mark.parametrize(
    "value,should_print,should_exit",
    [
        (True, True, True),
        (False, False, False),
    ],
    ids=["value_true_prints_and_exits", "value_false_does_nothing"],
)
def test_version_callback_behavior(value, should_print, should_exit):
    """
    Test version_callback behavior with different input values.

    Args:
        value: Boolean flag to pass to version_callback
        should_print: Whether the function should print version info
        should_exit: Whether the function should raise typer.Exit
    """
    # Arrange
    with (
        patch("mouse_tracking.cli.utils.print") as mock_print,
        patch("mouse_tracking.cli.utils.__version__", "1.2.3"),
    ):
        # Act & Assert
        if should_exit:
            with pytest.raises(typer.Exit):
                version_callback(value)
        else:
            version_callback(value)  # Should not raise

        # Assert print behavior
        if should_print:
            mock_print.assert_called_once_with(
                "Mouse Tracking Runtime version: [green]1.2.3[/green]"
            )
        else:
            mock_print.assert_not_called()


def test_version_callback_with_true_prints_correct_format():
    """Test that version_callback prints the correct formatted message when value is True."""
    # Arrange
    test_version = "2.5.1"
    expected_message = f"Mouse Tracking Runtime version: [green]{test_version}[/green]"

    with (
        patch("mouse_tracking.cli.utils.print") as mock_print,
        patch("mouse_tracking.cli.utils.__version__", test_version),
    ):
        # Act & Assert
        with pytest.raises(typer.Exit):
            version_callback(True)

        # Assert
        mock_print.assert_called_once_with(expected_message)


def test_version_callback_with_false_no_side_effects():
    """Test that version_callback has no side effects when value is False."""
    # Arrange
    with patch("mouse_tracking.cli.utils.print") as mock_print:
        # Act
        result = version_callback(False)

        # Assert
        assert result is None
        mock_print.assert_not_called()


def test_version_callback_exit_exception_type():
    """Test that version_callback raises specifically typer.Exit when value is True."""
    # Arrange
    with (
        patch("mouse_tracking.cli.utils.print"),
        patch("mouse_tracking.cli.utils.__version__", "1.0.0"),
    ):
        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            version_callback(True)

        # Verify it's specifically a typer.Exit exception
        assert isinstance(exc_info.value, typer.Exit)


@pytest.mark.parametrize(
    "version_string",
    [
        "0.1.0",
        "1.0.0-alpha",
        "2.3.4-beta.1",
        "10.20.30",
        "1.0.0+build.123",
    ],
    ids=[
        "simple_version",
        "alpha_version",
        "beta_version",
        "large_numbers",
        "build_metadata",
    ],
)
def test_version_callback_with_various_version_formats(version_string):
    """Test version_callback with various version string formats."""
    # Arrange
    expected_message = (
        f"Mouse Tracking Runtime version: [green]{version_string}[/green]"
    )

    with (
        patch("mouse_tracking.cli.utils.print") as mock_print,
        patch("mouse_tracking.cli.utils.__version__", version_string),
    ):
        # Act & Assert
        with pytest.raises(typer.Exit):
            version_callback(True)

        # Assert
        mock_print.assert_called_once_with(expected_message)


def test_version_callback_print_called_when_true():
    """Test that print is called when value is True."""
    # Arrange
    with (
        patch("mouse_tracking.cli.utils.print") as mock_print,
        patch("mouse_tracking.cli.utils.__version__", "1.0.0"),
    ):
        # Act & Assert
        with pytest.raises(typer.Exit):
            version_callback(True)

        # Assert print was called exactly once
        assert mock_print.call_count == 1
        mock_print.assert_called_with(
            "Mouse Tracking Runtime version: [green]1.0.0[/green]"
        )


@pytest.mark.parametrize(
    "edge_case_version,description",
    [
        ("", "empty_string"),
        (None, "none_value"),
        ("   ", "whitespace_only"),
        ("v1.0.0", "prefixed_version"),
        ("1.0.0\n", "version_with_newline"),
    ],
    ids=[
        "empty_string",
        "none_value",
        "whitespace_only",
        "prefixed_version",
        "version_with_newline",
    ],
)
def test_version_callback_with_edge_case_versions(edge_case_version, description):
    """Test version_callback behavior with edge case version values."""
    # Arrange
    expected_message = (
        f"Mouse Tracking Runtime version: [green]{edge_case_version}[/green]"
    )

    with (
        patch("mouse_tracking.cli.utils.print") as mock_print,
        patch("mouse_tracking.cli.utils.__version__", edge_case_version),
    ):
        # Act & Assert
        with pytest.raises(typer.Exit):
            version_callback(True)

        # Assert
        mock_print.assert_called_once_with(expected_message)


def test_version_callback_return_value_when_false():
    """Test that version_callback returns None when value is False."""
    # Arrange
    with patch("mouse_tracking.cli.utils.print"):
        # Act
        result = version_callback(False)

        # Assert
        assert result is None


def test_version_callback_no_exception_when_false():
    """Test that version_callback does not raise any exception when value is False."""
    # Arrange
    with patch("mouse_tracking.cli.utils.print"):
        # Act & Assert - should not raise any exception
        try:
            version_callback(False)
        except Exception as e:
            pytest.fail(f"version_callback(False) raised an unexpected exception: {e}")


@pytest.mark.parametrize(
    "boolean_equivalent",
    [
        True,
        1,
        "true",
        [1],
        {"key": "value"},
    ],
    ids=["true_bool", "truthy_int", "truthy_string", "truthy_list", "truthy_dict"],
)
def test_version_callback_with_truthy_values(boolean_equivalent):
    """Test version_callback with various truthy values."""
    # Arrange
    with (
        patch("mouse_tracking.cli.utils.print") as mock_print,
        patch("mouse_tracking.cli.utils.__version__", "1.0.0"),
    ):
        # Act & Assert
        with pytest.raises(typer.Exit):
            version_callback(boolean_equivalent)

        # Assert print was called
        mock_print.assert_called_once()


@pytest.mark.parametrize(
    "boolean_equivalent",
    [
        False,
        0,
        "",
        [],
        {},
        None,
    ],
    ids=[
        "false_bool",
        "falsy_int",
        "falsy_string",
        "falsy_list",
        "falsy_dict",
        "none_value",
    ],
)
def test_version_callback_with_falsy_values(boolean_equivalent):
    """Test version_callback with various falsy values."""
    # Arrange
    with patch("mouse_tracking.cli.utils.print") as mock_print:
        # Act
        version_callback(boolean_equivalent)

        # Assert
        mock_print.assert_not_called()
