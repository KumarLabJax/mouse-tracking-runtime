"""Regression tests for the ``click`` dependency used by the CLI options.

``mouse_tracking.cli.infer`` builds enumerated options with
``typer.Option(click_type=click.Choice([...]))``. That idiom depends on the
top-level ``click`` distribution being installed *and* on typer consuming the
same ``click`` rather than a vendored copy. Typer 0.26 dropped ``click`` from
its runtime requirements and vendored it as ``typer._click``, which breaks the
idiom in two distinct ways:

* ``click`` disappears entirely, so importing the module raises
  ``ModuleNotFoundError`` and every CLI invocation aborts before parsing.
* ``click`` is installed alongside typer >= 0.26, so the import succeeds but the
  vendored parser no longer recognises the external ``Choice`` instance and the
  option renders a ``<function>`` metavar instead of the enumerated values.

These tests fail loudly for both, so the version constraints in
``pyproject.toml`` cannot be relaxed without the call sites being migrated.
"""

import click
import typer
from typer.testing import CliRunner

from mouse_tracking.cli.main import app


def _choice_options():
    """Yield ``(command_path, option_name, choices)`` for every Choice option."""
    root = typer.main.get_command(app)

    def walk(command, path):
        subcommands = getattr(command, "commands", None)
        if subcommands:
            for name, sub in subcommands.items():
                yield from walk(sub, [*path, name])
            return
        for param in command.params:
            if isinstance(param.type, click.Choice):
                yield path, param.name, list(param.type.choices)

    yield from walk(root, [])


def test_choice_options_are_discovered():
    """Guard the test itself: the CLI must expose Choice-typed options."""
    # Arrange & Act
    discovered = list(_choice_options())

    # Assert
    assert discovered, "no click.Choice options found; has the CLI been migrated?"


def test_choice_options_render_their_choices():
    """Test that every Choice option advertises its values in --help output."""
    # Arrange
    runner = CliRunner()
    help_cache = {}

    # Act & Assert
    for path, option_name, choices in _choice_options():
        key = tuple(path)
        if key not in help_cache:
            result = runner.invoke(app, [*path, "--help"])
            assert result.exit_code == 0, f"{' '.join(path)} --help failed"
            help_cache[key] = result.output
        output = help_cache[key]

        assert "<function" not in output, (
            f"{' '.join(path)} renders a function metavar; typer is not using the "
            "installed click"
        )
        for choice in choices:
            assert choice in output, (
                f"{' '.join(path)} --{option_name} does not advertise {choice!r}"
            )
