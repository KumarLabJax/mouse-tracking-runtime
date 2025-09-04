"""Mouse Tracking Runtime CLI."""

from typing import Annotated

import typer

from mouse_tracking.cli import infer, qa, utils
from mouse_tracking.cli.utils import version_callback

app = typer.Typer(no_args_is_help=True)


@app.callback()
def callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", help="Show the version and exit.", callback=version_callback
        ),
    ] = None,
    verbose: bool = typer.Option(False, help="Enable verbose output"),
) -> None:
    """Mouse Tracking Runtime CLI."""


app.add_typer(
    infer.app, name="infer", help="Inference commands for mouse tracking runtime"
)
app.add_typer(
    qa.app, name="qa", help="Quality assurance commands for mouse tracking runtime"
)
app.add_typer(
    utils.app, name="utils", help="Utility commands for mouse tracking runtime"
)


if __name__ == "__main__":
    app()
