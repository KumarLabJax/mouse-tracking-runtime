"""Helper utilities for the CLI."""

import typer
from rich import print

from mouse_tracking import __version__

app = typer.Typer()


def version_callback(value: bool) -> None:
    """
    Display the application version and exit.

    Args:
        value: Flag indicating whether to show version

    """
    if value:
        print(f"Mouse Tracking Runtime version: [green]{__version__}[/green]")
        raise typer.Exit()


@app.command()
def aggregate_fecal_boli():
    """
    Aggregate fecal boli data.

    This command processes and aggregates fecal boli data from the specified source.
    """
    print("Aggregating fecal boli data... (not implemented yet)")


@app.command()
def clip_video_to_start():
    """
    Clip video to start.

    This command clips the video to the start time specified in the configuration.
    """
    print("Clipping video to start... (not implemented yet)")


@app.command()
def downgrade_multi_to_single():
    """
    Downgrade multi-identity data to single-identity.

    This command processes multi-identity data and downgrades it to single-identity format.
    """
    print("Downgrading multi-identity data to single-identity... (not implemented yet)")


@app.command()
def flip_xy_field():
    """
    Flip XY field.

    This command flips the XY coordinates in the dataset.
    """
    print("Flipping XY field... (not implemented yet)")


@app.command()
def render_pose():
    """
    Render pose data.

    This command renders the pose data from the specified source.
    """
    print("Rendering pose data... (not implemented yet)")


@app.command()
def stitch_tracklets():
    """
    Stitch tracklets.

    This command stitches tracklets from the specified source.
    """
    print("Stitching tracklets... (not implemented yet)")
