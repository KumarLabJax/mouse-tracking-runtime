"""Mouse Tracking Runtime QA CLI."""

from pathlib import Path

import pandas as pd
import typer

from mouse_tracking.pose.inspect import inspect_pose_v6

app = typer.Typer()


@app.command()
def single_pose(
    pose: Path = typer.Argument(..., help="Path to the pose file to inspect"),
    output: Path | None = typer.Option(
        None, help="Output filename. Will append row if already exists."
    ),
    pad: int = typer.Option(
        150, help="Number of frames to pad at the start of the video"
    ),
    duration: int = typer.Option(108000, help="Duration of the video in frames"),
):
    """Run single pose quality assurance."""
    # Dynamically set output filename if not provided
    if not output:
        output = Path(
            f"QA_{pose.stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    # Perform Single Pose QA Inspection
    result = inspect_pose_v6(pose, pad=pad, duration=duration)

    # Write the result to the output file
    pd.DataFrame(result, index=[0]).to_csv(
        output, mode="a", index=False, header=not output.exists()
    )


@app.command()
def single_feature(
    pose: Path = typer.Argument(..., help="Path to the pose file to inspect"),
    behavior: Path = typer.Argument(..., help="Path to the behavior table to inspect"),
    output: Path | None = typer.Option(
        None, help="Output filename. Will append row if already exists."
    ),
):
    """Run single mouse feature inspection."""
    # Dynamically set the output filename if not provided
    if not output:
        output = Path(
            f"QA_features_{pose.stem}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    # TODO implement desired plots of feature data for more in-depth inspection
    raise NotImplementedError("Feature inspection is not yet implemented.")


@app.command()
def multi_pose():
    """Run multi pose quality assurance."""
    typer.echo("Multi pose quality assurance is not implemented yet.")
    raise typer.Exit()
