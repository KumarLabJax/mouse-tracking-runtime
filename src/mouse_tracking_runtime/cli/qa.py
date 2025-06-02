"""Mouse Tracking Runtime QA CLI"""

import typer

app = typer.Typer()


@app.command()
def single_pose():
    """Run single pose quality assurance."""


@app.command()
def multi_pose():
    """Run multi pose quality assurance."""
