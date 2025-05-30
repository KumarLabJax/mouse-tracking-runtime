"""Mouse Tracking Runtime inference CLI"""

import typer

app = typer.Typer()


@app.command()
def arena_corner():
    """Run arena corder inference."""


@app.command()
def fecal_boli():
    """Run fecal boli inference."""


@app.command()
def food_hopper():
    """Run food_hopper inference."""


@app.command()
def lixit():
    """Run lixit inference."""


@app.command()
def multi_identity():
    """Run multi-identity inference."""


@app.command()
def multi_pose():
    """Run multi-pose inference."""


@app.command()
def single_pose():
    """Run single-pose inference."""


@app.command()
def single_segmentation():
    """Run single-segmentation inference."""

