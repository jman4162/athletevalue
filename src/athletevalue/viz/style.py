"""One style for every figure, and SVG output that is identical byte for byte across runs."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure

# Okabe-Ito / Wong palette: distinguishable with the common forms of colour blindness.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8C8C8C"

WIDTH = 7.0
"""Inches; renders about 700 px wide on GitHub."""

RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.dpi": 100,
    "svg.fonttype": "path",
    "svg.hashsalt": "athletevalue",
    "path.simplify": True,
}


def styled[**P](function: Callable[P, Figure]) -> Callable[P, Figure]:
    """Draw inside the package style; matplotlib reads most settings when an artist is made."""

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Figure:
        with mpl.rc_context(RC):
            return function(*args, **kwargs)

    return wrapper


def new_figure(height: float, ncols: int = 1) -> tuple[Figure, Any]:
    """A figure without pyplot's global state. Call from a ``styled`` function."""
    figure = Figure(figsize=(WIDTH, height), layout="constrained")
    return figure, figure.subplots(1, ncols)


def save_svg(figure: Figure, path: Path) -> None:
    """Write *figure* as SVG with no timestamp, so unchanged figures leave no diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(RC):
        figure.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
