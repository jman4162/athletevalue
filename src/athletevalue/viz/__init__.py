"""Figures drawn from the validation snapshot (``docs/_static/data/snapshot.json``).

Requires the optional extra: ``pip install "athletevalue[viz]"``. Every function
takes the snapshot as a plain dictionary and returns a matplotlib ``Figure``; none
downloads data or fits a model, so figures regenerate offline in seconds.
"""

from __future__ import annotations

try:
    import matplotlib as _matplotlib  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'athletevalue.viz needs matplotlib. Install it with: pip install "athletevalue[viz]"'
    ) from error

from athletevalue.viz.figures import (
    cv_curves_figure,
    example_roster_figure,
    rating_precision_figure,
    revenue_bootstrap_figure,
    team_ratings_figure,
    war_replacement_figure,
)
from athletevalue.viz.style import save_svg

FIGURES = {
    "team_ratings_vs_torvik": team_ratings_figure,
    "cv_error_by_penalty": cv_curves_figure,
    "war_by_replacement": war_replacement_figure,
    "rating_precision": rating_precision_figure,
    "revenue_per_win": revenue_bootstrap_figure,
    "example_roster": example_roster_figure,
}
"""File stem to figure function, in the order the README shows them."""

__all__ = [
    "FIGURES",
    "cv_curves_figure",
    "example_roster_figure",
    "rating_precision_figure",
    "revenue_bootstrap_figure",
    "save_svg",
    "team_ratings_figure",
    "war_replacement_figure",
]
