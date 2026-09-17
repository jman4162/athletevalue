"""The six README figures. Each reads the snapshot dictionary and returns a Figure."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from athletevalue.viz.style import (
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    SKY,
    VERMILION,
    new_figure,
    styled,
)

Snapshot = dict[str, Any]

_DEFINITION_LABELS = {
    "nba_convention": "NBA convention",
    "pooled": "Pooled low-minute players",
    "bench_median": "Bench median",
}
_DEFINITION_COLORS = {"nba_convention": ORANGE, "pooled": PURPLE, "bench_median": BLUE}


def _thousands(value: float, _position: int) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}k"


def _latest(snapshot: Snapshot) -> str:
    return str(max(snapshot["seasons"], key=int))


def _season_label(season: str) -> str:
    year = int(season)
    return f"{year - 1}-{str(year)[2:]}"


@styled
def team_ratings_figure(snapshot: Snapshot, season: str | None = None) -> Figure:
    """Team net rating against Torvik's AdjOE - AdjDE, without and with the box prior."""
    season = season or _latest(snapshot)
    teams = snapshot["seasons"][season]["teams"]["torvik"]
    torvik = np.asarray(teams["adj_em"], dtype=float)
    figure, axes = new_figure(3.3, ncols=2)
    limits = (min(torvik.min(), -35.0), max(torvik.max(), 35.0))
    for ax, key, color, title in (
        (axes[0], "adj_net_no_prior", SKY, "No prior"),
        (axes[1], "adj_net", BLUE, "Box-score prior"),
    ):
        ours = np.asarray(teams[key], dtype=float)
        slope, intercept = np.polyfit(torvik, ours, 1)
        ax.plot(limits, limits, color=GREY, linewidth=0.8, linestyle="--", label="equal")
        ax.scatter(torvik, ours, s=9, color=color, alpha=0.7, linewidths=0)
        grid = np.array(limits)
        ax.plot(grid, intercept + slope * grid, color=VERMILION, linewidth=1.4, label="fit")
        ax.set_title(title)
        ax.text(
            0.04,
            0.93,
            f"slope {slope:.2f}\n{len(torvik)} teams",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal")
        ax.set_xlabel("Torvik AdjOE − AdjDE, points per 100")
    axes[0].set_ylabel("athletevalue team net rating")
    axes[1].legend(loc="lower right")
    figure.suptitle(
        f"Team net rating against Torvik, {_season_label(season)}",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=10.5,
    )
    return figure


@styled
def cv_curves_figure(snapshot: Snapshot) -> Figure:
    """Held-out error against the ridge penalty for no prior, zero box plus team adjustment, box."""
    seasons = sorted(snapshot["seasons"], key=int)
    figure, axes = new_figure(3.0, ncols=len(seasons))
    axes = np.atleast_1d(axes)
    series = (
        ("no_prior", "No prior", GREY, "o"),
        ("zero_box_team_adjusted", "Team adjustment only", ORANGE, "s"),
        ("box_team_adjusted", "Box model + team adjustment", BLUE, "D"),
    )
    for ax, season in zip(axes, seasons, strict=True):
        cv = snapshot["seasons"][season]["cv"]
        lambdas = np.asarray(cv["lambdas"], dtype=float)
        for key, label, color, marker in series:
            if key in cv:
                ax.plot(lambdas, cv[key], color=color, marker=marker, markersize=4, label=label)
        ax.set_xscale("log")
        ax.set_xticks(lambdas, [f"{v:g}" for v in lambdas])
        ax.minorticks_off()
        ax.set_title(_season_label(season))
        ax.set_xlabel("Ridge penalty λ")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    axes[0].set_ylabel("Held-out squared error per 100")
    axes[-1].legend(loc="upper right")
    figure.suptitle(
        "Cross-validation with whole games held out and the prior rebuilt per fold",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=10.5,
    )
    return figure


@styled
def war_replacement_figure(snapshot: Snapshot, season: str | None = None) -> Figure:
    """Distribution of starters' linear WAR under each replacement definition."""
    season = season or _latest(snapshot)
    war = snapshot["seasons"][season]["war"]
    figure, ax = new_figure(3.0)
    everything = np.concatenate([np.asarray(v, dtype=float) for v in war["starters"].values()])
    bins = np.linspace(np.floor(everything.min()), np.ceil(everything.max()), 49)
    for definition, values in war["starters"].items():
        level = war["replacement_levels"][definition]
        headline = " (default)" if definition == war["headline_definition"] else ""
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=_DEFINITION_COLORS.get(definition, GREY),
            label=f"{_DEFINITION_LABELS.get(definition, definition)}, {level:+.1f}/100{headline}",
        )
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("Wins above replacement, linear, per starter-season")
    ax.set_ylabel("Starters")
    ax.set_title(f"The replacement level is a choice: starters' WAR, {_season_label(season)}")
    ax.legend(loc="upper right")
    return figure


@styled
def rating_precision_figure(snapshot: Snapshot, season: str | None = None) -> Figure:
    """Posterior SD of net rating by possessions played, with and without the prior."""
    season = season or _latest(snapshot)
    precision = snapshot["seasons"][season]["precision"]
    possessions = np.asarray(precision["possessions"], dtype=float)
    edges = np.unique(np.geomspace(max(possessions.min(), 50.0), possessions.max(), 17).round())
    centers = np.sqrt(edges[:-1] * edges[1:])
    figure, ax = new_figure(3.0)
    for key, label, color in (
        ("se_net_no_prior", "Shrunk toward zero", SKY),
        ("se_net", "Shrunk toward the box-score prior", BLUE),
    ):
        values = np.asarray(precision[key], dtype=float)
        low, mid, high = [], [], []
        for left, right in itertools.pairwise(edges):
            inside = values[(possessions >= left) & (possessions < right)]
            if inside.size == 0:
                low.append(np.nan)
                mid.append(np.nan)
                high.append(np.nan)
                continue
            q10, q50, q90 = np.quantile(inside, [0.1, 0.5, 0.9])
            low.append(q10)
            mid.append(q50)
            high.append(q90)
        ax.fill_between(centers, low, high, color=color, alpha=0.2, linewidth=0)
        ax.plot(centers, mid, color=color, linewidth=1.8, label=f"{label} (median, 10-90%)")
    ax.set_xscale("log")
    ticks = [t for t in (100, 300, 1000, 3000, 10000) if edges[0] <= t <= edges[-1]]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Possessions played (offense + defense)")
    ax.set_ylabel("Posterior SD of net rating, per 100")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Posterior SD of a player's net rating, {_season_label(season)}")
    ax.legend(loc="lower left")
    return figure


@styled
def revenue_bootstrap_figure(snapshot: Snapshot) -> Figure:
    """School-bootstrap distribution of log revenue per win, this season and two seasons."""
    economics = snapshot["economics"]
    figure, ax = new_figure(3.0)
    for key, label, color in (
        ("win_effect_current", "This season", BLUE),
        ("win_effect_two_season", "This season plus next", ORANGE),
    ):
        draws = np.asarray(economics["draws"][key], dtype=float) * 100
        summary = economics[key]
        low, high = (100 * v for v in summary["interval"])
        ax.hist(
            draws,
            bins=36,
            color=color,
            alpha=0.45,
            label=f"{label}: median {100 * summary['median']:.2f}%, 80% {low:.2f} to {high:.2f}",
        )
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("Change in reported men's basketball revenue per win, %")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title(
        f"Revenue per win: {economics['n_obs']:,} school-seasons from {economics['n_schools']} "
        f"schools, {economics['first_season']}-{economics['last_season']}"
    )
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper left")
    return figure


@styled
def example_roster_figure(snapshot: Snapshot) -> Figure:
    """Synthetic roster: annual program value against allocated price, with team medians."""
    roster = snapshot["synthetic"]["example_roster"]
    value = np.asarray(roster["program_value"], dtype=float) / 1e3
    price = np.asarray(roster["price"], dtype=float) / 1e3
    roles = [role or "unpaid" for role in roster["role"]]
    figure, ax = new_figure(3.4)
    for role, color, marker in (
        ("starter", BLUE, "o"),
        ("rotation", ORANGE, "s"),
        ("bench", PURPLE, "D"),
        ("unpaid", GREY, "x"),
    ):
        chosen = np.array([r == role for r in roles])
        if chosen.any():
            ax.scatter(price[chosen], value[chosen], s=42, color=color, marker=marker, label=role)
    value_cuts = [v for v in roster["value_cut"] if v is not None]
    price_cuts = [v for v in roster["price_cut"] if v is not None]
    if value_cuts and price_cuts:
        ax.axhline(value_cuts[0] / 1e3, color=GREEN, linewidth=0.9, linestyle="--")
        ax.axvline(price_cuts[0] / 1e3, color=GREEN, linewidth=0.9, linestyle="--")
        ax.text(
            0.99,
            value_cuts[0] / 1e3,
            "team median value ",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color=GREEN,
        )
        ax.text(
            price_cuts[0] / 1e3,
            0.02,
            " team median price",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8,
            color=GREEN,
        )
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Allocated price (share of a published tier budget), median")
    ax.set_ylabel("Annual program value, median")
    ax.set_title("A synthetic roster: value and price are different quantities")
    ax.legend(loc="upper left")
    return figure
