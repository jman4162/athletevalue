"""Numbers quoted in the README and METHODOLOGY match the committed snapshot.

Prose stays readable on GitHub and PyPI, and a figure that drifts from the evidence
fails here. Rebuild the snapshot with ``scripts/build_snapshot.py`` and update the
text in the same change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).parents[2]
SNAPSHOT = json.loads((ROOT / "docs/_static/data/snapshot.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
METHODOLOGY = (ROOT / "METHODOLOGY.md").read_text(encoding="utf-8")


def season(year: int) -> dict:
    return SNAPSHOT["seasons"][str(year)]


def gate(year: int, name: str) -> dict:
    return next(g for g in season(year)["gates"] if g["gate"] == name)


def count(detail: str) -> int:
    return int(re.match(r"(\d+)", detail).group(1))


def num(value: float, spec: str) -> str:
    """Format like the docs: thousands separators, typographic minus."""
    return format(value, spec).replace("-", "−")


def cv_at(year: int, curve: str, lam: float) -> float:
    cv = season(year)["cv"]
    return cv[curve][cv["lambdas"].index(lam)]


def readme_expectations() -> list[str]:
    ref = [gate(y, "spearman_vs_reference_rapm") for y in (2025, 2026)]
    torvik = [gate(y, "spearman_team_net_vs_torvik") for y in (2025, 2026)]
    returning = SNAPSHOT["returning_players"]
    coverage = SNAPSHOT["synthetic"]["rapm_normal_interval_coverage"]["mean"]
    return [
        f"| {ref[0]['value']:.3f} ({count(ref[0]['detail']):,}) "
        f"| {ref[1]['value']:.3f} ({count(ref[1]['detail']):,}) |",
        f"| {torvik[0]['value']:.3f} ({count(torvik[0]['detail'])} teams) "
        f"| {torvik[1]['value']:.3f} ({count(torvik[1]['detail'])} teams) |",
        "| {:.3f} | {:.3f} | ≤ 1.00 |".format(
            *(gate(y, "prior_cv_error_ratio")["value"] for y in (2025, 2026))
        ),
        "| {:.1f} | {:.1f} | 95–112 |".format(
            *(gate(y, "intercept")["value"] for y in (2025, 2026))
        ),
        "| {:.2f} | {:.2f} | 1–4 |".format(*(gate(y, "home_court")["value"] for y in (2025, 2026))),
        "| {:,.0f} | {:,.0f} |".format(*(gate(y, "sigma2")["value"] for y in (2025, 2026))),
        "| {:.2f} | {:.2f} | 10–12.5 |".format(
            *(gate(y, "margin_sd")["value"] for y in (2025, 2026))
        ),
        "| {:.2f} | {:.2f} | ≥ 0.70 |".format(
            *(gate(y, "team_war_vs_wins_correlation")["value"] for y in (2025, 2026))
        ),
        "| {:+.3f} | {:+.3f} | ≥ 0 |".format(
            *(gate(y, "returning_player_prior_gain")["value"] for y in (2025, 2026))
        ),
        "| {:.2f} | {:.2f} | ≤ 0.10 |".format(
            *(gate(y, "revenue_win_effect_nonpositive_share")["value"] for y in (2025, 2026))
        ),
        "| {:.3f} | {:.3f} | ≤ 0.05 |".format(
            *(gate(y, "bid_calibration_max_decile_gap")["value"] for y in (2025, 2026))
        ),
        "| {:,.1f} | {:,.1f} |".format(*(cv_at(y, "no_prior", 1000.0) for y in (2025, 2026))),
        "| {:,.1f} | {:,.1f} |".format(
            *(cv_at(y, "zero_box_team_adjusted", 3000.0) for y in (2025, 2026))
        ),
        "| {:,.1f} | {:,.1f} |".format(
            *(cv_at(y, "box_team_adjusted", 3000.0) for y in (2025, 2026))
        ),
        f"On {returning['n_players']:,} players who appear in both 2025 and 2026",
        f"correlates {returning['corr_with_prior']:.2f} with the 2026",
        f"against {returning['corr_without_prior']:.2f} without it",
        "pooled low-minute players rate about "
        + num(round(season(2026)["war"]["replacement_levels"]["pooled"]), "d"),
        f"cover {coverage:.0%} of true values",
    ]


def methodology_expectations() -> list[str]:
    economics = SNAPSHOT["economics"]
    levels = [season(y)["war"]["replacement_levels"] for y in (2025, 2026)]

    def effect(key: str, digits: int) -> str:
        item = economics[key]
        low, high = item["interval"]
        return f"| {item['median']:.{digits}f} | {low:.{digits}f} to {high:.{digits}f} |"

    torvik = season(2026)["teams"]["torvik"]
    slope_none = np.polyfit(torvik["adj_em"], torvik["adj_net_no_prior"], 1)[0]
    slope_prior = np.polyfit(torvik["adj_em"], torvik["adj_net"], 1)[0]
    rho_none = spearmanr(torvik["adj_em"], torvik["adj_net_no_prior"])[0]
    rho_prior = gate(2026, "spearman_team_net_vs_torvik")["value"]
    returning = SNAPSHOT["returning_players"]
    return [
        f"| `pooled` | {num(levels[0]['pooled'], '.1f')} | {num(levels[1]['pooled'], '.1f')} |",
        f"| `bench_median` (default) | {num(levels[0]['bench_median'], '.1f')} "
        f"| {num(levels[1]['bench_median'], '.1f')} |",
        effect("win_effect_current", 4),
        effect("win_effect_two_season", 4),
        effect("bid_effect_two_season", 3),
        f"The fit uses {economics['n_obs']:,} school-seasons from {economics['n_schools']} schools",
        f"the data give {economics['units_per_bid']:.2f} units per bid",
        "| 2026 held-out error | "
        + " | ".join(
            f"{cv_at(2026, 'box_team_adjusted', lam):,.1f}"
            for lam in (300.0, 1000.0, 3000.0, 10000.0)
        )
        + " |",
        "| 2025 held-out error | "
        + " | ".join(
            f"{cv_at(2025, 'box_team_adjusted', lam):,.1f}"
            for lam in (300.0, 1000.0, 3000.0, 10000.0)
        )
        + " |",
        f"against {cv_at(2026, 'no_prior', 1000.0):,.1f} (2026) and "
        f"{cv_at(2025, 'no_prior', 1000.0):,.1f} (2025) with no prior",
        f"team adjustment reaches {cv_at(2026, 'zero_box_team_adjusted', 3000.0):,.1f} (2026) and "
        f"{cv_at(2025, 'zero_box_team_adjusted', 3000.0):,.1f} (2025)",
        f"Torvik AdjEM {slope_none:.2f} without a prior, {slope_prior:.2f} with",
        f"Torvik at Spearman {rho_none:.3f} without a prior and {rho_prior:.3f} with the adjusted",
        f"on {returning['n_players']:,} players rated in both 2025 and 2026",
        "weighted error "
        + " / ".join(f"{v:,.0f}" for v in season(2026)["cv"]["no_prior"])
        + " at λ = "
        + " / ".join(f"{lam:g}" for lam in season(2026)["cv"]["lambdas"]),
    ]


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


@pytest.mark.parametrize("expected", readme_expectations())
def test_readme_quotes_the_snapshot(expected):
    assert _flat(expected) in _flat(README), expected


@pytest.mark.parametrize("expected", methodology_expectations())
def test_methodology_quotes_the_snapshot(expected):
    assert _flat(expected) in _flat(METHODOLOGY), expected
