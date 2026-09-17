"""MkDocs hook: render the validation snapshot as the Validation page."""

from __future__ import annotations

import json
from pathlib import Path

from mkdocs.structure.files import File, Files

SNAPSHOT = Path(__file__).resolve().parents[1] / "_static" / "data" / "snapshot.json"

FIGURES = (
    ("team_ratings_vs_torvik", "Team net rating against Torvik, without and with the prior"),
    ("cv_error_by_penalty", "Held-out error by ridge penalty"),
    ("rating_precision", "Posterior SD of net rating by possessions played"),
    ("war_by_replacement", "Starters' WAR under the three replacement definitions"),
    ("revenue_per_win", "Bootstrap distribution of revenue per win"),
    ("example_roster", "A synthetic roster: program value against allocated price"),
)


def _bound(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4g}"


def render(snapshot: dict) -> str:
    lines = [
        "# Validation",
        "",
        "Generated from `docs/_static/data/snapshot.json`, which "
        "`scripts/build_snapshot.py` builds from the pinned upstream files. Model "
        f"`{snapshot['model_version']}`, package {snapshot['package_version']}.",
        "",
    ]
    for season, section in sorted(snapshot["seasons"].items()):
        year = int(season)
        lines += [
            f"## {year - 1}-{str(year)[2:]} season",
            "",
            f"Games through {section['data_through']}; {section['n_possessions']:,} possessions; "
            f"{section['n_players_rated']:,} players with their own rating; box prior trained "
            f"on {', '.join(map(str, section['prior_training_seasons'])) or 'no season'}.",
            "",
            "| Gate | Value | Low | High | Result | Detail |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for gate in section["gates"]:
            result = "pass" if gate["passed"] else "**fail**"
            lines.append(
                f"| `{gate['gate']}` | {_number(gate['value'])} | {_bound(gate['low'])} | "
                f"{_bound(gate['high'])} | {result} | {gate['detail']} |"
            )
        lines += ["", *(f"- {note}" for note in section["notes"]), ""]

    returning = snapshot.get("returning_players")
    if returning:
        lines += [
            "## Returning players",
            "",
            f"{returning['n_players']:,} players with at least "
            f"{returning['min_possessions_earlier']} offensive possessions in "
            f"{returning['earlier_season']} and {returning['min_possessions_later']} in "
            f"{returning['later_season']}, linked by person id. Correlation of the earlier "
            f"rating with the later no-prior rating: {returning['corr_with_prior']:.3f} with "
            f"the prior, {returning['corr_without_prior']:.3f} without.",
            "",
        ]

    economics = snapshot["economics"]
    level = economics["interval_level"]
    lines += [
        "## Program economics",
        "",
        f"{economics['n_obs']:,} school-seasons from {economics['n_schools']} schools, "
        f"{economics['first_season']}-{economics['last_season']}; "
        f"{economics['units_per_bid']:.2f} tournament units per bid.",
        "",
        f"| Effect on log revenue | Median | {level:.0%} interval |",
        "| --- | --- | --- |",
    ]
    for key, label in (
        ("win_effect_current", "One win, this season"),
        ("win_effect_two_season", "One win, this and next season"),
        ("bid_effect_current", "NCAA bid, this season"),
        ("bid_effect_two_season", "NCAA bid, this and next season"),
    ):
        item = economics[key]
        low, high = item["interval"]
        lines.append(f"| {label} | {item['median']:.4f} | {low:.4f} to {high:.4f} |")
    lines += ["", *(f"- {note}" for note in economics.get("notes", [])), ""]

    synthetic = snapshot["synthetic"]
    rapm = synthetic["rapm_normal_interval_coverage"]
    cv_plus = synthetic["cv_plus_interval_coverage"]
    lines += [
        "## Interval coverage on synthetic data",
        "",
        synthetic["note"],
        "",
        "| Interval | Nominal | Mean coverage | Range | Runs |",
        "| --- | --- | --- | --- | --- |",
        f"| RAPM normal interval, no prior | {level:.0%} | {rapm['mean']:.1%} | "
        f"{rapm['min']:.1%} to {rapm['max']:.1%} | {rapm['seasons']} seasons |",
        f"| Market model CV+ | {level:.0%} | {cv_plus['mean']:.1%} | "
        f"{cv_plus['min']:.1%} to {cv_plus['max']:.1%} | {cv_plus['replications']} fits |",
        "",
        "## Figures",
        "",
    ]
    for stem, caption in FIGURES:
        lines += [f"![{caption}](_static/{stem}.svg)", ""]
    return "\n".join(lines)


def on_files(files: Files, config: dict) -> Files:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    files.append(File.generated(config, "validation.md", content=render(snapshot)))
    return files
