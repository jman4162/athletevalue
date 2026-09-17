"""Build ``docs/_static/data/snapshot.json``, the evidence behind the docs and figures.

Fits the requested seasons from live data (or the cache), runs every validation gate
including the extended ones, and adds sections computed on synthetic data with known
truth: interval coverage and the example roster. The output holds no player names or
ids; the script checks every string against the names in the fitted seasons before
writing.

    uv run python scripts/build_snapshot.py                 # seasons 2025 and 2026
    uv run python scripts/build_snapshot.py --seasons 2026 --offline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixtures.example_program import example_player, example_program  # noqa: E402
from tests.fixtures.raw_season import make_raw_season  # noqa: E402

from athletevalue.api import Session  # noqa: E402
from athletevalue.market.model import fit_market_model  # noqa: E402
from athletevalue.sources.cache import (  # noqa: E402
    ArtifactCache,
    ArtifactMismatchError,
    SourceUnavailableError,
)
from athletevalue.sources.torvik import TorvikClient  # noqa: E402
from athletevalue.valuation.checks import player_war_table, returning_players  # noqa: E402
from athletevalue.valuation.player import team_table  # noqa: E402
from athletevalue.valuation.season import assemble_season  # noqa: E402
from athletevalue.valuation.snapshot import (  # noqa: E402
    SCHEMA_VERSION,
    assert_anonymous,
    cv_curves,
    economics_payload,
    gates_payload,
    normal_interval_coverage,
    rating_precision,
    returning_payload,
    round_floats,
    team_ratings_payload,
    war_payload,
)
from athletevalue.valuation.team import team_draws  # noqa: E402
from athletevalue.versions import MODEL_VERSION, __version__  # noqa: E402

OUTPUT = ROOT / "docs" / "_static" / "data" / "snapshot.json"
SIGNIFICANT_DIGITS = 5
COVERAGE_SEEDS = range(10)
CV_PLUS_REPLICATIONS = 20


def season_section(session: Session, season: int) -> tuple[dict[str, Any], set[str]]:
    model = session.fit_season(season)
    report = session.validate(model, extended=True)
    try:
        torvik = TorvikClient(session.cache).team_results(season)
    except (SourceUnavailableError, ArtifactMismatchError):
        torvik = None
    players = player_war_table(model)
    section = {
        "data_through": model.data_date.isoformat(),
        "prior_training_seasons": list(model.prior.train_seasons) if model.prior else [],
        "lambda": model.rapm.lam,
        "intercept": model.rapm.intercept,
        "home_court": model.rapm.home_court,
        "exponent": model.exponent,
        "margin_sd": model.margin_sd,
        "n_possessions": model.rapm.n_possessions,
        "n_players_rated": model.rapm.table.filter(~pl.col("pooled")).height,
        "passed": report.passed,
        "gates": gates_payload(report),
        "notes": list(report.notes),
        "cv": cv_curves(model, session.registry),
        "teams": team_ratings_payload(model, torvik),
        "precision": rating_precision(model),
        "war": war_payload(model, players),
    }
    names = set(model.rapm.table["name"].drop_nulls().to_list())
    return section, names


def synthetic_section() -> dict[str, Any]:
    season, economics, registry = example_program()
    level = registry.get("mbb.wins.interval_level").scalar()
    team = str(example_player(season)["team"])
    roster = team_table(team_draws(season, economics, registry, team, seed=0))
    roster_payload = {
        column: roster[column].to_list()
        for column in (
            "role",
            "possession_share",
            "net",
            "war",
            "program_value",
            "price",
            "surplus",
            "quadrant",
            "value_cut",
            "price_cut",
        )
    }

    coverage = []
    players = 0
    for seed in COVERAGE_SEEDS:
        frames, truth = make_raw_season(seed=100 + seed, season=2026)
        fitted = assemble_season(frames, registry).rapm.table.filter(~pl.col("pooled"))
        players += fitted.height
        coverage.append(
            normal_interval_coverage(
                fitted["net"].to_numpy(),
                fitted["se_net"].to_numpy(),
                np.array([truth[a] for a in fitted["athlete_id"]]),
                level,
            )
        )

    cv_plus = []
    for replication in range(CV_PLUS_REPLICATIONS):
        rng = np.random.default_rng(replication)
        X = rng.normal(size=(2240, 3))
        schools = rng.integers(0, 24, 2240)
        tiers = schools % 3
        log_pay = 12.0 + 0.8 * X[:, 0] - 0.4 * X[:, 1] - 0.5 * tiers + rng.normal(0, 0.3, 2240)
        train = slice(0, 240)
        model = fit_market_model(
            X[train],
            log_pay[train],
            schools[train],
            tiers[train],
            features=("a", "b", "c"),
            ridge_grid=(1.0,),
            min_labels=40,
            min_schools=10,
        )
        lower, upper = model.interval_log(X[240:], level)
        cv_plus.append(float(np.mean((log_pay[240:] >= lower) & (log_pay[240:] <= upper))))

    return {
        "note": "Synthetic seasons and labels with known truth; no real player or program.",
        "interval_level": level,
        "example_roster": roster_payload,
        "rapm_normal_interval_coverage": {
            "seasons": len(coverage),
            "players": players,
            "mean": float(np.mean(coverage)),
            "min": float(np.min(coverage)),
            "max": float(np.max(coverage)),
        },
        "cv_plus_interval_coverage": {
            "replications": len(cv_plus),
            "labels": 240,
            "schools": 24,
            "mean": float(np.mean(cv_plus)),
            "min": float(np.min(cv_plus)),
            "max": float(np.max(cv_plus)),
        },
    }


def dumps(payload: dict[str, Any]) -> str:
    """Indented JSON with each numeric list on one line, so diffs stay readable."""
    text = json.dumps(payload, indent=2, sort_keys=True)
    number_list = re.compile(r"\[\s*((?:-?[\d.eE+-]+|null)(?:,\s*(?:-?[\d.eE+-]+|null))*)\s*\]")
    return number_list.sub(lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    session = Session(cache=ArtifactCache.default(offline=args.offline))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "model_version": MODEL_VERSION,
        "seasons": {},
    }
    names: set[str] = set()
    for season in sorted(args.seasons):
        section, season_names = season_section(session, season)
        payload["seasons"][str(season)] = section
        names |= season_names
        print(f"{season}: {'all gates pass' if section['passed'] else 'SOME GATES FAIL'}")

    last = max(args.seasons)
    registry = session.registry
    payload["returning_players"] = returning_payload(
        returning_players(
            session.fit_season(last - 1),
            session.fit_season(last),
            min_possessions_earlier=int(
                registry.get("mbb.prior.returning_min_possessions_earlier").scalar()
            ),
            min_possessions_later=int(
                registry.get("mbb.prior.returning_min_possessions_later").scalar()
            ),
        )
    )
    economics = session.fit_economics(last)
    payload["economics"] = economics_payload(economics, registry)
    payload["economics"]["notes"] = list(economics.notes)
    payload["synthetic"] = synthetic_section()

    rounded = round_floats(payload, SIGNIFICANT_DIGITS)
    assert_anonymous(rounded, forbidden_strings=frozenset(n for n in names if len(n) > 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps(rounded) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
