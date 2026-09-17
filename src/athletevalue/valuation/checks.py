"""Checks below the rating layer: wins, revenue, tournament bids and returning players.

``validate.validate_season`` compares ratings with published references. These
checks test the layers built on the ratings, where no published reference exists,
against the data's own outcomes: team wins, next season's ratings for the same
people, and observed bid rates. Each returns plain tables so the same numbers feed
the validation gates, the documentation snapshot and the figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.constants import DECILES, PLAYERS_ON_COURT
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.season import SeasonModel
from athletevalue.wins.war import wins_per_point_schedule


@dataclass(frozen=True)
class ReturningPlayers:
    """How well one season's ratings predict the next season's no-prior ratings."""

    earlier_season: int
    later_season: int
    n_players: int
    corr_without_prior: float
    corr_with_prior: float
    min_possessions_earlier: int
    min_possessions_later: int

    @property
    def gain(self) -> float:
        return self.corr_with_prior - self.corr_without_prior


def returning_players(
    earlier: SeasonModel,
    later: SeasonModel,
    *,
    min_possessions_earlier: int,
    min_possessions_later: int,
) -> ReturningPlayers | None:
    """Correlate *earlier* ratings, with and without its prior, with *later* no-prior ratings.

    Players are linked by ``person_id``. The later season is scored without a prior
    so the target does not share the box-score model being tested. ``None`` when
    either season lacks person ids or no player returns.
    """
    target = (
        later.baseline.table.filter(
            (pl.col("off_poss") >= min_possessions_later) & pl.col("person_id").is_not_null()
        )
        .group_by("person_id")
        .agg(pl.col("net").first().alias("net_later"), pl.len().alias("_ids"))
        .filter(pl.col("_ids") == 1)
    )
    correlations = []
    n_players = 0
    for table in (earlier.baseline.table, earlier.rapm.table):
        joined = (
            table.filter(
                (pl.col("off_poss") >= min_possessions_earlier) & pl.col("person_id").is_not_null()
            )
            .unique(subset=["person_id"], keep="none")
            .join(target, on="person_id", how="inner")
        )
        if joined.height < 2:
            return None
        n_players = joined.height
        correlations.append(float(np.corrcoef(joined["net"], joined["net_later"])[0, 1]))
    return ReturningPlayers(
        earlier_season=earlier.season,
        later_season=later.season,
        n_players=n_players,
        corr_without_prior=correlations[0],
        corr_with_prior=correlations[1],
        min_possessions_earlier=min_possessions_earlier,
        min_possessions_later=min_possessions_later,
    )


def player_war_table(model: SeasonModel, *, floor_non_d1: bool = True) -> pl.DataFrame:
    """Linear WAR of every rated player under each replacement definition.

    Columns: athlete_id, team, possession_share, starter, net, se_net, possessions,
    then ``war_<definition>`` for each definition in ``model.replacement_levels``.
    """
    frames = []
    for team in model.teams["team"].to_list():
        games = model.schedule(team, floor_non_d1=floor_non_d1)
        if not games:
            continue
        context = model.team_context(team)
        slope = wins_per_point_schedule(
            context, games, home_court=model.rapm.home_court, margin_sd=model.margin_sd
        )
        roster = model.roster(team).with_columns(
            (pl.col("off_poss") + pl.col("def_poss")).alias("possessions")
        )
        roster = roster.with_columns(
            (pl.col("possessions") / context.lineup_poss).alias("possession_share"),
            (pl.col("possessions").rank(method="ordinal", descending=True) <= PLAYERS_ON_COURT)
            .cast(pl.Boolean)
            .alias("starter"),
        )
        frames.append(
            roster.select(
                "athlete_id",
                "team",
                "possession_share",
                "starter",
                "net",
                "se_net",
                "possessions",
                *(
                    ((pl.col("net") - level) * pl.col("possession_share") * slope).alias(
                        f"war_{definition}"
                    )
                    for definition, level in model.replacement_levels.items()
                ),
            )
        )
    return pl.concat(frames)


def team_war_table(model: SeasonModel, players: pl.DataFrame) -> pl.DataFrame:
    """team, games, wins, war (sum under the headline definition) from ``player_war_table``."""
    column = f"war_{model.replacement_definition}"
    return (
        players.group_by("team")
        .agg(pl.col(column).sum().alias("war"))
        .join(model.teams.select("team", "games", "wins"), on="team", how="inner")
        .sort("team")
    )


def bid_calibration(
    economics: EconomicsModel, registry: AssumptionRegistry, *, bins: int = DECILES
) -> pl.DataFrame:
    """Predicted and observed bid rates by decile of predicted probability.

    Scored on the rows the bid model was fitted on, so this checks the logistic form,
    not out-of-sample accuracy.
    """
    excluded = [int(s) for s in registry.get("economics.excluded_seasons").numbers()]
    sample = economics.outcomes.filter(
        (pl.col("games") > 0)
        & ~pl.col("season").is_in(pl.Series(excluded, dtype=pl.Int64).implode())
    )
    win_pct = (sample["wins"] / sample["games"]).cast(pl.Float64).to_numpy()
    power = sample["conference"].is_in(sorted(economics.power_conferences)).to_numpy()
    predicted = np.where(
        power,
        economics.bids.probability(win_pct, True),
        economics.bids.probability(win_pct, False),
    )
    frame = pl.DataFrame(
        {"predicted": predicted, "observed": sample["ncaa_bid"].cast(pl.Float64).to_numpy()}
    )
    order = np.argsort(predicted, kind="stable")
    decile = np.empty(len(predicted), dtype=np.int64)
    decile[order] = np.arange(len(predicted)) * bins // len(predicted)
    return (
        frame.with_columns(pl.Series("decile", decile))
        .group_by("decile")
        .agg(
            pl.col("predicted").mean(),
            pl.col("observed").mean(),
            pl.len().alias("n"),
        )
        .sort("decile")
    )
