"""Everything fitted for one season, assembled from already-loaded frames."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.constants import PER_100
from athletevalue.frames.box import player_box_totals
from athletevalue.frames.games import SeasonGames, build_season_games, combined_game_map
from athletevalue.frames.possessions import LineupData, build_lineup_data
from athletevalue.frames.team_possessions import team_possession_totals
from athletevalue.identity.people import with_person_ids
from athletevalue.impact.cv import CvResult, PredictionsFor, cv_lambda, cv_with_prior, game_folds
from athletevalue.impact.design import Design
from athletevalue.impact.prior import BoxPriorModel, prior_offset, team_adjust
from athletevalue.impact.rapm import RapmResult, fit_rapm
from athletevalue.impact.reconstruct import team_ratings
from athletevalue.schemas.source import SourceReference
from athletevalue.wins.pythag import fit_exponent
from athletevalue.wins.war import GameContext, PlayerImpact, TeamContext, fit_margin_sd


@dataclass(frozen=True)
class SeasonFrames:
    season: int
    possessions: pl.DataFrame
    team_ids: pl.DataFrame
    schedule: pl.DataFrame
    espn_schedule: pl.DataFrame
    player_box: pl.DataFrame | None = None
    sources: tuple[SourceReference, ...] = field(default=())
    people: pl.DataFrame | None = None
    """athlete_id to person_id, for linking seasons; see ``identity.people``."""


@dataclass(frozen=True)
class SeasonModel:
    season: int
    rapm: RapmResult
    """Headline ratings: with the box-score prior when one was supplied."""
    baseline: RapmResult
    """Ratings shrunk toward zero; compared with the published reference RAPM."""
    prior: BoxPriorModel | None
    design: Design
    prior_offset: NDArray[np.float64] | None
    lineups: LineupData
    box_predictions: pl.DataFrame | None
    """Unadjusted box-model predictions from the full season's box scores."""
    player_box: pl.DataFrame | None
    """Per-game box scores, kept so validation can rebuild the prior from training games."""
    teams: pl.DataFrame
    """team, conference, games, wins, losses, ncaa_bid, ncaa_games, ncaa_wins, ortg,
    drtg, pace, adj_off, adj_def, adj_net, adj_net_no_prior, lineup_poss,
    n_conference_members. ``adj_net_no_prior`` comes from the ratings shrunk toward zero."""
    games: pl.DataFrame
    """contest_id, home, away, home_score, away_score, neutral, ncaa_tournament, possessions."""
    exponent: float
    margin_sd: float
    replacement: float
    """Net rating of a replacement player, under ``replacement_definition``."""
    replacement_definition: str
    replacement_levels: dict[str, float]
    """Every replacement definition the registry knows, for sensitivity reporting."""
    lineup_counts: dict[str, int]
    data_date: date
    sources: tuple[SourceReference, ...]
    notes: tuple[str, ...] = ()

    @property
    def worst_d1_net(self) -> float:
        return float(self.teams["adj_net"].min())  # type: ignore[arg-type]

    def non_d1_opponent_net(self, floor_at_worst_d1: bool) -> float:
        """Rating used for a non-D1 opponent in the win model."""
        if floor_at_worst_d1:
            return max(self.rapm.non_d1_net, self.worst_d1_net)
        return self.rapm.non_d1_net

    def fold_predictions(self) -> PredictionsFor | None:
        """Box predictions rebuilt from a subset of games, for leak-free cross-validation."""
        if self.prior is None or self.player_box is None:
            return None
        return fold_predictions(self.prior, self.player_box, self.design)

    def team_context(self, team: str) -> TeamContext:
        row = self._team_row(team)
        return TeamContext(
            team=team,
            games=int(row["games"]),
            ortg=float(row["ortg"]),
            drtg=float(row["drtg"]),
            pace=float(row["pace"]),
            adj_net=float(row["adj_net"]),
            lineup_poss=float(row["lineup_poss"]),
        )

    def schedule(self, team: str, *, floor_non_d1: bool = True) -> list[GameContext]:
        nets = dict(zip(self.teams["team"], self.teams["adj_net"], strict=True))
        non_d1 = self.non_d1_opponent_net(floor_non_d1)
        out: list[GameContext] = []
        for game in self.games.filter(
            ((pl.col("home") == team) | (pl.col("away") == team))
            & pl.col("possessions").is_not_null()
        ).iter_rows(named=True):
            at_home = game["home"] == team
            opponent = game["away"] if at_home else game["home"]
            venue = 0 if game["neutral"] else (1 if at_home else -1)
            out.append(
                GameContext(
                    opponent_net=float(nets.get(opponent, non_d1)),
                    venue=venue,
                    possessions=float(game["possessions"]),
                )
            )
        return out

    def player_impact(self, athlete_id: str) -> PlayerImpact:
        row = self.player_row(athlete_id)
        return PlayerImpact(
            net=float(row["net"]),
            se_net=float(row["se_net"]),
            player_poss=float(row["off_poss"]) + float(row["def_poss"]),
        )

    def player_row(self, athlete_id: str) -> dict[str, Any]:
        rows = self.rapm.table.filter(pl.col("athlete_id") == athlete_id)
        if rows.is_empty():
            raise KeyError(f"athlete {athlete_id} has no possessions in {self.season}")
        return rows.row(0, named=True)

    def roster(self, team: str) -> pl.DataFrame:
        """The team's players in athlete-id order.

        Ranking by playing time breaks ties by row order. The rating table is sorted by
        rating, whose last bits vary with BLAS threading, so a fixed order keeps
        starters, roles and the bench from changing between runs.
        """
        return self.rapm.table.filter(pl.col("team") == team).sort("athlete_id")

    def _team_row(self, team: str) -> dict[str, Any]:
        rows = self.teams.filter(pl.col("team") == team)
        if rows.is_empty():
            raise KeyError(f"{team} is not a D1 team with possessions in {self.season}")
        return rows.row(0, named=True)


def season_lineups(
    frames: SeasonFrames, registry: AssumptionRegistry
) -> tuple[SeasonGames, LineupData]:
    """Game results and lineup rows, with neutral sites resolved."""
    game_map = combined_game_map(frames.possessions, frames.schedule, frames.espn_schedule)
    season_games = build_season_games(
        frames.schedule, frames.espn_schedule, frames.team_ids, game_map
    )
    neutral = frozenset(season_games.games.filter("neutral")["contest_id"].to_list())
    lineups = build_lineup_data(
        frames.possessions,
        d1_teams=frozenset(frames.team_ids["team"].to_list()),
        neutral_contests=neutral,
        drop_garbage_time=registry.get("mbb.impact.drop_garbage_time").flag(),
    )
    return season_games, lineups


def baseline_rapm(frames: SeasonFrames, registry: AssumptionRegistry) -> RapmResult:
    """Ratings shrunk toward zero with the registry penalty; the box prior's training target."""
    _, lineups = season_lineups(frames, registry)
    result, _, _ = fit_rapm(
        lineups,
        season=frames.season,
        lam=registry.get("mbb.impact.ridge_lambda").scalar(),
        min_possessions=registry.get("mbb.impact.min_possessions").scalar(),
    )
    return result


def fold_predictions(
    prior: BoxPriorModel, player_box: pl.DataFrame, design: Design
) -> PredictionsFor:
    """A function from a training-row mask to box predictions built from those games only."""
    contests = np.asarray(design.contest_ids)
    box = player_box.with_columns(pl.col("contest_id").cast(pl.Utf8))

    def predictions_for(train: NDArray[np.bool_]) -> pl.DataFrame:
        games = pl.Series(sorted(set(contests[np.unique(design.groups[train])])), dtype=pl.Utf8)
        return prior.predict(
            player_box_totals(box.filter(pl.col("contest_id").is_in(games.implode())))
        )

    return predictions_for


def assemble_season(
    frames: SeasonFrames,
    registry: AssumptionRegistry,
    *,
    lam: float | None = None,
    choose_lambda_by_cv: bool = False,
    seed: int = 0,
    prior: BoxPriorModel | None = None,
    notes: tuple[str, ...] = (),
) -> SeasonModel:
    """Fit one season. With *prior* and box scores in *frames*, ratings shrink toward it."""
    season = frames.season
    season_games, lineups = season_lineups(frames, registry)
    min_poss = registry.get("mbb.impact.min_possessions").scalar()

    baseline_lam = registry.get("mbb.impact.ridge_lambda").scalar()
    baseline, design, baseline_fit = fit_rapm(
        lineups, season=season, lam=baseline_lam, min_possessions=min_poss
    )
    baseline_design = design
    predictions = None
    box_predictions = None
    offset = None
    if prior is not None and frames.player_box is not None:
        box_predictions = prior.predict(player_box_totals(frames.player_box))
        predictions = box_predictions
        if registry.get("mbb.prior.team_adjustment").flag():
            predictions = team_adjust(box_predictions, lineups, design, baseline_fit.beta)
        offset = prior_offset(design, predictions)
    headline_lam = registry.get(
        "mbb.impact.ridge_lambda" if offset is None else "mbb.impact.ridge_lambda_with_prior"
    ).scalar()
    if lam is not None:
        headline_lam = lam

    cv: CvResult | None = None
    if choose_lambda_by_cv:
        grid = registry.get("mbb.impact.lambda_grid").numbers()
        n_folds = int(registry.get("mbb.impact.cv_folds").scalar())
        if offset is None or prior is None or frames.player_box is None:
            cv = cv_lambda(design, grid, n_folds=n_folds, rng=np.random.default_rng(seed))
        else:
            # The prior for each fold is rebuilt from that fold's training games; a
            # full-season prior would let held-out outcomes into the shrinkage target.
            _, cv = cv_with_prior(
                design,
                lineups,
                game_folds(design.groups, n_folds, np.random.default_rng(seed)),
                grid,
                lam_baseline=baseline_lam,
                adjust_to_team=registry.get("mbb.prior.team_adjustment").flag(),
                predictions_for=fold_predictions(prior, frames.player_box, design),
            )
        headline_lam = cv.best
    if offset is None and headline_lam == baseline_lam and cv is None:
        rapm, fit = baseline, baseline_fit
    else:
        rapm, design, fit = fit_rapm(
            lineups,
            season=season,
            lam=headline_lam,
            min_possessions=min_poss,
            cv=cv,
            prior=predictions,
        )

    ratings = (
        team_ratings(lineups, design, fit)
        .with_columns((pl.col("lineup_off_poss") + pl.col("lineup_def_poss")).alias("lineup_poss"))
        .join(
            team_ratings(lineups, baseline_design, baseline_fit).select(
                "team", pl.col("adj_net").alias("adj_net_no_prior")
            ),
            on="team",
            how="left",
        )
    )
    totals = team_possession_totals(frames.possessions)
    members = frames.team_ids.group_by("conference").agg(pl.len().alias("n_conference_members"))
    teams = (
        season_games.teams.join(
            totals.select("team", "ortg", "drtg", "pace"), on="team", how="inner"
        )
        .join(
            ratings.select(
                "team", "adj_off", "adj_def", "adj_net", "adj_net_no_prior", "lineup_poss"
            ),
            on="team",
            how="inner",
        )
        .join(members, on="conference", how="left")
        .filter(pl.col("games") > 0)
    )
    exponent = fit_exponent(
        teams["ortg"].to_numpy(),
        teams["drtg"].to_numpy(),
        (teams["wins"] / teams["games"]).cast(pl.Float64).to_numpy(),
        teams["games"].cast(pl.Float64).to_numpy(),
        bounds=registry.get("mbb.wins.exponent_search_bounds").interval(),
    )

    game_poss = frames.possessions.group_by("contest_id").agg((pl.len() / 2).alias("possessions"))
    games = season_games.games.join(game_poss, on="contest_id", how="left")
    margin_sd = _margin_sd(
        games, teams, rapm, d1_only=registry.get("mbb.wins.margin_sd_d1_only").flag()
    )
    levels = replacement_levels(rapm, registry)
    definition = registry.get("mbb.wins.replacement_definition").text()
    if definition not in levels:
        raise ValueError(
            f"unknown replacement definition {definition!r}; choose from {sorted(levels)}"
        )
    rapm = replace(rapm, table=with_person_ids(rapm.table, frames.people))
    baseline = replace(baseline, table=with_person_ids(baseline.table, frames.people))
    return SeasonModel(
        season=season,
        rapm=rapm,
        baseline=baseline,
        prior=prior if offset is not None else None,
        design=design,
        prior_offset=offset,
        lineups=lineups,
        box_predictions=box_predictions,
        player_box=frames.player_box,
        teams=teams,
        games=games,
        exponent=exponent,
        margin_sd=margin_sd,
        replacement=levels[definition],
        replacement_definition=definition,
        replacement_levels=levels,
        lineup_counts=lineups.counts,
        data_date=_last_game_date(frames.schedule),
        sources=frames.sources,
        notes=notes,
    )


def replacement_levels(rapm: RapmResult, registry: AssumptionRegistry) -> dict[str, float]:
    """Net rating of a replacement player under each definition the registry offers.

    ``nba_convention`` is the Box Plus/Minus constant. ``pooled`` is the fitted
    coefficient of the players below the modelling threshold, the marginal D1 player
    in these data. ``bench_median`` is the possession-weighted median rating of the
    players ranked in the bench range by playing time on each team: the player a
    coach actually turns to when a rotation player is unavailable.
    """
    low, high = registry.get("mbb.wins.bench_rank_range").interval()
    # Sorted by id so ties in playing time break the same way on every run.
    table = rapm.table.sort("athlete_id").with_columns(
        (pl.col("off_poss") + pl.col("def_poss")).alias("_poss")
    )
    ranked = table.with_columns(
        pl.col("_poss").rank(method="ordinal", descending=True).over("team").alias("_rank")
    ).filter((pl.col("_rank") >= low) & (pl.col("_rank") <= high) & (pl.col("_poss") > 0))
    if ranked.is_empty():
        bench = rapm.pool_net
    else:
        order = ranked.sort("net")
        weights = order["_poss"].cast(pl.Float64).to_numpy()
        cumulative = np.cumsum(weights) / weights.sum()
        bench = float(order["net"].to_numpy()[int(np.searchsorted(cumulative, 0.5))])
    return {
        "nba_convention": registry.get("mbb.wins.replacement_level").scalar(),
        "pooled": float(rapm.pool_net),
        "bench_median": bench,
    }


def _margin_sd(
    games: pl.DataFrame, teams: pl.DataFrame, rapm: RapmResult, *, d1_only: bool
) -> float:
    """Root-mean-square gap between actual and expected margins.

    With *d1_only*, games against non-D1 opponents are left out: their opponent
    rating is one shared coefficient, so their margins are not something the model
    can be expected to predict, and they would inflate the spread used for every
    close D1 game.
    """
    nets = teams.select("team", "adj_net")
    frame = (
        games.filter(pl.col("possessions").is_not_null())
        .join(nets.rename({"team": "home", "adj_net": "home_net"}), on="home", how="left")
        .join(nets.rename({"team": "away", "adj_net": "away_net"}), on="away", how="left")
    )
    if d1_only:
        frame = frame.filter(pl.col("home_net").is_not_null() & pl.col("away_net").is_not_null())
    frame = frame.with_columns(
        pl.col("home_net").fill_null(rapm.non_d1_net),
        pl.col("away_net").fill_null(rapm.non_d1_net),
        pl.when(pl.col("neutral")).then(0.0).otherwise(1.0).alias("venue"),
    ).with_columns(
        (
            (pl.col("home_net") - pl.col("away_net") + 2 * rapm.home_court * pl.col("venue"))
            * pl.col("possessions")
            / PER_100
        ).alias("expected"),
        (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).alias("actual"),
    )
    return fit_margin_sd(frame["actual"].to_numpy(), frame["expected"].to_numpy())


def _last_game_date(schedule: pl.DataFrame) -> date:
    parsed = schedule.select(
        pl.col("game_date").str.strptime(pl.Date, "%m/%d/%Y", strict=False).max()
    ).item()
    if isinstance(parsed, date):
        return parsed
    return datetime.now().date()
