"""Everything fitted for one season, assembled from already-loaded frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.constants import PER_100
from athletevalue.frames.games import build_season_games, espn_game_map
from athletevalue.frames.possessions import build_lineup_data
from athletevalue.frames.team_possessions import team_possession_totals
from athletevalue.impact.cv import CvResult, cv_lambda
from athletevalue.impact.design import build_design
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
    sources: tuple[SourceReference, ...] = field(default=())


@dataclass(frozen=True)
class SeasonModel:
    season: int
    rapm: RapmResult
    teams: pl.DataFrame
    """team, conference, games, wins, losses, ncaa_bid, ncaa_games, ncaa_wins, ortg,
    drtg, pace, adj_off, adj_def, adj_net, lineup_poss, n_conference_members."""
    games: pl.DataFrame
    """contest_id, home, away, home_score, away_score, neutral, ncaa_tournament, possessions."""
    exponent: float
    margin_sd: float
    lineup_counts: dict[str, int]
    data_date: date
    sources: tuple[SourceReference, ...]

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

    def schedule(self, team: str) -> list[GameContext]:
        nets = dict(zip(self.teams["team"], self.teams["adj_net"], strict=True))
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
                    opponent_net=float(nets.get(opponent, self.rapm.non_d1_net)),
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
        return self.rapm.table.filter(pl.col("team") == team)

    def _team_row(self, team: str) -> dict[str, Any]:
        rows = self.teams.filter(pl.col("team") == team)
        if rows.is_empty():
            raise KeyError(f"{team} is not a D1 team with possessions in {self.season}")
        return rows.row(0, named=True)


def assemble_season(
    frames: SeasonFrames,
    registry: AssumptionRegistry,
    *,
    lam: float | None = None,
    choose_lambda_by_cv: bool = False,
    seed: int = 0,
) -> SeasonModel:
    season = frames.season
    game_map = espn_game_map(frames.possessions)
    season_games = build_season_games(
        frames.schedule, frames.espn_schedule, frames.team_ids, game_map
    )
    neutral = frozenset(
        game_map.join(season_games.games.filter("neutral").select("contest_id"), on="contest_id")[
            "espn_game_id"
        ].to_list()
    )
    lineups = build_lineup_data(
        frames.possessions,
        d1_teams=frozenset(frames.team_ids["team"].to_list()),
        neutral_espn_games=neutral,
        drop_garbage_time=registry.get("mbb.impact.drop_garbage_time").flag(),
    )
    min_poss = registry.get("mbb.impact.min_possessions").scalar()

    cv: CvResult | None = None
    penalty = registry.get("mbb.impact.ridge_lambda").scalar() if lam is None else lam
    if choose_lambda_by_cv:
        design = build_design(lineups, min_possessions=min_poss)
        cv = cv_lambda(
            design,
            registry.get("mbb.impact.lambda_grid").numbers(),
            n_folds=int(registry.get("mbb.impact.cv_folds").scalar()),
            rng=np.random.default_rng(seed),
        )
        penalty = cv.best
    rapm, design, fit = fit_rapm(
        lineups, season=season, lam=penalty, min_possessions=min_poss, cv=cv
    )

    ratings = team_ratings(lineups, design, fit).with_columns(
        (pl.col("lineup_off_poss") + pl.col("lineup_def_poss")).alias("lineup_poss")
    )
    totals = team_possession_totals(frames.possessions)
    members = frames.team_ids.group_by("conference").agg(pl.len().alias("n_conference_members"))
    teams = (
        season_games.teams.join(
            totals.select("team", "ortg", "drtg", "pace"), on="team", how="inner"
        )
        .join(
            ratings.select("team", "adj_off", "adj_def", "adj_net", "lineup_poss"),
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
    margin_sd = _margin_sd(games, teams, rapm)
    return SeasonModel(
        season=season,
        rapm=rapm,
        teams=teams,
        games=games,
        exponent=exponent,
        margin_sd=margin_sd,
        lineup_counts=lineups.counts,
        data_date=_last_game_date(frames.schedule),
        sources=frames.sources,
    )


def _margin_sd(games: pl.DataFrame, teams: pl.DataFrame, rapm: RapmResult) -> float:
    nets = teams.select("team", "adj_net")
    frame = (
        games.filter(pl.col("possessions").is_not_null())
        .join(nets.rename({"team": "home", "adj_net": "home_net"}), on="home", how="left")
        .join(nets.rename({"team": "away", "adj_net": "away_net"}), on="away", how="left")
        .with_columns(
            pl.col("home_net").fill_null(rapm.non_d1_net),
            pl.col("away_net").fill_null(rapm.non_d1_net),
            pl.when(pl.col("neutral")).then(0.0).otherwise(1.0).alias("venue"),
        )
        .with_columns(
            (
                (pl.col("home_net") - pl.col("away_net") + 2 * rapm.home_court * pl.col("venue"))
                * pl.col("possessions")
                / PER_100
            ).alias("expected"),
            (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).alias("actual"),
        )
    )
    return fit_margin_sd(frame["actual"].to_numpy(), frame["expected"].to_numpy())


def _last_game_date(schedule: pl.DataFrame) -> date:
    parsed = schedule.select(
        pl.col("game_date").str.strptime(pl.Date, "%m/%d/%Y", strict=False).max()
    ).item()
    if isinstance(parsed, date):
        return parsed
    return datetime.now().date()
