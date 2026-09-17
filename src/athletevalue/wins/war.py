"""Wins above replacement from a player's net rating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from athletevalue.constants import PER_100
from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.uncertainty.draws import FloatArray, summarize
from athletevalue.wins.pythag import margin_slope


@dataclass(frozen=True)
class TeamContext:
    team: str
    games: int
    ortg: float
    drtg: float
    pace: float
    """Possessions per team per game."""
    adj_net: float
    lineup_poss: float
    """Offensive plus defensive possessions in the lineup data (garbage time excluded)."""


@dataclass(frozen=True)
class GameContext:
    opponent_net: float
    venue: int
    """+1 home, -1 away, 0 neutral, from the player's team's perspective."""
    possessions: float


@dataclass(frozen=True)
class PlayerImpact:
    net: float
    se_net: float
    player_poss: float
    """Offensive plus defensive possessions played (garbage time excluded)."""

    def on_share(self, team: TeamContext) -> float:
        return self.player_poss / team.lineup_poss


def wins_per_point_pythag(team: TeamContext, exponent: float) -> float:
    """Season wins per point per 100 possessions of net rating, from Pythagorean expectation.

    Uses the team's raw season efficiencies, so it ignores who the team played. Kept
    as a league-level check on the schedule-based conversion.
    """
    return margin_slope(team.ortg, team.drtg, exponent) * team.games


def wins_per_point_schedule(
    team: TeamContext, games: list[GameContext], *, home_court: float, margin_sd: float
) -> float:
    """Season wins per point per 100 possessions, from the per-game win model.

    The derivative of ``sum_g Phi(margin_g / sd)`` with respect to the team's net
    rating, evaluated at the team's fitted rating over its actual schedule.
    """
    margin, poss = _game_margins(team, games, home_court)
    return float(np.sum(norm.pdf(margin / margin_sd) * poss / PER_100 / margin_sd))


def war_analytic(
    player: PlayerImpact,
    team: TeamContext,
    games: list[GameContext],
    *,
    replacement: float,
    home_court: float,
    margin_sd: float,
    level: float,
    status: EvidenceStatus,
) -> Estimate:
    """Linearized WAR: ``(net - replacement) * on_share * dWins/dNet``.

    Accurate for small impacts. For a high-usage player on a team far from .500 the
    per-game win curve bends, and ``war_simulated`` is the headline figure.
    """
    slope = wins_per_point_schedule(team, games, home_court=home_court, margin_sd=margin_sd)
    scale = player.on_share(team) * slope
    return Estimate.normal(
        (player.net - replacement) * scale,
        player.se_net * scale,
        unit="wins",
        status=status,
        method="war_analytic",
        level=level,
    )


def _game_margins(
    team: TeamContext, games: list[GameContext], home_court: float
) -> tuple[FloatArray, FloatArray]:
    if not games:
        raise ValueError(f"{team.team}: no games with possession data")
    opponent = np.array([g.opponent_net for g in games])
    venue = np.array([g.venue for g in games], dtype=np.float64)
    poss = np.array([g.possessions for g in games])
    margin = (team.adj_net - opponent + 2 * home_court * venue) * poss / PER_100
    return margin, poss


def war_draws(
    player: PlayerImpact,
    team: TeamContext,
    games: list[GameContext],
    *,
    replacement: float,
    home_court: float,
    margin_sd: float,
    rng: np.random.Generator,
    n_draws: int,
    net_draws: FloatArray | None = None,
) -> FloatArray:
    """Expected-win difference over the team's actual schedule, one value per draw.

    Each draw samples the player's net rating from its posterior (or takes it from
    *net_draws*, so other quantities can share the same draw), lowers the team's
    rating by the player's share of possessions times his margin over replacement,
    and sums the change in win probability across games.
    """
    with_player, poss = _game_margins(team, games, home_court)
    if net_draws is None:
        net_draws = rng.normal(player.net, player.se_net, n_draws)
    delta = (net_draws - replacement) * player.on_share(team)
    without = with_player[None, :] - delta[:, None] * poss[None, :] / PER_100
    p_with = norm.cdf(with_player / margin_sd)
    p_without = norm.cdf(without / margin_sd)
    wins: FloatArray = (p_with[None, :] - p_without).sum(axis=1)
    return wins


def war_simulated(
    player: PlayerImpact,
    team: TeamContext,
    games: list[GameContext],
    *,
    replacement: float,
    home_court: float,
    margin_sd: float,
    rng: np.random.Generator,
    n_draws: int,
    level: float,
    status: EvidenceStatus,
) -> tuple[Estimate, FloatArray]:
    draws = war_draws(
        player,
        team,
        games,
        replacement=replacement,
        home_court=home_court,
        margin_sd=margin_sd,
        rng=rng,
        n_draws=n_draws,
    )
    estimate = summarize(draws, unit="wins", status=status, method="war_simulated", level=level)
    return estimate, draws


def fit_margin_sd(actual_margin: FloatArray, expected_margin: FloatArray) -> float:
    """Standard deviation of game margins around their expectation."""
    residual = actual_margin - expected_margin
    return float(np.sqrt(np.mean(residual**2)))
