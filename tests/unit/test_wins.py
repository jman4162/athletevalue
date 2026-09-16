from __future__ import annotations

import numpy as np
import pytest

from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.wins.pythag import fit_exponent, margin_slope, win_pct
from athletevalue.wins.war import (
    GameContext,
    PlayerImpact,
    TeamContext,
    war_analytic,
    war_simulated,
    wins_per_point_pythag,
)


def test_slope_at_parity_is_x_over_4o():
    assert margin_slope(104, 104, 11.5) == pytest.approx(11.5 / (4 * 104))


def test_slope_matches_numeric_derivative():
    o, d, x, h = 112.0, 98.0, 9.0, 1e-4
    # Offense up h and defense down h moves the margin by 2h.
    numeric = (win_pct(o + h / 2, d - h / 2, x) - win_pct(o - h / 2, d + h / 2, x)) / (2 * h)
    assert margin_slope(o, d, x) == pytest.approx(numeric, rel=1e-5)


def test_wins_per_point_pythag_scales_with_games():
    team = TeamContext("T", games=32, ortg=104, drtg=104, pace=68, adj_net=0, lineup_poss=4000)
    assert wins_per_point_pythag(team, 11.5) == pytest.approx(11.5 / 416 * 32)


def test_fit_exponent_recovers_truth():
    rng = np.random.default_rng(0)
    ortg = rng.uniform(95, 120, 300)
    drtg = rng.uniform(95, 120, 300)
    wins = 1 / (1 + (drtg / ortg) ** 9.0)
    assert fit_exponent(ortg, drtg, wins, np.full(300, 30.0), bounds=(2, 30)) == pytest.approx(
        9.0, abs=1e-2
    )


def _schedule(n=30):
    rng = np.random.default_rng(1)
    return [
        GameContext(
            opponent_net=float(rng.normal(0, 8)),
            venue=int(rng.choice([-1, 0, 1])),
            possessions=70.0,
        )
        for _ in range(n)
    ]


def test_simulated_and_analytic_agree_for_small_impacts():
    team = TeamContext("T", games=30, ortg=105, drtg=103, pace=70, adj_net=2.0, lineup_poss=4200)
    player = PlayerImpact(net=-1.0, se_net=0.5, player_poss=2000)
    games = _schedule()
    kwargs = {
        "replacement": -2.0,
        "home_court": 3.0,
        "margin_sd": 11.5,
        "level": 0.8,
        "status": EvidenceStatus.ESTIMATED,
    }
    analytic = war_analytic(player, team, games, **kwargs)
    simulated, draws = war_simulated(
        player, team, games, rng=np.random.default_rng(2), n_draws=4000, **kwargs
    )
    assert simulated.value == pytest.approx(analytic.value, rel=0.15)
    assert draws.shape == (4000,)
    assert analytic.status is EvidenceStatus.ESTIMATED


def test_replacement_player_has_zero_war():
    team = TeamContext("T", games=30, ortg=105, drtg=103, pace=70, adj_net=2.0, lineup_poss=4200)
    player = PlayerImpact(net=-2.0, se_net=0.0, player_poss=2000)
    est = war_analytic(
        player,
        team,
        _schedule(),
        replacement=-2.0,
        home_court=3.0,
        margin_sd=11.5,
        level=0.8,
        status=EvidenceStatus.ESTIMATED,
    )
    assert est.value == 0.0
