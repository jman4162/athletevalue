"""Splitting a roster budget among players.

No public data record individual college basketball pay, so this is an
allocation, not a fitted prediction. The split has two parts:

1. Role. Players are ranked by possessions played. The top five are starters,
   the next few are rotation players and the rest are bench. Reported pay ratios
   set each role's average pay relative to starters.
2. Impact. Within a role, pay is proportional to impact above replacement raised
   to an exponent. Impact is normalized within the role, so the role averages in
   step 1 still hold.

Players who are on the floor for very few of the team's possessions receive
nothing. Every step draws from the uncertainty in ratings, budgets and ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from athletevalue.constants import PLAYERS_ON_COURT
from athletevalue.uncertainty.draws import FloatArray

ROLES = ("starter", "rotation", "bench", "unpaid")


@dataclass(frozen=True)
class AllocationRules:
    replacement: float
    rotation_size: int
    min_possession_share: float
    performance_exponent: float
    performance_floor: float
    rotation_ratio: tuple[float, float]
    bench_ratio: tuple[float, float]


@dataclass(frozen=True)
class Allocation:
    athlete_ids: tuple[str, ...]
    roles: tuple[str, ...]
    pay: FloatArray
    """Draws by player: shape (n_draws, n_players), USD."""

    def player(self, athlete_id: str) -> FloatArray:
        column: FloatArray = self.pay[:, self.athlete_ids.index(athlete_id)]
        return column

    def role(self, athlete_id: str) -> str:
        return self.roles[self.athlete_ids.index(athlete_id)]


def assign_roles(possession_share: NDArray[np.float64], rules: AllocationRules) -> list[str]:
    order = np.argsort(-possession_share, kind="stable")
    roles = ["bench"] * possession_share.size
    for rank, index in enumerate(order):
        if possession_share[index] < rules.min_possession_share:
            roles[index] = "unpaid"
        elif rank < PLAYERS_ON_COURT:
            roles[index] = "starter"
        elif rank < PLAYERS_ON_COURT + rules.rotation_size:
            roles[index] = "rotation"
    return roles


def allocate(
    roster: pl.DataFrame,
    budget: FloatArray,
    rules: AllocationRules,
    *,
    rng: np.random.Generator,
) -> Allocation:
    """Split each budget draw across *roster*.

    ``roster`` needs athlete_id, net, se_net and possession_share columns.
    """
    ids = tuple(roster["athlete_id"].to_list())
    share = roster["possession_share"].cast(pl.Float64).to_numpy()
    roles = assign_roles(share, rules)
    n_draws, n_players = budget.size, len(ids)

    net = rng.normal(
        roster["net"].to_numpy()[None, :],
        roster["se_net"].to_numpy()[None, :],
        (n_draws, n_players),
    )
    impact = (
        np.maximum(net - rules.replacement, rules.performance_floor) ** rules.performance_exponent
    )

    role_ratio = {
        "starter": np.ones(n_draws),
        "rotation": rng.uniform(*rules.rotation_ratio, n_draws),
        "bench": rng.uniform(*rules.bench_ratio, n_draws),
        "unpaid": np.zeros(n_draws),
    }
    weight = np.zeros((n_draws, n_players))
    role_array = np.array(roles)
    for role, ratio in role_ratio.items():
        members = role_array == role
        if not members.any():
            continue
        within = impact[:, members] / impact[:, members].mean(axis=1, keepdims=True)
        weight[:, members] = ratio[:, None] * within
    totals = weight.sum(axis=1, keepdims=True)
    pay = np.divide(budget[:, None] * weight, totals, out=np.zeros_like(weight), where=totals > 0)
    return Allocation(athlete_ids=ids, roles=tuple(roles), pay=pay)
