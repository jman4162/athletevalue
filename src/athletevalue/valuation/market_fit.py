"""Features and labels for the fitted roster-market model, and its use gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.constants import PLAYERS_ON_COURT
from athletevalue.identity.resolve import (
    AmbiguousPlayerError,
    PlayerNotFoundError,
    TeamNotFoundError,
    resolve_player,
    resolve_team,
)
from athletevalue.market.budget import BudgetTier, budget_tier
from athletevalue.market.model import MarketModel, fit_market_model
from athletevalue.schemas.registry import DealRecord
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.season import SeasonModel
from athletevalue.wins.war import war_analytic

FEATURES: tuple[str, ...] = (
    "net",
    "orapm",
    "drapm",
    "prior_net",
    "possession_share",
    "starter",
    "war",
    "team_net",
    "team_win_pct",
    "ncaa_bid",
    "power",
    "tier_b",
    "log_revenue",
    "revenue_missing",
)
_TIER_CODE = {BudgetTier.POWER: 0, BudgetTier.TIER_B: 1, BudgetTier.TIER_C: 2}


def player_features(
    season: SeasonModel,
    economics: EconomicsModel | None,
    registry: AssumptionRegistry,
    *,
    teams: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """One row per rated player: identifiers, tier code and every column in ``FEATURES``."""
    replacement = registry.get("mbb.wins.replacement_level").scalar()
    level = registry.get("mbb.wins.interval_level").scalar()
    status = registry.get("mbb.impact.ridge_lambda").status
    rows = []
    frame = season.teams if teams is None else season.teams.filter(pl.col("team").is_in(teams))
    for team_row in frame.iter_rows(named=True):
        team = team_row["team"]
        context = season.team_context(team)
        games = season.schedule(team)
        if not games:
            continue
        tier, _ = budget_tier(team_row["conference"], registry)
        base = None
        if economics is not None:
            program = economics.context(
                team,
                season=season.season,
                games=int(team_row["games"]),
                wins=int(team_row["wins"]),
                conference=team_row["conference"],
                members=int(team_row["n_conference_members"]),
            )
            base = None if program is None else program.revenue_base
        roster = season.roster(team).with_columns(
            ((pl.col("off_poss") + pl.col("def_poss")) / context.lineup_poss).alias("share")
        )
        rank = roster["share"].rank(method="ordinal", descending=True).to_numpy()
        for position, player in enumerate(roster.iter_rows(named=True)):
            war = war_analytic(
                season.player_impact(player["athlete_id"]),
                context,
                games,
                replacement=replacement,
                home_court=season.rapm.home_court,
                margin_sd=season.margin_sd,
                level=level,
                status=status,
            )
            rows.append(
                {
                    "athlete_id": player["athlete_id"],
                    "name": player["name"],
                    "team": team,
                    "season": season.season,
                    "tier": _TIER_CODE[tier],
                    "net": player["net"],
                    "orapm": player["orapm"],
                    "drapm": player["drapm"],
                    "prior_net": player["prior_net"],
                    "possession_share": player["share"],
                    "starter": float(rank[position] <= PLAYERS_ON_COURT),
                    "war": war.value,
                    "team_net": context.adj_net,
                    "team_win_pct": team_row["wins"] / team_row["games"],
                    "ncaa_bid": float(team_row["ncaa_bid"]),
                    "power": float(tier is BudgetTier.POWER),
                    "tier_b": float(tier is BudgetTier.TIER_B),
                    "log_revenue": float(np.log(base)) if base else 0.0,
                    "revenue_missing": float(base is None),
                }
            )
    return pl.DataFrame(rows)


@dataclass(frozen=True)
class LabelMatch:
    labels: pl.DataFrame
    """athlete_id, season, team, log_pay, n_deals."""
    unmatched: tuple[str, ...]


def match_labels(
    deals: list[DealRecord], features: pl.DataFrame, registry: AssumptionRegistry
) -> LabelMatch:
    """Annual roster pay per player-season from deals, matched to rated players."""
    types = set(registry.get("market.model.label_deal_types").names())
    totals: dict[tuple[int, str, str], list[float]] = {}
    ids: dict[tuple[int, str, str], str | None] = {}
    for deal in deals:
        if deal.sport != "mbb" or deal.deal_type.value not in types:
            continue
        key = (deal.season, deal.athlete_name, deal.school)
        totals.setdefault(key, []).append(deal.annualized_value)
        ids[key] = deal.athlete_id or ids.get(key)
    rows, unmatched = [], []
    for (season, name, school), values in totals.items():
        table = features.filter(pl.col("season") == season)
        label = f"{name} ({school}, {season})"
        if table.is_empty():
            unmatched.append(f"{label}: season not fitted")
            continue
        athlete_id = ids[(season, name, school)]
        try:
            if athlete_id is None or table.filter(pl.col("athlete_id") == athlete_id).is_empty():
                team = resolve_team(table.select("team").unique(), school)
                athlete_id = str(resolve_player(table, name, team=team)["athlete_id"])
        except (PlayerNotFoundError, AmbiguousPlayerError, TeamNotFoundError) as error:
            unmatched.append(f"{label}: {error}")
            continue
        total = float(sum(values))
        if total <= 0:
            unmatched.append(f"{label}: non-positive pay")
            continue
        team_name = table.filter(pl.col("athlete_id") == athlete_id)["team"][0]
        rows.append(
            {
                "athlete_id": athlete_id,
                "season": season,
                "team": team_name,
                "log_pay": float(np.log(total)),
                "n_deals": len(values),
            }
        )
    schema = {
        "athlete_id": pl.Utf8,
        "season": pl.Int64,
        "team": pl.Utf8,
        "log_pay": pl.Float64,
        "n_deals": pl.Int64,
    }
    return LabelMatch(labels=pl.DataFrame(rows, schema=schema), unmatched=tuple(unmatched))


@dataclass(frozen=True)
class MarketFit:
    model: MarketModel
    labels: pl.DataFrame
    unmatched: tuple[str, ...]
    allocation_log_mae: float | None
    """Mean absolute log error of the allocation on the same labels, when every label has a
    budget. The allocation never sees labels, so no hold-out is needed."""

    def usable(self, registry: AssumptionRegistry) -> tuple[bool, str]:
        if self.model.cv_log_mae >= self.model.baseline_log_mae:
            return False, (
                f"held-out log error {self.model.cv_log_mae:.3f} does not beat the tier median "
                f"({self.model.baseline_log_mae:.3f})"
            )
        if (
            registry.get("market.model.require_beats_allocation").flag()
            and self.allocation_log_mae is not None
            and self.model.cv_log_mae >= self.allocation_log_mae
        ):
            return False, (
                f"held-out log error {self.model.cv_log_mae:.3f} does not beat the allocation "
                f"({self.allocation_log_mae:.3f})"
            )
        return True, (
            f"held-out log error {self.model.cv_log_mae:.3f} vs tier median "
            f"{self.model.baseline_log_mae:.3f}"
            + (
                ""
                if self.allocation_log_mae is None
                else f" and allocation {self.allocation_log_mae:.3f}"
            )
        )


def fit_market(
    features: pl.DataFrame,
    deals: list[DealRecord],
    registry: AssumptionRegistry,
    *,
    allocation_medians: pl.DataFrame | None = None,
) -> MarketFit:
    """Fit on every labeled player-season in *features*.

    *allocation_medians* (athlete_id, season, price) lets the fit report how the
    allocation does on the same players.
    """
    matched = match_labels(deals, features, registry)
    data = matched.labels.join(features, on=["athlete_id", "season"], how="inner", suffix="_f")
    model = fit_market_model(
        data.select(FEATURES).to_numpy().astype(np.float64),
        data["log_pay"].to_numpy(),
        data["team"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64),
        data["tier"].to_numpy().astype(np.int64),
        features=FEATURES,
        ridge_grid=registry.get("market.model.ridge_grid").numbers(),
        min_labels=int(registry.get("market.model.min_labels").scalar()),
        min_schools=int(registry.get("market.model.min_schools").scalar()),
    )
    allocation_mae = None
    if allocation_medians is not None:
        joined = data.join(allocation_medians, on=["athlete_id", "season"], how="inner")
        priced = joined.filter(pl.col("price") > 0)
        if priced.height == data.height:
            allocation_mae = float(
                np.mean(np.abs(priced["log_pay"].to_numpy() - np.log(priced["price"].to_numpy())))
            )
    return MarketFit(
        model=model, labels=data, unmatched=matched.unmatched, allocation_log_mae=allocation_mae
    )


MODEL_ASSUMPTIONS = (
    "market.model.label_deal_types",
    "market.model.min_labels",
    "market.model.min_schools",
    "market.model.ridge_grid",
    "market.model.require_beats_allocation",
)
