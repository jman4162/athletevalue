"""Features and labels for the fitted roster-market model, and its use gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.constants import PLAYERS_ON_COURT
from athletevalue.frames.codes import first_seen_codes
from athletevalue.identity.names import normalize_name
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
    replacement = season.replacement
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
                sos=float(team_row["sos"]),
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
    groups: dict[tuple[int, str, str], list[DealRecord]] = {}
    for deal in deals:
        if deal.sport != "mbb" or deal.deal_type.value not in types:
            continue
        key = (deal.season, normalize_name(deal.athlete_name), normalize_name(deal.school))
        groups.setdefault(key, []).append(deal)
    # Spellings can differ between deals for one player, so pay is summed per resolved
    # athlete, not per name as written.
    pay: dict[tuple[int, str], list[float]] = {}
    found: dict[tuple[int, str], tuple[str, str]] = {}
    unmatched: list[str] = []
    for (season, _, _), group in groups.items():
        first = group[0]
        label = f"{first.athlete_name} ({first.school}, {season})"
        table = features.filter(pl.col("season") == season)
        if table.is_empty():
            unmatched.append(f"{label}: season not fitted")
            continue
        athlete_id = next((d.athlete_id for d in reversed(group) if d.athlete_id), None)
        try:
            if athlete_id is None or table.filter(pl.col("athlete_id") == athlete_id).is_empty():
                team = resolve_team(table.select("team").unique(), first.school)
                athlete_id = str(resolve_player(table, first.athlete_name, team=team)["athlete_id"])
        except (PlayerNotFoundError, AmbiguousPlayerError, TeamNotFoundError) as error:
            unmatched.append(f"{label}: {error}")
            continue
        resolved = (season, athlete_id)
        pay.setdefault(resolved, []).extend(d.annualized_value for d in group)
        team_name = table.filter(pl.col("athlete_id") == athlete_id)["team"][0]
        found.setdefault(resolved, (label, team_name))
    rows = []
    for (season, athlete_id), values in pay.items():
        label, team_name = found[(season, athlete_id)]
        total = float(sum(values))
        if total <= 0:
            unmatched.append(f"{label}: non-positive pay")
            continue
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
    allocation_log_mae: float | None = None
    """Mean absolute log error of the allocation on the labels it prices above zero. The
    allocation never sees labels, so no hold-out is needed."""
    model_log_mae_allocated: float | None = None
    """The model's held-out log error on those same labels."""
    n_allocated: int = 0

    def usable(self, registry: AssumptionRegistry) -> tuple[bool, str]:
        model = self.model
        if model.nested_log_mae >= model.baseline_log_mae:
            return False, (
                f"nested held-out log error {model.nested_log_mae:.3f} does not beat the tier "
                f"median ({model.baseline_log_mae:.3f})"
            )
        note = (
            f"nested held-out log error {model.nested_log_mae:.3f} vs tier median "
            f"{model.baseline_log_mae:.3f}"
        )
        if self.allocation_log_mae is None or self.model_log_mae_allocated is None:
            return True, f"{note}; no label has an allocated price to compare"
        compared = (
            f"{self.model_log_mae_allocated:.3f} vs allocation {self.allocation_log_mae:.3f} "
            f"on the {self.n_allocated} labels it prices"
        )
        if (
            registry.get("market.model.require_beats_allocation").flag()
            and self.model_log_mae_allocated >= self.allocation_log_mae
        ):
            return False, f"held-out log error does not beat the allocation: {compared}"
        return True, f"{note}; {compared}"


def fit_market(
    features: pl.DataFrame,
    deals: list[DealRecord],
    registry: AssumptionRegistry,
    *,
    allocation_medians: pl.DataFrame | None = None,
    matched: LabelMatch | None = None,
) -> MarketFit:
    """Fit on every labeled player-season in *features*.

    *allocation_medians* (athlete_id, season, price) lets the fit report how the
    allocation does on the same players. *matched* reuses an earlier ``match_labels``
    result for these deals and features.
    """
    if matched is None:
        matched = match_labels(deals, features, registry)
    data = matched.labels.join(features, on=["athlete_id", "season"], how="inner", suffix="_f")
    model = fit_market_model(
        data.select(FEATURES).to_numpy().astype(np.float64),
        data["log_pay"].to_numpy(),
        first_seen_codes(data["team"]),
        data["tier"].to_numpy().astype(np.int64),
        features=FEATURES,
        ridge_grid=registry.get("market.model.ridge_grid").numbers(),
        min_labels=int(registry.get("market.model.min_labels").scalar()),
        min_schools=int(registry.get("market.model.min_schools").scalar()),
    )
    if allocation_medians is None:
        return MarketFit(model=model, labels=data, unmatched=matched.unmatched)
    # Labels from seasons without cited budgets, or players the allocation leaves unpaid,
    # have no allocated price; both predictors are scored on the rest.
    priced = (
        data.with_row_index("row")
        .join(allocation_medians, on=["athlete_id", "season"], how="inner")
        .filter(pl.col("price") > 0)
    )
    if priced.is_empty():
        return MarketFit(model=model, labels=data, unmatched=matched.unmatched)
    allocation_mae = float(
        np.mean(np.abs(priced["log_pay"].to_numpy() - np.log(priced["price"].to_numpy())))
    )
    return MarketFit(
        model=model,
        labels=data,
        unmatched=matched.unmatched,
        allocation_log_mae=allocation_mae,
        model_log_mae_allocated=float(model.nested_residuals[priced["row"].to_numpy()].mean()),
        n_allocated=priced.height,
    )


FEATURE_ASSUMPTIONS = ("market.power_conferences", "market.tier_b_conferences")
"""Registry entries ``player_features`` reads for the tier flags of every label."""

MODEL_ASSUMPTIONS = (
    "market.model.label_deal_types",
    "market.model.min_labels",
    "market.model.min_schools",
    "market.model.ridge_grid",
    "market.model.require_beats_allocation",
)
