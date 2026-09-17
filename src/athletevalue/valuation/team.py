"""Draws for every player on one team: wins, program value and allocated pay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.economics.program_value import (
    ProgramValueDraws,
    UnitOverlapPolicy,
    UnitTerms,
    program_value_draws,
)
from athletevalue.market.allocation import Allocation, AllocationRules, allocate
from athletevalue.market.budget import MarketUnavailableError, RosterBudget, roster_budget
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.uncertainty.draws import FloatArray
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.season import SeasonModel
from athletevalue.wins.war import war_draws

IMPACT_ASSUMPTIONS = (
    "mbb.impact.ridge_lambda",
    "mbb.impact.min_possessions",
    "mbb.impact.drop_garbage_time",
)
PRIOR_ASSUMPTIONS = (
    "mbb.prior.use_box_prior",
    "mbb.prior.training_seasons",
    "mbb.prior.rate_pseudo_possessions",
    "mbb.prior.box_ridge",
    "mbb.prior.team_adjustment",
    "mbb.impact.ridge_lambda_with_prior",
)


def impact_assumptions(season: SeasonModel) -> tuple[str, ...]:
    """Registry entries behind the headline ratings of *season*."""
    if season.rapm.prior == "box":
        return (*IMPACT_ASSUMPTIONS, *PRIOR_ASSUMPTIONS)
    return IMPACT_ASSUMPTIONS


WAR_ASSUMPTIONS = (
    "mbb.wins.replacement_definition",
    "mbb.wins.replacement_level",
    "mbb.wins.bench_rank_range",
    "mbb.wins.game_margin_sd",
    "mbb.wins.margin_sd_d1_only",
    "mbb.wins.non_d1_opponent_floor",
    "mbb.wins.simulation_draws",
)
PROGRAM_ASSUMPTIONS = (
    "economics.bid_ridge_lambda",
    "economics.excluded_seasons",
    "economics.panel_first_season",
    "economics.panel_min_seasons",
    "economics.bootstrap_draws",
    "economics.revenue_reference_seasons",
    "market.power_conferences",
)
UNIT_ASSUMPTIONS = (
    "economics.tournament_unit_value",
    "economics.conference_unit_share_multiplier",
    "economics.unit_payout_installments",
    "economics.unit_payout_first_year",
    "economics.unit_discount_rate",
    "economics.unit_revenue_overlap",
)


def unit_terms(registry: AssumptionRegistry, units_per_bid: float) -> UnitTerms:
    """The tournament-unit payout, its schedule and how it is kept off the bid effect."""
    return UnitTerms(
        unit_value=registry.get("economics.tournament_unit_value").scalar(),
        units_per_bid=units_per_bid,
        share_multiplier=registry.get("economics.conference_unit_share_multiplier").scalar(),
        discount_rate=registry.get("economics.unit_discount_rate").scalar(),
        installments=int(registry.get("economics.unit_payout_installments").scalar()),
        first_year=int(registry.get("economics.unit_payout_first_year").scalar()),
        overlap=UnitOverlapPolicy(registry.get("economics.unit_revenue_overlap").text()),
    )


ALLOCATION_ASSUMPTIONS = (
    "market.role_weight_rotation",
    "market.role_weight_bench",
    "market.rotation_size",
    "market.min_possession_share",
    "market.performance_exponent",
    "market.performance_floor",
)


@dataclass(frozen=True)
class TeamDraws:
    team: str
    conference: str
    roster: pl.DataFrame
    """athlete_id, name, net, se_net, orapm, drapm, se_off, se_def, off_poss, def_poss,
    pooled, possession_share."""
    net_draws: dict[str, FloatArray]
    """Posterior draws of each player's net rating, shared by every layer below."""
    war: dict[str, FloatArray]
    program: dict[str, ProgramValueDraws] | None
    budget: RosterBudget | None
    allocation: Allocation | None
    notes: tuple[str, ...]

    def status(self, registry: AssumptionRegistry, ids: tuple[str, ...]) -> EvidenceStatus:
        return EvidenceStatus.weakest(*(registry.get(i).status for i in ids))


def team_draws(
    season: SeasonModel,
    economics: EconomicsModel | None,
    registry: AssumptionRegistry,
    team: str,
    *,
    seed: int,
) -> TeamDraws:
    rng = np.random.default_rng(seed)
    n_draws = int(registry.get("mbb.wins.simulation_draws").scalar())
    replacement = season.replacement
    context = season.team_context(team)
    games = season.schedule(
        team, floor_non_d1=registry.get("mbb.wins.non_d1_opponent_floor").flag()
    )
    team_row = season.teams.filter(pl.col("team") == team).row(0, named=True)
    conference = str(team_row["conference"])

    roster = season.roster(team).with_columns(
        ((pl.col("off_poss") + pl.col("def_poss")) / context.lineup_poss).alias("possession_share")
    )
    # One posterior draw per player per index. WAR, program value and the allocation
    # all read the same draw, so a player who is better than estimated in a draw is
    # better in every layer of that draw.
    net_matrix = rng.normal(
        roster["net"].to_numpy()[None, :],
        roster["se_net"].to_numpy()[None, :],
        (n_draws, roster.height),
    )
    net_draws = {athlete: net_matrix[:, k] for k, athlete in enumerate(roster["athlete_id"])}
    war = {
        athlete: war_draws(
            season.player_impact(athlete),
            context,
            games,
            replacement=replacement,
            home_court=season.rapm.home_court,
            margin_sd=season.margin_sd,
            rng=rng,
            n_draws=n_draws,
            net_draws=net_draws[athlete],
        )
        for athlete in roster["athlete_id"]
    }

    notes: list[str] = []
    program: dict[str, ProgramValueDraws] | None = None
    if economics is None:
        notes.append("program economics were not loaded")
    else:
        program_context = economics.context(
            team,
            season=season.season,
            games=int(team_row["games"]),
            wins=int(team_row["wins"]),
            conference=conference,
            members=int(team_row["n_conference_members"]),
            sos=float(team_row["sos"]),
        )
        if program_context is None:
            notes.append(f"EADA reports no men's basketball revenue for {team}")
        else:
            units = unit_terms(registry, economics.units_per_bid)
            program = {
                athlete: program_value_draws(
                    draws,
                    program_context,
                    economics.revenue,
                    economics.bids,
                    units=units,
                    rng=rng,
                )
                for athlete, draws in war.items()
            }

    budget: RosterBudget | None = None
    allocation: Allocation | None = None
    try:
        budget = roster_budget(conference, season.season, registry, n_draws=n_draws, rng=rng)
    except MarketUnavailableError as error:
        notes.append(str(error))
    if budget is not None:
        rules = AllocationRules(
            replacement=replacement,
            rotation_size=int(registry.get("market.rotation_size").scalar()),
            min_possession_share=registry.get("market.min_possession_share").scalar(),
            performance_exponent=registry.get("market.performance_exponent").scalar(),
            performance_floor=registry.get("market.performance_floor").scalar(),
            rotation_ratio=registry.get("market.role_weight_rotation").interval(),
            bench_ratio=registry.get("market.role_weight_bench").interval(),
        )
        allocation = allocate(roster, budget.draws, rules, rng=rng, net_draws=net_matrix)
    return TeamDraws(
        team=team,
        conference=conference,
        roster=roster,
        net_draws=net_draws,
        war=war,
        program=program,
        budget=budget,
        allocation=allocation,
        notes=tuple(notes),
    )
