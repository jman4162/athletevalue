"""Composing a player's valuation from the fitted season, economics and market layers."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.identity.resolve import resolve_player
from athletevalue.market.registry import observed_annual_pay
from athletevalue.market.surplus import assign_quadrants, surplus_draws
from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.schemas.identity import PlayerRef
from athletevalue.schemas.registry import DealRecord
from athletevalue.uncertainty.draws import summarize
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.market_fit import (
    FEATURES,
    MODEL_ASSUMPTIONS,
    MarketFit,
    player_features,
)
from athletevalue.valuation.report import money
from athletevalue.valuation.result import Driver, PlayerValuation
from athletevalue.valuation.season import SeasonModel
from athletevalue.valuation.team import (
    ALLOCATION_ASSUMPTIONS,
    PROGRAM_ASSUMPTIONS,
    UNIT_ASSUMPTIONS,
    WAR_ASSUMPTIONS,
    TeamDraws,
    impact_assumptions,
    team_draws,
)
from athletevalue.versions import MODEL_VERSION
from athletevalue.wins.war import war_analytic

USD = "USD"
RATING = "points per 100 possessions"


def value_player(
    season: SeasonModel,
    economics: EconomicsModel | None,
    registry: AssumptionRegistry,
    name: str,
    *,
    team: str | None = None,
    deals: list[DealRecord] | None = None,
    market_fit: MarketFit | None = None,
    market_note: str | None = None,
    seed: int = 0,
    as_of: date | None = None,
) -> PlayerValuation:
    row = resolve_player(season.rapm.table, name, team=team)
    athlete, team_name = str(row["athlete_id"]), str(row["team"])
    draws = team_draws(season, economics, registry, team_name, seed=seed)
    level = registry.get("mbb.wins.interval_level").scalar()
    impact_ids = impact_assumptions(season)
    impact_status = EvidenceStatus.weakest(*(registry.get(i).status for i in impact_ids))
    suffix = "_box_prior" if season.rapm.prior == "box" else ""
    war_status = EvidenceStatus.weakest(
        impact_status, *(registry.get(i).status for i in WAR_ASSUMPTIONS)
    )
    warnings = list(draws.notes)
    if row["pooled"]:
        warnings.append(
            "fewer possessions than the modelling threshold; rating is the pooled "
            "low-minute coefficient, not this player's own"
        )

    impact = Estimate.normal(
        float(row["net"]),
        float(row["se_net"]),
        unit=RATING,
        status=impact_status,
        method=f"rapm_net{suffix}",
        level=level,
    )
    offense = Estimate.normal(
        float(row["orapm"]),
        float(row["se_off"]),
        unit=RATING,
        status=impact_status,
        method=f"rapm_offense{suffix}",
        level=level,
    )
    defense = Estimate.normal(
        float(row["drapm"]),
        float(row["se_def"]),
        unit=RATING,
        status=impact_status,
        method=f"rapm_defense{suffix}",
        level=level,
    )
    war = summarize(
        draws.war[athlete], unit="wins", status=war_status, method="war_simulated", level=level
    )
    war_linear = war_analytic(
        season.player_impact(athlete),
        season.team_context(team_name),
        season.schedule(team_name),
        replacement=registry.get("mbb.wins.replacement_level").scalar(),
        home_court=season.rapm.home_court,
        margin_sd=season.margin_sd,
        level=level,
        status=war_status,
    )

    assumptions = [*impact_ids, *WAR_ASSUMPTIONS]
    program_value: Estimate | None = None
    components: dict[str, Estimate] = {}
    program_draws = None
    if draws.program is not None:
        pv = draws.program[athlete]
        estimated = EvidenceStatus.weakest(
            war_status, *(registry.get(i).status for i in PROGRAM_ASSUMPTIONS)
        )
        with_units = EvidenceStatus.weakest(
            estimated, *(registry.get(i).status for i in UNIT_ASSUMPTIONS)
        )
        components = {
            "win_revenue": summarize(
                pv.win_revenue, unit=USD, status=estimated, method="win_revenue", level=level
            ),
            "bid_revenue": summarize(
                pv.bid_revenue, unit=USD, status=estimated, method="bid_revenue", level=level
            ),
            "tournament_units": summarize(
                pv.tournament_units,
                unit=USD,
                status=with_units,
                method="tournament_units",
                level=level,
            ),
        }
        program_draws = pv.total
        program_value = summarize(
            program_draws, unit=USD, status=with_units, method="program_value", level=level
        )
        assumptions += [*PROGRAM_ASSUMPTIONS, *UNIT_ASSUMPTIONS]

    market: Estimate | None = None
    price_draws = None
    role = None
    if draws.allocation is not None and draws.budget is not None:
        role = draws.allocation.role(athlete)
        market_status = EvidenceStatus.weakest(
            draws.budget.status, *(registry.get(i).status for i in ALLOCATION_ASSUMPTIONS)
        )
        price_draws = draws.allocation.player(athlete)
        market = summarize(
            price_draws,
            unit=USD,
            status=market_status,
            method="roster_share_allocation",
            level=level,
        )
        assumptions += [*draws.budget.assumption_ids, *ALLOCATION_ASSUMPTIONS]

    allocated = market
    price_basis = "allocation" if market is not None else None
    if market_note:
        warnings.append(f"fitted market model not used: {market_note}")
    if market_fit is not None:
        usable, note = market_fit.usable(registry)
        if not usable:
            warnings.append(f"fitted market model not used: {note}")
        else:
            features = player_features(season, economics, registry, teams=(team_name,))
            x = features.filter(pl.col("athlete_id") == athlete).select(FEATURES).to_numpy()[0]
            model = market_fit.model
            lower, upper = model.interval_log(x[None, :], level)
            median = float(np.clip(model.predict_log(x[None, :])[0], lower[0], upper[0]))
            price_draws = np.exp(
                model.draws_log(x, len(draws.war[athlete]), np.random.default_rng(seed + 1))
            )
            model_status = EvidenceStatus.weakest(
                EvidenceStatus.ESTIMATED, *(registry.get(i).status for i in MODEL_ASSUMPTIONS)
            )
            market = Estimate(
                value=float(np.exp(median)),
                lower=float(np.exp(lower[0])),
                upper=float(np.exp(upper[0])),
                level=level,
                unit=USD,
                status=model_status,
                method=f"fitted_market_model:{model.n_labels} labels",
            )
            price_basis = "fitted_model"
            assumptions += list(MODEL_ASSUMPTIONS)

    observed: Estimate | None = None
    if deals:
        found = observed_annual_pay(
            deals, name=str(row["name"]), school=team_name, season=season.season
        )
        if found is not None:
            total, used = found
            weakest_source = EvidenceStatus.REPORTED
            observed = Estimate.exact(
                total, unit=USD, status=weakest_source, method=f"deal_registry:{len(used)} deals"
            )
            price_basis = "registry"

    surplus: Estimate | None = None
    quadrant: str | None = None
    if program_draws is not None and (price_draws is not None or observed is not None):
        price_for_surplus = (
            np.full(program_draws.size, observed.value) if observed is not None else price_draws
        )
        assert price_for_surplus is not None
        status = EvidenceStatus.weakest(
            program_value.status if program_value else EvidenceStatus.UNRESOLVED,
            observed.status
            if observed is not None
            else (market.status if market else EvidenceStatus.UNRESOLVED),
        )
        surplus = summarize(
            surplus_draws(program_draws, price_for_surplus),
            unit=USD,
            status=status,
            method="program_value_minus_price",
            level=level,
        )
        quadrant = _quadrant_for(draws, athlete)

    return PlayerValuation(
        player=PlayerRef(
            athlete_id=athlete, name=str(row["name"]), team=team_name, season=season.season
        ),
        conference=draws.conference,
        role=role,
        athletic_impact=impact,
        offense=offense,
        defense=defense,
        war=war,
        war_linear=war_linear,
        program_value=program_value,
        program_value_components=components,
        roster_market_value=market,
        allocated_market_value=allocated if price_basis == "fitted_model" else None,
        observed_price=observed,
        price_basis=price_basis,
        surplus=surplus,
        quadrant=quadrant,
        drivers=_drivers(season, draws, athlete, row, economics),
        model_version=MODEL_VERSION,
        as_of=as_of or date.today(),
        data_through=season.data_date,
        sources=[*season.sources, *(economics.sources if economics else ())],
        assumptions_used=sorted(set(assumptions)),
        warnings=warnings,
    )


def team_table(draws: TeamDraws) -> pl.DataFrame:
    """One row per player: role, rating, WAR, program value, price, surplus, quadrant (medians)."""
    rows = []
    for record in draws.roster.iter_rows(named=True):
        athlete = record["athlete_id"]
        value = float(np.median(draws.program[athlete].total)) if draws.program else float("nan")
        price = (
            float(np.median(draws.allocation.player(athlete))) if draws.allocation else float("nan")
        )
        rows.append(
            {
                "athlete_id": athlete,
                "name": record["name"],
                "role": draws.allocation.role(athlete) if draws.allocation else None,
                "possession_share": record["possession_share"],
                "net": record["net"],
                "se_net": record["se_net"],
                "war": float(np.median(draws.war[athlete])),
                "program_value": value,
                "price": price,
                "surplus": value - price,
            }
        )
    table = pl.DataFrame(rows).sort("possession_share", descending=True)
    if draws.program is None or draws.allocation is None:
        return table.with_columns(pl.lit(None, dtype=pl.Utf8).alias("quadrant"))
    return assign_quadrants(table)


def _quadrant_for(draws: TeamDraws, athlete: str) -> str | None:
    table = team_table(draws)
    value = table.filter(pl.col("athlete_id") == athlete)["quadrant"]
    return None if value.is_empty() else value[0]


def _drivers(
    season: SeasonModel,
    draws: TeamDraws,
    athlete: str,
    row: dict[str, object],
    economics: EconomicsModel | None,
) -> list[Driver]:
    drivers: list[Driver] = []
    table = season.rapm.table.filter(~pl.col("pooled"))
    net = float(row["net"])  # type: ignore[arg-type]
    rank = int((table["net"] > net).sum()) + 1
    sign = "+" if net > float(table["net"].median()) else "-"  # type: ignore[arg-type]
    drivers.append(
        Driver(sign=sign, text=f"on-court impact ranks {rank} of {table.height} rated players")
    )
    if season.rapm.prior == "box":
        prior_net = float(row["prior_net"])  # type: ignore[arg-type]
        drivers.append(
            Driver(
                sign="~",
                text=(
                    f"box-score prior {prior_net:+.1f} per 100; his possessions moved the "
                    f"rating {net - prior_net:+.1f}"
                ),
            )
        )

    share = float(draws.roster.filter(pl.col("athlete_id") == athlete)["possession_share"][0])
    team_median = float(draws.roster["possession_share"].median())  # type: ignore[arg-type]
    drivers.append(
        Driver(
            sign="+" if share > team_median else "-",
            text=f"on the floor for {share:.0%} of team possessions",
        )
    )
    team_row = season.teams.filter(pl.col("team") == draws.team).row(0, named=True)
    drivers.append(
        Driver(
            sign="~",
            text=(
                f"team went {team_row['wins']}-{team_row['losses']}, "
                f"net rating {team_row['adj_net']:+.1f}"
            ),
        )
    )
    if economics is not None:
        base = economics.context(
            draws.team,
            season=season.season,
            games=int(team_row["games"]),
            wins=int(team_row["wins"]),
            conference=draws.conference,
            members=int(team_row["n_conference_members"]),
        )
        if base is not None:
            latest = economics.panel.filter(pl.col("season") == economics.panel["season"].max())
            median = float(latest["rev_men"].median())  # type: ignore[arg-type]
            drivers.append(
                Driver(
                    sign="+" if base.revenue_base > median else "-",
                    text=(
                        f"program reports {money(base.revenue_base)} basketball revenue "
                        f"(D1 median {money(median)})"
                    ),
                )
            )
    if draws.budget is not None:
        drivers.append(
            Driver(
                sign="+" if draws.budget.tier.value == "power" else "-",
                text=f"{draws.conference} roster budget tier: {draws.budget.tier.value}",
            )
        )
    return drivers
