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
    FEATURE_ASSUMPTIONS,
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
from athletevalue.wins.war import war_analytic, war_draws

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
    market_economics: EconomicsModel | None = None,
    seed: int = 0,
    as_of: date | None = None,
) -> PlayerValuation:
    """*market_economics* is the economics model the market fit's features used; it
    defaults to *economics* and must be given when that is ``None`` but the fit had one."""
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
    impact_row = season.player_impact(athlete)
    context = season.team_context(team_name)
    games = season.schedule(
        team_name, floor_non_d1=registry.get("mbb.wins.non_d1_opponent_floor").flag()
    )
    war_linear = war_analytic(
        impact_row,
        context,
        games,
        replacement=season.replacement,
        home_court=season.rapm.home_court,
        margin_sd=season.margin_sd,
        level=level,
        status=war_status,
    )
    war_sensitivity = {
        name: float(
            np.median(
                war_draws(
                    impact_row,
                    context,
                    games,
                    replacement=level_value,
                    home_court=season.rapm.home_court,
                    margin_sd=season.margin_sd,
                    rng=np.random.default_rng(seed),
                    n_draws=len(draws.net_draws[athlete]),
                    net_draws=draws.net_draws[athlete],
                )
            )
        )
        for name, level_value in season.replacement_levels.items()
    }

    assumptions = [*impact_ids, *WAR_ASSUMPTIONS]
    program_value: Estimate | None = None
    program_two_season: Estimate | None = None
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
            "win revenue, this season": summarize(
                pv.win_revenue_current,
                unit=USD,
                status=estimated,
                method="win_revenue",
                level=level,
            ),
            "bid revenue, this season": summarize(
                pv.bid_revenue_current,
                unit=USD,
                status=estimated,
                method="bid_revenue",
                level=level,
            ),
            "tournament units": summarize(
                pv.tournament_units,
                unit=USD,
                status=with_units,
                method="tournament_units",
                level=level,
            ),
        }
        program_draws = pv.annual
        program_value = summarize(
            program_draws, unit=USD, status=with_units, method="program_value_annual", level=level
        )
        program_two_season = summarize(
            pv.two_season,
            unit=USD,
            status=with_units,
            method="program_value_two_season",
            level=level,
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
            features = player_features(
                season,
                economics if market_economics is None else market_economics,
                registry,
                teams=(team_name,),
            )
            feature_rows = features.filter(pl.col("athlete_id") == athlete)
            if feature_rows.is_empty():
                raise LookupError(
                    f"{row['name']} has no market features: his team has no games with "
                    "possession data"
                )
            x = feature_rows.select(FEATURES).to_numpy()[0]
            model = market_fit.model
            # Point, interval and the draws surplus uses all come from the same CV+ set.
            price_draws = np.exp(
                model.draws_log(x, len(draws.war[athlete]), np.random.default_rng(seed + 1))
            )
            model_ids = (*FEATURE_ASSUMPTIONS, *MODEL_ASSUMPTIONS)
            model_status = EvidenceStatus.weakest(
                EvidenceStatus.ESTIMATED,
                war_status,
                *(registry.get(i).status for i in model_ids),
            )
            market = summarize(
                price_draws,
                unit=USD,
                status=model_status,
                method=f"fitted_market_model:{model.n_labels} labels",
                level=level,
            )
            price_basis = "fitted_model"
            assumptions += list(model_ids)

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
        replacement_definition=season.replacement_definition,
        replacement_level=season.replacement,
        war_sensitivity=war_sensitivity,
        program_value=program_value,
        program_value_two_season=program_two_season,
        program_value_components=components,
        roster_market_value=market,
        allocated_market_value=allocated if market is not allocated else None,
        observed_price=observed,
        price_basis=price_basis,
        surplus=surplus,
        quadrant=quadrant,
        drivers=_drivers(season, draws, athlete, row, economics, registry),
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
        value = float(np.median(draws.program[athlete].annual)) if draws.program else float("nan")
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
    registry: AssumptionRegistry,
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
        cap_id = f"economics.house_revenue_share_cap_{season.season}"
        context = ""
        if cap_id in registry:
            cap = money(registry.get(cap_id).scalar())
            context = f"; the House cap for direct revenue sharing across all sports is {cap}"
        drivers.append(
            Driver(
                sign="+" if draws.budget.tier.value == "power" else "-",
                text=(
                    f"{draws.conference} roster budget tier: {draws.budget.tier.value}"
                    f" (published average, collectives included){context}"
                ),
            )
        )
    return drivers
