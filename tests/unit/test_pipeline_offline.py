"""The whole valuation pipeline on a synthetic season, with no downloads."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from athletevalue.economics.revenue import RevenueModel
from athletevalue.economics.tournament import BidModel
from athletevalue.frames.box import player_box_totals
from athletevalue.impact.prior import fit_box_prior
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.player import team_table, value_player
from athletevalue.valuation.season import assemble_season, baseline_rapm
from athletevalue.valuation.team import team_draws
from athletevalue.valuation.validate import validate_season
from tests.fixtures.raw_season import make_raw_season


@pytest.fixture(scope="module")
def season(registry):
    training_frames, _ = make_raw_season(seed=1, season=2025)
    prior = fit_box_prior(
        [
            (
                2025,
                baseline_rapm(training_frames, registry).table,
                player_box_totals(training_frames.player_box),
            )
        ],
        pseudo_possessions=registry.get("mbb.prior.rate_pseudo_possessions").scalar(),
        ridge=registry.get("mbb.prior.box_ridge").scalar(),
    )
    frames, truth = make_raw_season()
    return assemble_season(frames, registry, prior=prior), truth


@pytest.fixture(scope="module")
def economics(season):
    model, _ = season
    teams = model.teams["team"].to_list()
    panel = pl.DataFrame(
        {"team": teams, "season": 2025, "rev_men": [12e6 if "Big" in t else 2e6 for t in teams]}
    )
    return EconomicsModel(
        panel=panel,
        outcomes=pl.DataFrame(),
        revenue=RevenueModel(
            point=np.zeros(4),
            draws=np.tile([0.004, 0.002, 0.03, 0.02], (20, 1)),
            n_obs=1,
            n_schools=1,
            first_season=2012,
            last_season=2025,
        ),
        bids=BidModel(coef=np.array([-10.0, 15.0, -6.0, 14.0, 2.0]), n_obs=1),
        bid_holdout=pl.DataFrame(),
        units_per_bid=1.6,
        units_per_bid_field=1.9,
        power_conferences=frozenset({"Big Ten"}),
        reference_seasons=3,
        sources=(),
    )


def test_season_fit_recovers_player_order(season):
    model, truth = season
    table = model.rapm.table.filter(~pl.col("pooled"))
    # About 1,000 possessions per player leaves the posterior SD (~3.6) above the true
    # spread (3.0), so player order is recovered only loosely; team strength clearly.
    rho = spearmanr(table["net"], [truth[a] for a in table["athlete_id"]])[0]
    assert rho > 0.25
    by_conference = model.teams.group_by("conference").agg(pl.col("adj_net").mean())
    nets = dict(by_conference.iter_rows())
    assert nets["Big Ten"] - nets["WCC"] > 3
    assert 95 < model.rapm.intercept < 115
    assert model.teams.height == 8
    assert model.teams.filter(pl.col("ncaa_bid")).height >= 2
    assert model.margin_sd > 0


def test_validation_gates_run(season, registry):
    model, _ = season
    reference = model.rapm.table.select(
        pl.col("athlete_id").alias("player_id"), pl.col("net").alias("rapm_net"), "off_poss"
    )
    torvik = model.teams.select("team", pl.col("adj_net").alias("adj_em"))
    report = validate_season(model, registry, reference_rapm=reference, torvik=torvik)
    names = {g.name: g for g in report.gates}
    assert names["spearman_team_net_vs_torvik"].value == pytest.approx(1.0)
    assert names["torvik_teams_matched"].passed is False  # 8 teams is below the 250 minimum
    assert "prior_cv_error_ratio" in names
    assert model.rapm.prior == "box" and model.baseline.prior == "none"
    assert any("Pythagorean" in n for n in report.notes)


def test_team_draws_and_table(season, economics, registry):
    model, _ = season
    team = model.teams.filter(pl.col("conference") == "Big Ten")["team"][0]
    draws = team_draws(model, economics, registry, team, seed=1)
    assert draws.budget is not None and draws.allocation is not None and draws.program is not None
    np.testing.assert_allclose(draws.allocation.pay.sum(axis=1), draws.budget.draws)
    table = team_table(draws)
    assert set(table["quadrant"]) <= {
        "value_above_price",
        "both_above_median",
        "value_below_price",
        "both_below_median",
        None,
    }
    unpaid = table.filter(pl.col("role") == "unpaid")
    assert unpaid["quadrant"].null_count() == unpaid.height
    assert set(model.replacement_levels) == {"nba_convention", "pooled", "bench_median"}
    assert model.replacement == model.replacement_levels[model.replacement_definition]
    # Every layer reads the same rating draw, so price and value move together.
    athlete = table["athlete_id"][0]
    price = draws.allocation.player(athlete)
    value = draws.program[athlete].annual
    assert np.corrcoef(draws.net_draws[athlete], price)[0, 1] > 0.5
    assert np.corrcoef(draws.net_draws[athlete], value)[0, 1] > 0.5
    assert table["price"].sum() == pytest.approx(float(np.median(draws.budget.draws)), rel=0.25)


def test_value_player_end_to_end(season, economics, registry):
    model, _ = season
    name = model.rapm.table.filter(~pl.col("pooled"))["name"][0]
    valuation = value_player(model, economics, registry, name, seed=2)
    assert valuation.athletic_impact.status is EvidenceStatus.ESTIMATED
    assert (
        valuation.program_value is not None
        and valuation.program_value.status is EvidenceStatus.SCENARIO
    )
    assert valuation.roster_market_value is not None and valuation.surplus is not None
    assert valuation.price_basis == "allocation"
    assert valuation.war.lower <= valuation.war.value <= valuation.war.upper
    assert "Drivers" in valuation.summary()
    assert "market.performance_exponent" in valuation.assumptions_used


def test_market_is_absent_before_revenue_sharing(registry):
    frames, _ = make_raw_season(season=2025)
    model = assemble_season(frames, registry)
    name = model.rapm.table.filter(~pl.col("pooled"))["name"][0]
    valuation = value_player(model, None, registry, name)
    assert valuation.roster_market_value is None and valuation.program_value is None
    assert any("roster-budget" in w for w in valuation.warnings)


def _small_label_registry(tmp_path):
    from athletevalue.assumptions.registry import AssumptionRegistry

    override = tmp_path / "labels.toml"
    override.write_text(
        '["market.model.min_labels"]\ndescription = "test"\nunit = "count"\n'
        'basis = "user_input"\nvalue = 20.0\n\n'
        '["market.model.min_schools"]\ndescription = "test"\nunit = "count"\n'
        'basis = "user_input"\nvalue = 4.0\n',
        encoding="utf-8",
    )
    return AssumptionRegistry.load(extra_paths=(override,))


def _deals(season_model, pay_from_net, seed=0):
    from athletevalue.schemas.registry import DealRecord

    rng = np.random.default_rng(seed)
    table = season_model.rapm.table.filter(~pl.col("pooled"))
    deals = []
    for k, row in enumerate(table.iter_rows(named=True)):
        pay = float(np.exp(pay_from_net(row["net"], rng)))
        deals.append(
            DealRecord(
                deal_id=f"t{k}",
                athlete_name=row["name"],
                season=season_model.season,
                school=row["team"],
                sport="mbb",
                cash_value=pay,
                duration_months=12,
                deal_type="revenue_share",
                source_url="https://example.org/deal",
                source_quality="named_report",
            )
        )
    return deals


def test_fitted_market_model_sets_the_price_when_it_beats_baselines(season, economics, tmp_path):
    from athletevalue.valuation.market_fit import fit_market, match_labels, player_features

    model, _ = season
    registry = _small_label_registry(tmp_path)
    deals = _deals(model, lambda net, rng: 12.5 + 0.25 * net + rng.normal(0, 0.05))
    deals.append(deals[0].model_copy(update={"deal_id": "ghost", "athlete_name": "Nobody Here"}))
    features = player_features(model, economics, registry)
    matched = match_labels(deals, features, registry)
    assert any("Nobody Here" in line for line in matched.unmatched)

    # A second deal for the same player-season with a different spelling adds to one label.
    respelled = deals[1].model_copy(
        update={"deal_id": "respelled", "athlete_name": deals[1].athlete_name.upper()}
    )
    merged = match_labels([*deals, respelled], features, registry).labels
    assert merged.height == matched.labels.height
    assert merged.filter(pl.col("n_deals") == 2).height == 1

    fit = fit_market(features, deals, registry)
    usable, note = fit.usable(registry)
    assert usable, note
    name = model.rapm.table.filter(~pl.col("pooled"))["name"][0]
    valuation = value_player(model, economics, registry, name, deals=deals, market_fit=fit, seed=3)
    assert valuation.price_basis == "registry"  # the player's own deal outranks the model
    assert valuation.allocated_market_value is not None  # model value shown, allocation kept

    other = [d for d in deals if d.athlete_name != name]
    fit_without = fit_market(features, other, registry)
    valuation = value_player(model, economics, registry, name, market_fit=fit_without, seed=3)
    assert valuation.price_basis == "fitted_model"
    assert valuation.allocated_market_value is not None
    assert valuation.roster_market_value.method.startswith("fitted_market_model")
    assert "  allocation" in valuation.summary()
    assert "market.model.min_labels" in valuation.assumptions_used


def test_noise_labels_do_not_replace_the_allocation(season, economics, tmp_path):
    from athletevalue.valuation.market_fit import fit_market, player_features

    model, _ = season
    registry = _small_label_registry(tmp_path)
    deals = _deals(model, lambda net, rng: 12.5 + rng.normal(0, 1.0), seed=4)
    fit = fit_market(player_features(model, economics, registry), deals, registry)
    usable, note = fit.usable(registry)
    assert not usable, note
    name = model.rapm.table.filter(~pl.col("pooled"))["name"][1]
    valuation = value_player(model, economics, registry, name, market_fit=fit, seed=3)
    assert valuation.price_basis == "allocation"
    assert any("fitted market model not used" in w for w in valuation.warnings)


def test_allocation_gate_uses_the_labels_the_allocation_prices(season, economics, tmp_path):
    from athletevalue.valuation.market_fit import fit_market, player_features

    model, _ = season
    registry = _small_label_registry(tmp_path)
    deals = _deals(model, lambda net, rng: 12.5 + 0.25 * net + rng.normal(0, 0.05))
    features = player_features(model, economics, registry)
    fit = fit_market(features, deals, registry)
    # The allocation prices every label but one exactly; the comparison runs on the rest
    # instead of being skipped, and the model cannot beat an exact price.
    exact = fit.labels.select("athlete_id", "season", pl.col("log_pay").exp().alias("price"))
    fit = fit_market(features, deals, registry, allocation_medians=exact.slice(1))
    assert fit.n_allocated == fit.labels.height - 1
    usable, note = fit.usable(registry)
    assert not usable and "does not beat the allocation" in note
