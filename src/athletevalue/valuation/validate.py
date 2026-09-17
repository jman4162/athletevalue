"""Gates that compare a season fit with published references.

Each gate states what it measures, the threshold, where the threshold comes from
and whether the fit passed. A failed gate does not stop a valuation; it is
reported so a reader can decide how much to trust the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.identity.names import normalize_name
from athletevalue.impact.cv import cv_prior_comparison, game_folds
from athletevalue.impact.prior import weighted_r2
from athletevalue.valuation.checks import (
    bid_calibration,
    player_war_table,
    returning_players,
    team_war_table,
)
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.season import SeasonModel


@dataclass(frozen=True)
class Gate:
    name: str
    value: float
    low: float | None
    high: float | None
    detail: str

    @property
    def passed(self) -> bool:
        above = self.low is None or self.value >= self.low
        below = self.high is None or self.value <= self.high
        return above and below


@dataclass(frozen=True)
class ValidationReport:
    season: int
    gates: tuple[Gate, ...]
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)


def validate_season(
    model: SeasonModel,
    registry: AssumptionRegistry,
    *,
    reference_rapm: pl.DataFrame | None = None,
    torvik: pl.DataFrame | None = None,
    check_prior: bool = True,
    seed: int = 0,
) -> ValidationReport:
    rapm = model.rapm
    gates: list[Gate] = []
    notes: list[str] = []

    low, high = registry.get("mbb.impact.intercept_band").interval()
    gates.append(Gate("intercept", rapm.intercept, low, high, "points per 100 possessions"))
    low, high = registry.get("mbb.impact.home_court_band").interval()
    gates.append(Gate("home_court", rapm.home_court, low, high, "per side, points per 100"))
    low, high = registry.get("mbb.impact.sigma2_band").interval()
    gates.append(Gate("sigma2", rapm.sigma2, low, high, "residual variance, (pts/100)^2"))

    if reference_rapm is not None:
        threshold = registry.get("mbb.impact.reference_min_possessions").scalar()
        # The reference shrinks toward zero, so it is compared with the no-prior fit.
        joined = model.baseline.table.join(
            reference_rapm.select(
                pl.col("player_id").cast(pl.Utf8).alias("athlete_id"),
                pl.col("rapm_net").alias("reference_net"),
                pl.col("off_poss").alias("reference_off_poss"),
            ),
            on="athlete_id",
            how="inner",
        ).filter(pl.col("reference_off_poss") >= threshold)
        rho = float(spearmanr(joined["net"], joined["reference_net"])[0])
        gates.append(
            Gate(
                "spearman_vs_reference_rapm",
                rho,
                registry.get("mbb.impact.reference_spearman_min").scalar(),
                None,
                f"{joined.height} players with >= {threshold:.0f} offensive possessions",
            )
        )

    if torvik is not None:
        ours = model.teams.select("team", "adj_net").with_columns(
            pl.col("team").map_elements(normalize_name, return_dtype=pl.Utf8).alias("key")
        )
        theirs = torvik.with_columns(
            pl.col("team").map_elements(normalize_name, return_dtype=pl.Utf8).alias("key")
        ).select("key", "adj_em")
        joined = ours.join(theirs, on="key", how="inner")
        rho = float(spearmanr(joined["adj_net"], joined["adj_em"])[0])
        gates.append(
            Gate(
                "spearman_team_net_vs_torvik",
                rho,
                registry.get("mbb.impact.torvik_spearman_min").scalar(),
                None,
                f"{joined.height} of {ours.height} teams matched by name",
            )
        )
        gates.append(
            Gate(
                "torvik_teams_matched",
                float(joined.height),
                registry.get("mbb.impact.torvik_min_teams").scalar(),
                None,
                "count",
            )
        )

    if check_prior and model.prior is not None and model.prior_offset is not None:
        gate, note = _prior_checks(model, registry, seed=seed)
        gates.append(gate)
        notes.append(note)

    ref_low, ref_high = registry.get("mbb.wins.pythag_exponent_reference").interval()
    notes.append(
        f"Pythagorean exponent fitted on raw efficiencies: {model.exponent:.2f}. Pomeroy's "
        f"ratings page reports {ref_low:g}-{ref_high:g} for schedule-adjusted efficiencies; raw "
        "efficiencies give lower exponents because strong teams face strong schedules."
    )
    levels = ", ".join(
        f"{k.replace('_', ' ')} {v:+.1f}" for k, v in model.replacement_levels.items()
    )
    notes.append(
        f"Replacement level in use: {model.replacement_definition.replace('_', ' ')} "
        f"({model.replacement:+.1f} per 100). Alternatives: {levels}. WAR and every dollar "
        "figure scale with this choice."
    )
    scope = "D1 games only" if registry.get("mbb.wins.margin_sd_d1_only").flag() else "all games"
    notes.append(f"Game margin SD around fitted ratings: {model.margin_sd:.1f} points ({scope}).")
    for note in model.notes:
        notes.append(note)
    return ValidationReport(season=model.season, gates=tuple(gates), notes=tuple(notes))


def _prior_checks(
    model: SeasonModel, registry: AssumptionRegistry, *, seed: int
) -> tuple[Gate, str]:
    assert model.prior is not None and model.prior_offset is not None
    design = model.design
    folds = game_folds(
        design.groups,
        int(registry.get("mbb.impact.cv_folds").scalar()),
        np.random.default_rng(seed),
    )
    lam_none = registry.get("mbb.impact.ridge_lambda").scalar()
    predictions_for = model.fold_predictions()
    assert predictions_for is not None and model.box_predictions is not None
    # Box totals and the team adjustment are both rebuilt from each fold's training
    # games, so no held-out outcome reaches the prior it is scored against.
    without, with_prior = cv_prior_comparison(
        design,
        model.lineups,
        predictions_for,
        folds,
        lam_without=lam_none,
        lam_with=model.rapm.lam,
        adjust_to_team=registry.get("mbb.prior.team_adjustment").flag(),
    )
    gate = Gate(
        "prior_cv_error_ratio",
        with_prior / without,
        None,
        registry.get("mbb.prior.max_cv_error_ratio").scalar(),
        f"held-out MSE {with_prior:,.1f} with prior (lambda {model.rapm.lam:g}) "
        f"vs {without:,.1f} without (lambda {lam_none:g})",
    )

    threshold = registry.get("mbb.impact.reference_min_possessions").scalar()
    # The unadjusted predictions: what box scores alone, through a model fitted on
    # other seasons, say about this season's no-prior ratings. The team-adjusted
    # prior would score far higher because it is solved to match those ratings.
    joined = model.baseline.table.join(model.box_predictions, on="athlete_id").filter(
        pl.col("off_poss") >= threshold
    )
    r2_off = weighted_r2(
        joined["orapm"].to_numpy(),
        joined["prior_off"].to_numpy(),
        joined["off_poss"].cast(pl.Float64).to_numpy(),
    )
    r2_def = weighted_r2(
        joined["drapm"].to_numpy(),
        joined["prior_def"].to_numpy(),
        joined["def_poss"].cast(pl.Float64).to_numpy(),
    )
    seasons = ", ".join(str(s) for s in model.prior.train_seasons)
    note = (
        f"Box model fitted on {seasons} explains {r2_off:.0%} of offensive and {r2_def:.0%} of "
        f"defensive no-prior ratings this season before the team adjustment "
        f"({joined.height} players with >= {threshold:.0f} possessions). The adjustment is "
        "solved to match team ratings, so it is not scored here."
    )
    return gate, note


def extended_gates(
    model: SeasonModel,
    registry: AssumptionRegistry,
    *,
    previous: SeasonModel | None = None,
    economics: EconomicsModel | None = None,
) -> tuple[tuple[Gate, ...], tuple[str, ...]]:
    """Gates on the layers above the ratings: wins, returning players, revenue, bids.

    *previous* is the season before *model*, fitted with its default prior; without it
    the returning-player gate is skipped. Without *economics* the revenue and bid
    gates are skipped. Each skip is reported as a note.
    """
    gates: list[Gate] = []
    notes: list[str] = []

    low, high = registry.get("mbb.wins.margin_sd_band").interval()
    gates.append(Gate("margin_sd", model.margin_sd, low, high, "points, D1 game margins"))

    players = player_war_table(
        model, floor_non_d1=registry.get("mbb.wins.non_d1_opponent_floor").flag()
    )
    teams = team_war_table(model, players)
    correlation = float(np.corrcoef(teams["war"], teams["wins"])[0, 1])
    slope, intercept = np.polyfit(teams["war"].to_numpy(), teams["wins"].to_numpy(), 1)
    gates.append(
        Gate(
            "team_war_vs_wins_correlation",
            correlation,
            registry.get("mbb.wins.team_war_min_correlation").scalar(),
            None,
            f"{teams.height} teams; wins = {intercept:.1f} + {slope:.2f} x summed linear WAR",
        )
    )

    if previous is None:
        notes.append("Returning-player gate skipped: no earlier season supplied.")
    else:
        returning = returning_players(
            previous,
            model,
            min_possessions_earlier=int(
                registry.get("mbb.prior.returning_min_possessions_earlier").scalar()
            ),
            min_possessions_later=int(
                registry.get("mbb.prior.returning_min_possessions_later").scalar()
            ),
        )
        if returning is None:
            notes.append("Returning-player gate skipped: person ids are missing for a season.")
        else:
            gates.append(
                Gate(
                    "returning_player_prior_gain",
                    returning.gain,
                    registry.get("mbb.prior.returning_min_gain").scalar(),
                    None,
                    f"{returning.n_players} players from {returning.earlier_season}: "
                    f"correlation with {returning.later_season} no-prior ratings "
                    f"{returning.corr_with_prior:.3f} with prior vs "
                    f"{returning.corr_without_prior:.3f} without",
                )
            )

    if economics is None:
        notes.append("Revenue and bid gates skipped: program economics were not loaded.")
    else:
        share = float(np.mean(economics.revenue.win_effect_current() <= 0))
        gates.append(
            Gate(
                "revenue_win_effect_nonpositive_share",
                share,
                None,
                registry.get("economics.max_prob_nonpositive_win_effect").scalar(),
                f"of {economics.revenue.draws.shape[0]} school bootstrap draws",
            )
        )
        calibration = bid_calibration(economics, registry)
        gaps = (calibration["predicted"] - calibration["observed"]).abs().to_numpy()
        gap = float(gaps.max())
        gates.append(
            Gate(
                "bid_calibration_max_decile_gap",
                gap,
                None,
                registry.get("economics.bid_calibration_max_gap").scalar(),
                f"{int(calibration['n'].sum())} team-seasons in {calibration.height} "
                "deciles, in sample",
            )
        )
    return tuple(gates), tuple(notes)
