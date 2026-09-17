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
        f"Pythagorean exponent fitted on raw efficiencies: {model.exponent:.2f}. Published "
        f"values for schedule-adjusted efficiencies are {ref_low:g}-{ref_high:g}; raw "
        "efficiencies give lower exponents because strong teams face strong schedules."
    )
    replacement = registry.get("mbb.wins.replacement_level").scalar()
    notes.append(
        f"Pooled low-minute players rate {rapm.pool_net:+.1f} per 100 against the "
        f"{replacement:+.1f} replacement convention."
    )
    notes.append(f"Game margin SD around fitted ratings: {model.margin_sd:.1f} points.")
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
    assert model.box_predictions is not None
    without, with_prior = cv_prior_comparison(
        design,
        model.lineups,
        model.box_predictions,
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
    p = design.n_players
    prior = pl.DataFrame(
        {
            "athlete_id": list(design.athlete_ids),
            "prior_off": model.prior_offset[:p],
            "prior_def": model.prior_offset[p : 2 * p],
        }
    )
    joined = model.baseline.table.join(prior, on="athlete_id").filter(
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
        f"Box prior fitted on {seasons}, after the team adjustment from this season's games, "
        f"explains {r2_off:.0%} of offensive and {r2_def:.0%} of defensive no-prior ratings "
        f"({joined.height} players with >= {threshold:.0f} possessions; in-season, so optimistic)."
    )
    return gate, note
