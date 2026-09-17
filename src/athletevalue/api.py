"""Public entry points that load data and run the models.

A ``Session`` holds one download cache and one assumption registry, and remembers
the seasons and economics models it has fitted, so repeated valuations reuse them::

    from athletevalue.api import Session

    session = Session()
    first = session.value_player("A. Guard", 2026, team="State U")
    second = session.value_player("B. Wing", 2026, team="State U")  # no refit

The module-level functions do the same work. Called without ``cache`` or
``registry`` they share one default session; with either argument they use a
fresh session for that call.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Hashable
from pathlib import Path

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry, default_registry
from athletevalue.identity.resolve import resolve_team
from athletevalue.impact.prior import PriorTrainingError
from athletevalue.market.model import InsufficientLabelsError
from athletevalue.market.registry import load_deal_registry
from athletevalue.runtime import numerical_threads
from athletevalue.schemas.registry import DealRecord
from athletevalue.sources.cache import ArtifactCache, ArtifactMismatchError, SourceUnavailableError
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.sources.torvik import TorvikClient
from athletevalue.sports.mbb.economics_loaders import load_economics_frames
from athletevalue.sports.mbb.loaders import load_box_prior, load_season_frames
from athletevalue.valuation.economy import EconomicsModel, assemble_economics
from athletevalue.valuation.market_fit import MarketFit, match_labels, player_features
from athletevalue.valuation.market_fit import fit_market as fit_market_labels
from athletevalue.valuation.player import team_table
from athletevalue.valuation.player import value_player as compose_player
from athletevalue.valuation.result import PlayerValuation
from athletevalue.valuation.season import SeasonModel, assemble_season
from athletevalue.valuation.team import team_draws
from athletevalue.valuation.validate import ValidationReport, validate_season
from athletevalue.versions import MODEL_VERSION


class _Memo[V]:
    """A small least-recently-used map."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._items: OrderedDict[Hashable, V] = OrderedDict()

    def get(self, key: Hashable) -> V | None:
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: Hashable, value: V) -> None:
        if self.capacity == 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()


class Session:
    """One cache, one registry, and the models fitted with them.

    Fitted seasons are keyed on the season, the penalty settings, the prior, the
    prior's training cap, the registry digest and the model version; economics
    models on the last season, the registry digest and the model version. The
    least recently used entries are dropped beyond *max_seasons* and
    *max_economics*; a season model for a full year holds a few hundred megabytes.
    """

    def __init__(
        self,
        *,
        cache: ArtifactCache | None = None,
        registry: AssumptionRegistry | None = None,
        offline: bool = False,
        allow_unpinned: bool = False,
        max_seasons: int = 3,
        max_economics: int = 2,
    ) -> None:
        self.cache = cache or ArtifactCache.default(offline=offline, allow_unpinned=allow_unpinned)
        self.registry = registry or default_registry()
        self._seasons: _Memo[SeasonModel] = _Memo(max_seasons)
        self._economics: _Memo[EconomicsModel] = _Memo(max_economics)
        self._markets: _Memo[MarketFit] = _Memo(max_economics)

    @property
    def cached_seasons(self) -> int:
        return len(self._seasons)

    def clear(self) -> None:
        """Forget fitted models (files on disk are untouched)."""
        self._seasons.clear()
        self._economics.clear()
        self._markets.clear()

    def fit_season(
        self,
        season: int,
        *,
        lam: float | None = None,
        choose_lambda_by_cv: bool = False,
        prior: bool | None = None,
        last_available_season: int | None = None,
    ) -> SeasonModel:
        """Download (or reuse) one season's inputs and fit ratings, team ratings and win models.

        *prior* defaults to the registry's ``mbb.prior.use_box_prior``. The prior is fit on
        seasons before *season* only; *last_available_season* can lower that cap further
        (for example to reproduce what was knowable at an earlier date). The first season
        with data has no earlier season and is fitted without a prior.
        """
        use_prior = self.registry.get("mbb.prior.use_box_prior").flag() if prior is None else prior
        cap = (
            season - 1 if last_available_season is None else min(last_available_season, season - 1)
        )
        key = (
            season,
            lam,
            choose_lambda_by_cv,
            use_prior,
            cap,
            self.registry.digest,
            MODEL_VERSION,
        )
        cached = self._seasons.get(key)
        if cached is not None:
            return cached
        with numerical_threads():
            frames = load_season_frames(season, self.cache)
            box_prior = None
            notes: list[str] = []
            if use_prior:
                try:
                    box_prior = load_box_prior(season, self.cache, self.registry, last_season=cap)
                except PriorTrainingError as error:
                    notes.append(f"no box-score prior: {error}")
            model = assemble_season(
                frames,
                self.registry,
                lam=lam,
                choose_lambda_by_cv=choose_lambda_by_cv,
                prior=box_prior,
                notes=tuple(notes),
            )
        self._seasons.put(key, model)
        return model

    def fit_economics(self, last_season: int) -> EconomicsModel:
        """Fit the revenue and tournament-bid models on every season through *last_season*."""
        key = (last_season, self.registry.digest, MODEL_VERSION)
        cached = self._economics.get(key)
        if cached is not None:
            return cached
        with numerical_threads():
            model = assemble_economics(
                load_economics_frames(self.cache, last_season=last_season), self.registry
            )
        self._economics.put(key, model)
        return model

    def validate(self, model: SeasonModel, *, use_torvik: bool = True) -> ValidationReport:
        """Compare a fitted season with the SportsDataverse RAPM and Torvik team ratings."""
        reference = SdvClient(self.cache).frame(SdvDataset.REFERENCE_RAPM, model.season)
        torvik = None
        notes: tuple[str, ...] = ()
        if use_torvik:
            try:
                torvik = TorvikClient(self.cache).team_results(model.season)
            except (SourceUnavailableError, ArtifactMismatchError) as error:
                # barttorvik.com refuses some cloud address ranges, and its files can be
                # revised after the pinned copy; the comparison is optional and its
                # absence is reported rather than hidden.
                notes = (f"Torvik comparison skipped: {error}",)
        with numerical_threads():
            report = validate_season(model, self.registry, reference_rapm=reference, torvik=torvik)
        return ValidationReport(
            season=report.season, gates=report.gates, notes=(*report.notes, *notes)
        )

    def fit_market(
        self,
        deals: list[DealRecord] | None = None,
        *,
        labels: Path | None = None,
        fitted: dict[int, SeasonModel] | None = None,
        economics: EconomicsModel | None = None,
    ) -> MarketFit:
        """Fit the roster-market model on disclosed men's basketball deals.

        Raises ``InsufficientLabelsError`` when the labels do not meet the registry
        minimums; the valuation then keeps the allocation as the price.
        """
        deals = load_deals(labels) if deals is None else deals
        seasons = _label_seasons(deals, self.registry)
        econ = economics or self.fit_economics(max(seasons))
        key = None
        if fitted is None and economics is None:
            key = (_deals_digest(deals), self.registry.digest, MODEL_VERSION)
            cached = self._markets.get(key)
            if cached is not None:
                return cached
        models = dict(fitted or {})
        with numerical_threads():
            features = []
            for season in seasons:
                if season not in models:
                    models[season] = self.fit_season(season)
                features.append(player_features(models[season], econ, self.registry))
            table = pl.concat(features, how="vertical_relaxed")
            matched = match_labels(deals, table, self.registry)
            prices = []
            for season, team in matched.labels.select("season", "team").unique().iter_rows():
                allocation = team_draws(
                    models[season], econ, self.registry, team, seed=0
                ).allocation
                if allocation is None:
                    continue
                prices.append(
                    pl.DataFrame(
                        {
                            "athlete_id": list(allocation.athlete_ids),
                            "season": [season] * len(allocation.athlete_ids),
                            "price": np.median(allocation.pay, axis=0),
                        },
                        schema_overrides={"season": pl.Int64},
                    )
                )
            fit = fit_market_labels(
                table,
                deals,
                self.registry,
                allocation_medians=pl.concat(prices) if prices else None,
                matched=matched,
            )
        if key is not None:
            self._markets.put(key, fit)
        return fit

    def value_player(
        self,
        name: str,
        season: int,
        *,
        team: str | None = None,
        sport: str = "mbb",
        labels: Path | None = None,
        economics: bool = True,
        seed: int = 0,
        last_available_season: int | None = None,
        model: SeasonModel | None = None,
        economics_model: EconomicsModel | None = None,
    ) -> PlayerValuation:
        """Value one player-season: impact, wins, program value, market price and surplus.

        The price comes from, in order: disclosed pay for this player-season, the fitted
        market model when it passes its gates, then the roster-budget allocation.
        *model* and *economics_model* reuse models fitted elsewhere; *economics* set to
        ``False`` skips program value.
        """
        if sport != "mbb":
            raise ValueError(f"sport {sport!r} is not modelled yet; this release covers mbb")
        model = self._season_or(model, season, last_available_season)
        econ = economics_model or (self.fit_economics(season) if economics else None)
        deals = load_deals(labels)
        market_fit: MarketFit | None = None
        market_econ = econ
        note: str | None = None
        try:
            last_label_season = max(_label_seasons(deals, self.registry))
            if market_econ is None or last_label_season > season:
                # The model is trained on revenue features, so the player's own features
                # need the same economics even when program value is switched off.
                market_econ = self.fit_economics(max(last_label_season, season))
            market_fit = self.fit_market(deals, fitted={season: model}, economics=market_econ)
        except InsufficientLabelsError as error:
            note = str(error)
        with numerical_threads():
            return compose_player(
                model,
                econ,
                self.registry,
                name,
                team=team,
                deals=deals,
                market_fit=market_fit,
                market_note=note,
                market_economics=market_econ,
                seed=seed,
            )

    def value_team(
        self,
        team: str,
        season: int,
        *,
        economics: bool = True,
        seed: int = 0,
        model: SeasonModel | None = None,
        economics_model: EconomicsModel | None = None,
    ) -> pl.DataFrame:
        """Value-vs-price table (medians) for every player on *team*."""
        model = self._season_or(model, season, None)
        econ = economics_model or (self.fit_economics(season) if economics else None)
        resolved = resolve_team(model.teams, team)
        with numerical_threads():
            return team_table(team_draws(model, econ, self.registry, resolved, seed=seed))

    def _season_or(
        self, model: SeasonModel | None, season: int, last_available_season: int | None
    ) -> SeasonModel:
        if model is None:
            return self.fit_season(season, last_available_season=last_available_season)
        if model.season != season:
            raise ValueError(f"model is for season {model.season}, not {season}")
        return model


_default: Session | None = None


def default_session() -> Session:
    """The session module-level calls share when given no cache or registry."""
    global _default
    root = ArtifactCache.default().root
    if _default is None or _default.cache.root != root:
        _default = Session()
    return _default


def _session(cache: ArtifactCache | None, registry: AssumptionRegistry | None) -> Session:
    if cache is None and registry is None:
        return default_session()
    return Session(cache=cache, registry=registry)


def fit_season(
    season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    lam: float | None = None,
    choose_lambda_by_cv: bool = False,
    prior: bool | None = None,
    last_available_season: int | None = None,
) -> SeasonModel:
    """See ``Session.fit_season``."""
    return _session(cache, registry).fit_season(
        season,
        lam=lam,
        choose_lambda_by_cv=choose_lambda_by_cv,
        prior=prior,
        last_available_season=last_available_season,
    )


def validate(
    model: SeasonModel,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    use_torvik: bool = True,
) -> ValidationReport:
    """See ``Session.validate``."""
    return _session(cache, registry).validate(model, use_torvik=use_torvik)


def fit_economics(
    last_season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
) -> EconomicsModel:
    """See ``Session.fit_economics``."""
    return _session(cache, registry).fit_economics(last_season)


def load_deals(labels: Path | None = None) -> list[DealRecord]:
    """Deals from the packaged registry, plus a user CSV in the same schema if given."""
    deals = load_deal_registry()
    if labels is not None:
        # Read every column as text: numeric ids such as stats.ncaa.org athlete ids would
        # otherwise load as integers and fail the string fields of DealRecord.
        deals += load_deal_registry(pl.read_csv(labels, infer_schema_length=0))
    return deals


def fit_market(
    deals: list[DealRecord] | None = None,
    *,
    labels: Path | None = None,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    fitted: dict[int, SeasonModel] | None = None,
    economics: EconomicsModel | None = None,
) -> MarketFit:
    """See ``Session.fit_market``."""
    return _session(cache, registry).fit_market(
        deals, labels=labels, fitted=fitted, economics=economics
    )


def value_player(
    name: str,
    season: int,
    *,
    team: str | None = None,
    sport: str = "mbb",
    labels: Path | None = None,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    economics: bool = True,
    seed: int = 0,
    last_available_season: int | None = None,
    model: SeasonModel | None = None,
    economics_model: EconomicsModel | None = None,
) -> PlayerValuation:
    """See ``Session.value_player``."""
    return _session(cache, registry).value_player(
        name,
        season,
        team=team,
        sport=sport,
        labels=labels,
        economics=economics,
        seed=seed,
        last_available_season=last_available_season,
        model=model,
        economics_model=economics_model,
    )


def value_team(
    team: str,
    season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    economics: bool = True,
    seed: int = 0,
    model: SeasonModel | None = None,
    economics_model: EconomicsModel | None = None,
) -> pl.DataFrame:
    """See ``Session.value_team``."""
    return _session(cache, registry).value_team(
        team,
        season,
        economics=economics,
        seed=seed,
        model=model,
        economics_model=economics_model,
    )


def _label_seasons(deals: list[DealRecord], registry: AssumptionRegistry) -> list[int]:
    """Seasons of the deals that can become labels; raises below the registry minimum."""
    types = set(registry.get("market.model.label_deal_types").names())
    labeled = [d for d in deals if d.sport == "mbb" and d.deal_type.value in types]
    minimum = int(registry.get("market.model.min_labels").scalar())
    if not labeled or len(labeled) < minimum:
        raise InsufficientLabelsError(
            f"{len(labeled)} disclosed deals available; at least {minimum} labeled "
            "player-seasons needed"
        )
    return sorted({d.season for d in labeled})


def _deals_digest(deals: list[DealRecord]) -> str:
    payload = json.dumps([deal.model_dump(mode="json") for deal in deals], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ArtifactCache",
    "ArtifactMismatchError",
    "AssumptionRegistry",
    "EconomicsModel",
    "InsufficientLabelsError",
    "MarketFit",
    "PlayerValuation",
    "SeasonModel",
    "Session",
    "ValidationReport",
    "default_session",
    "fit_economics",
    "fit_market",
    "fit_season",
    "load_deals",
    "validate",
    "value_player",
    "value_team",
]
