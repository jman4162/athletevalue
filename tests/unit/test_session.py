"""Session memoization against a synthetic offline cache."""

from __future__ import annotations

import time

import pytest

from athletevalue import api
from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.sources.cache import CACHE_ENV, ArtifactCache


@pytest.fixture
def registry_one_prior(one_prior_season):
    return AssumptionRegistry.load(extra_paths=(one_prior_season,))


@pytest.fixture
def session(synthetic_cache_dir, registry_one_prior):
    cache = ArtifactCache(synthetic_cache_dir, offline=True)
    return api.Session(cache=cache, registry=registry_one_prior)


@pytest.fixture
def season_fits(monkeypatch):
    calls: list[int] = []
    original = api.assemble_season

    def counting(frames, *args, **kwargs):
        calls.append(frames.season)
        return original(frames, *args, **kwargs)

    monkeypatch.setattr(api, "assemble_season", counting)
    return calls


def test_repeat_fits_reuse_the_model(session, season_fits):
    first = session.fit_season(2042)
    assert session.fit_season(2042) is first
    assert session.fit_season(2042, prior=False) is not first
    assert season_fits == [2042, 2042]
    assert session.cached_seasons == 2
    session.clear()
    assert session.cached_seasons == 0


def test_least_recently_used_model_is_dropped(synthetic_cache_dir, registry_one_prior, season_fits):
    small = api.Session(
        cache=ArtifactCache(synthetic_cache_dir, offline=True),
        registry=registry_one_prior,
        max_seasons=1,
    )
    small.fit_season(2041, prior=False)
    small.fit_season(2042)
    small.fit_season(2041, prior=False)
    assert season_fits == [2041, 2042, 2041]


def test_second_valuation_does_not_refit(session, season_fits):
    session.value_player("BigU0 Player1", 2042, economics=False)
    start = time.perf_counter()
    other = session.value_player("WCCU1 Player2", 2042, economics=False)
    assert time.perf_counter() - start < 1.0
    assert other.player.name == "WCCU1 Player2"
    assert season_fits == [2042]
    table = session.value_team("Big U0", 2042, economics=False)
    assert table.height == 9
    assert season_fits == [2042]


def test_a_given_model_is_used_as_is_and_must_match_the_season(session, season_fits):
    model = session.fit_season(2041, prior=False)
    valuation = session.value_player("BigU0 Player1", 2041, economics=False, model=model)
    assert valuation.player.season == 2041
    with pytest.raises(ValueError, match="season 2041"):
        session.value_team("Big U0", 2042, economics=False, model=model)
    assert season_fits == [2041]


def test_registry_changes_are_part_of_the_key(synthetic_cache_dir, registry_one_prior):
    cache = ArtifactCache(synthetic_cache_dir, offline=True)
    first = api.Session(cache=cache, registry=registry_one_prior).fit_season(2041, prior=False)
    default = api.Session(cache=cache).fit_season(2041, prior=False)
    assert first is not default


def test_module_functions_share_the_default_session(
    synthetic_cache_dir, monkeypatch, tmp_path, season_fits
):
    monkeypatch.setenv(CACHE_ENV, str(synthetic_cache_dir))
    monkeypatch.setattr(api, "_default", None)
    first = api.fit_season(2041, prior=False)
    assert api.fit_season(2041, prior=False) is first
    assert api.default_session().cache.root == synthetic_cache_dir
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    assert api.default_session().cache.root == tmp_path
    assert season_fits == [2041]
