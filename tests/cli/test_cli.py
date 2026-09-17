"""Every CLI command against a synthetic offline cache."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from athletevalue.cli import main as cli
from athletevalue.sources.cache import OfflineError

runner = CliRunner()


@pytest.fixture
def cache_dir(synthetic_cache_dir, tmp_path):
    """A private copy, so commands that write derived files or clear the cache stay isolated."""
    target = tmp_path / "cache"
    shutil.copytree(synthetic_cache_dir, target)
    return target


def invoke(cache_dir, *args, stdin=None):
    return runner.invoke(cli.app, ["--cache-dir", str(cache_dir), "--offline", *args], input=stdin)


def test_fit_prints_ratings_and_the_prior_seasons(cache_dir, one_prior_season, tmp_path):
    output = tmp_path / "ratings.parquet"
    result = invoke(
        cache_dir,
        "fit",
        "--season",
        "2042",
        "--top",
        "3",
        "--output",
        str(output),
        "-a",
        str(one_prior_season),
    )
    assert result.exit_code == 0, result.output
    assert "box prior fitted on 2041" in result.output
    assert "  1. " in result.output and "  4. " not in result.output
    assert output.exists()


def test_fit_without_prior_and_with_cv(cache_dir):
    result = invoke(cache_dir, "fit", "--season", "2041", "--no-prior", "--cv", "--top", "1")
    assert result.exit_code == 0, result.output
    assert "prior none" in result.output
    assert "held-out weighted MSE" in result.output


def test_validate_prints_gates(cache_dir, one_prior_season):
    result = invoke(
        cache_dir, "validate", "--season", "2042", "--no-torvik", "-a", str(one_prior_season)
    )
    assert result.exit_code in (0, 1), result.output
    assert "spearman_vs_reference_rapm" in result.output
    assert "prior_cv_error_ratio" in result.output


def test_extended_validation_without_economics(cache_dir, one_prior_season):
    result = invoke(
        cache_dir,
        "validate",
        "--season",
        "2042",
        "--no-torvik",
        "--extended",
        "--no-economics",
        "-a",
        str(one_prior_season),
    )
    assert result.exit_code in (0, 1), result.output
    assert "team_war_vs_wins_correlation" in result.output
    assert "previous season unavailable" in result.output
    assert "Revenue and bid gates skipped" in result.output


def test_value_summary_and_json(cache_dir, one_prior_season):
    args = ("--season", "2042", "--no-economics", "-a", str(one_prior_season))
    summary = invoke(cache_dir, "value", "BigU0 Player1", *args)
    assert summary.exit_code == 0, summary.output
    assert "BigU0 Player1" in summary.output
    payload = json.loads(invoke(cache_dir, "value", "BigU0 Player1", "--json", *args).output)
    assert payload["player"]["season"] == 2042
    assert payload["program_value"] is None


def test_team_without_economics(cache_dir, one_prior_season):
    result = invoke(
        cache_dir,
        "team",
        "Big U0",
        "--season",
        "2042",
        "--no-economics",
        "-a",
        str(one_prior_season),
    )
    assert result.exit_code == 0, result.output
    assert "Program value is off" in result.output
    assert result.output.count("BigU0 Player") == 9
    assert "n/a" in result.output


def test_fetch_reports_unpinned_files_and_prior_seasons(cache_dir, one_prior_season):
    result = invoke(
        cache_dir, "fetch", "--season", "2042", "--no-economics", "-a", str(one_prior_season)
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("  unpinned\n") == 6
    assert "box-prior training seasons: 2041" in result.output


def test_market_fit_reports_missing_labels(cache_dir):
    result = invoke(cache_dir, "market-fit")
    assert result.exit_code == 1
    assert "not fitted" in result.output


def test_assumptions_and_cache_path(cache_dir):
    listed = invoke(cache_dir, "assumptions")
    assert listed.exit_code == 0
    assert "mbb.impact.ridge_lambda" in listed.output
    assert invoke(cache_dir, "cache-path").output.strip() == str(cache_dir)


def test_cache_info_verify_and_clear(cache_dir, one_prior_season):
    invoke(cache_dir, "fit", "--season", "2042", "--top", "1", "-a", str(one_prior_season))
    info = invoke(cache_dir, "cache", "info")
    assert "raw files         12" in info.output
    assert "derived files      1" in info.output
    verified = invoke(cache_dir, "cache", "verify")
    assert verified.exit_code == 0, verified.output

    (
        cache_dir / "raw/sportsdataverse/ncaa_mbb_team_ids/ncaa_mbb_team_ids_2042.parquet"
    ).write_bytes(b"damaged")
    damaged = invoke(cache_dir, "cache", "verify")
    assert damaged.exit_code == 1
    assert "contents differ from the manifest" in damaged.output

    declined = invoke(cache_dir, "cache", "clear", stdin="n\n")
    assert declined.exit_code == 1
    assert (cache_dir / "manifest.json").exists()
    derived = invoke(cache_dir, "cache", "clear", "--derived-only", "--yes")
    assert "removed 1 files" in derived.output
    cleared = invoke(cache_dir, "cache", "clear", "--yes")
    assert "removed 12 files" in cleared.output
    assert not (cache_dir / "raw").exists()


def test_missing_season_offline_is_an_error(cache_dir):
    result = invoke(cache_dir, "fit", "--season", "2043", "--no-prior")
    assert isinstance(result.exception, OfflineError)


def test_entry_point_prints_expected_errors_without_a_traceback(cache_dir, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "athletevalue",
            "--cache-dir",
            str(cache_dir),
            "--offline",
            "fit",
            "-s",
            "2043",
            "--no-prior",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.run()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "athletevalue fetch --season" in err
    assert "Traceback" not in err
