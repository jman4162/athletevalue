"""Command-line interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from athletevalue.api import InsufficientLabelsError, Session
from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.sources.cache import (
    CACHE_ENV,
    ArtifactCache,
    ArtifactMismatchError,
    OfflineError,
    SourceUnavailableError,
)
from athletevalue.sources.eada import EadaClient
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.sports.mbb.economics_loaders import load_economics_frames
from athletevalue.sports.mbb.loaders import fetch_prior_inputs
from athletevalue.valuation.report import money

app = typer.Typer(
    help="Separate what a college basketball player is worth to a program from what the market pays.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Inspect, verify or clear the download cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

SeasonOption = Annotated[
    int, typer.Option("--season", "-s", help="Ending year, e.g. 2026 for 2025-26.")
]
LastSeasonOption = Annotated[
    int | None,
    typer.Option(
        "--last-season",
        help="Latest season the box-score prior may learn from (default: the season before).",
    ),
]
AssumptionsOption = Annotated[
    list[Path] | None,
    typer.Option("--assumptions", "-a", help="TOML file overriding registry entries. Repeatable."),
]
EconomicsOption = Annotated[
    bool,
    typer.Option(
        "--economics/--no-economics",
        help="Fit the revenue model for program value (needs EADA files).",
    ),
]


@dataclass(frozen=True)
class CliOptions:
    offline: bool = False
    cache_dir: Path | None = None
    allow_unpinned: bool = False

    def cache(self) -> ArtifactCache:
        if self.cache_dir is None:
            return ArtifactCache.default(offline=self.offline, allow_unpinned=self.allow_unpinned)
        return ArtifactCache(
            self.cache_dir, offline=self.offline, allow_unpinned=self.allow_unpinned
        )

    def session(self, assumptions: list[Path] | None = None) -> Session:
        registry = AssumptionRegistry.load(extra_paths=tuple(assumptions or ()))
        return Session(cache=self.cache(), registry=registry)


def _options(ctx: typer.Context) -> CliOptions:
    return ctx.obj if isinstance(ctx.obj, CliOptions) else CliOptions()


@app.callback()
def main(
    ctx: typer.Context,
    offline: bool = typer.Option(
        False, "--offline", help="Use cached files only; fail instead of downloading."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", envvar=CACHE_ENV, help="Cache directory (default: the user cache)."
    ),
    allow_unpinned: bool = typer.Option(
        False,
        "--allow-unpinned",
        help="Accept upstream files whose SHA-256 differs from the digest pinned by this release.",
    ),
) -> None:
    ctx.obj = CliOptions(offline=offline, cache_dir=cache_dir, allow_unpinned=allow_unpinned)


@app.command()
def fetch(
    ctx: typer.Context,
    season: SeasonOption,
    economics: bool = typer.Option(True, help="Also fetch EADA and past seasons."),
    prior: bool = typer.Option(
        True, help="Also fetch the earlier seasons the box-score prior trains on."
    ),
    last_season: LastSeasonOption = None,
    assumptions: AssumptionsOption = None,
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Download this season's files and the EADA file list again, even if cached.",
    ),
) -> None:
    """Download and cache one season's inputs, so later commands can run with --offline."""
    cache = _options(ctx).cache()
    client = SdvClient(cache)
    for dataset in (
        SdvDataset.POSSESSIONS,
        SdvDataset.TEAM_IDS,
        SdvDataset.SCHEDULE,
        SdvDataset.ESPN_SCHEDULE,
        SdvDataset.PLAYER_BOX,
        SdvDataset.REFERENCE_RAPM,
    ):
        artifact = client.artifact(dataset, season, refresh=refresh)
        pin = "pinned" if artifact.pinned else "unpinned"
        typer.echo(
            f"{dataset.value:<45} {artifact.byte_count / 1e6:7.1f} MB  sha256 {artifact.sha256[:12]}  {pin}"
        )
    if prior:
        registry = AssumptionRegistry.load(extra_paths=tuple(assumptions or ()))
        cap = season - 1 if last_season is None else min(last_season, season - 1)
        seasons = fetch_prior_inputs(season, cache, registry, last_season=cap)
        typer.echo(f"box-prior training seasons: {', '.join(map(str, seasons)) or 'none'}")
    if economics:
        if refresh:
            EadaClient(cache).file_list(refresh=True)
        frames = load_economics_frames(cache, last_season=season)
        typer.echo(
            f"economics inputs: {frames.outcomes.height} team-seasons, {frames.eada.height} EADA rows"
        )
        for note in frames.notes:
            typer.echo(f"note: {note}")
    typer.echo(f"cache: {cache.root}")


@app.command()
def fit(
    ctx: typer.Context,
    season: SeasonOption,
    cv: bool = typer.Option(
        False, "--cv", help="Choose the penalty by cross-validation grouped on games."
    ),
    top: int = typer.Option(25, help="Players to print."),
    prior: bool | None = typer.Option(
        None,
        "--prior/--no-prior",
        help="Shrink toward the box-score prior. Default from the registry.",
    ),
    output: Path | None = typer.Option(
        None, help="Write the full rating table to this parquet file."
    ),
    last_season: LastSeasonOption = None,
    assumptions: AssumptionsOption = None,
) -> None:
    """Fit RAPM for a season and print the top players."""
    session = _options(ctx).session(assumptions)
    model = session.fit_season(
        season, choose_lambda_by_cv=cv, prior=prior, last_available_season=last_season
    )
    for note in model.notes:
        typer.echo(f"note: {note}")
    rapm = model.rapm
    typer.echo(
        f"season {season}: {rapm.n_possessions:,} possessions, {rapm.n_rows:,} lineup rows, "
        f"lambda {rapm.lam:g}, prior {rapm.prior}, intercept {rapm.intercept:.1f}, home court {rapm.home_court:+.2f}/100 per side"
    )
    if model.prior is not None:
        seasons = ", ".join(str(s) for s in model.prior.train_seasons)
        typer.echo(
            f"box prior fitted on {seasons}: in-sample R^2 offense {model.prior.r2_off:.2f}, defense {model.prior.r2_def:.2f}"
        )
    if rapm.cv is not None:
        for lam, err in zip(rapm.cv.lambdas, rapm.cv.mean_error, strict=True):
            typer.echo(f"  cv lambda {lam:>8g}  held-out weighted MSE {err:,.1f}")
    shown = rapm.table.filter(~pl.col("pooled")).head(top)
    for rank, row in enumerate(shown.iter_rows(named=True), start=1):
        typer.echo(
            f"{rank:>3}. {row['name']:<24} {row['team']:<20} net {row['net']:+6.1f} ± {row['se_net']:.1f}"
            f"  (off {row['orapm']:+.1f}, def {row['drapm']:+.1f}, prior {row['prior_net']:+.1f})  {row['off_poss']:>5} poss"
        )
    if output is not None:
        rapm.table.write_parquet(output)
        typer.echo(f"wrote {output}")


@app.command()
def validate(
    ctx: typer.Context,
    season: SeasonOption,
    torvik: bool = typer.Option(
        True, help="Compare team ratings with Bart Torvik's published CSV."
    ),
    extended: bool = typer.Option(
        False,
        "--extended",
        help="Also check wins, returning players, revenue and bids (fits the previous season).",
    ),
    economics: EconomicsOption = True,
    last_season: LastSeasonOption = None,
    assumptions: AssumptionsOption = None,
) -> None:
    """Check a season fit against published references."""
    session = _options(ctx).session(assumptions)
    model = session.fit_season(season, last_available_season=last_season)
    report = session.validate(model, use_torvik=torvik, extended=extended, economics=economics)
    for gate in report.gates:
        band = f"[{'' if gate.low is None else f'{gate.low:g}'}, {'' if gate.high is None else f'{gate.high:g}'}]"
        typer.echo(
            f"{'PASS' if gate.passed else 'FAIL'}  {gate.name:<30} {gate.value:10.4f}  {band:<18} {gate.detail}"
        )
    for note in report.notes:
        typer.echo(f"note  {note}")
    raise typer.Exit(code=0 if report.passed else 1)


@app.command()
def value(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Player name as listed in box scores.")],
    season: SeasonOption,
    team: str | None = typer.Option(
        None, "--team", "-t", help="Disambiguate players with the same name."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the full valuation as JSON."),
    labels: Path | None = typer.Option(
        None, help="CSV of disclosed deals (deal-registry schema) to train the market model."
    ),
    seed: int = typer.Option(0, help="Random seed for the Monte Carlo draws."),
    economics: EconomicsOption = True,
    last_season: LastSeasonOption = None,
    assumptions: AssumptionsOption = None,
) -> None:
    """Value one player-season."""
    session = _options(ctx).session(assumptions)
    valuation = session.value_player(
        name,
        season,
        team=team,
        labels=labels,
        economics=economics,
        seed=seed,
        last_available_season=last_season,
    )
    if as_json:
        typer.echo(json.dumps(valuation.model_dump(mode="json"), indent=2))
    else:
        typer.echo(valuation.summary())


@app.command("team")
def team_command(
    ctx: typer.Context,
    team: Annotated[str, typer.Argument(help="Team name, e.g. 'Michigan St.'")],
    season: SeasonOption,
    seed: int = typer.Option(0, help="Random seed for the Monte Carlo draws."),
    economics: EconomicsOption = True,
    assumptions: AssumptionsOption = None,
) -> None:
    """Value vs. price for every player on a team (medians)."""
    session = _options(ctx).session(assumptions)
    table = session.value_team(team, season, economics=economics, seed=seed)
    cuts = table.filter(pl.col("value_cut").is_not_null()) if "value_cut" in table.columns else None
    if cuts is not None and not cuts.is_empty():
        typer.echo(
            "Position among paid teammates uses this team's medians: "
            f"value {money(float(cuts['value_cut'][0]))}, price {money(float(cuts['price_cut'][0]))}. "
            "Value is annual program revenue under EADA accounting; price is an allocation "
            "of a published conference-tier budget, not anyone's contract."
        )
    if not economics:
        typer.echo("Program value is off (--no-economics); value and surplus are not computed.")
    typer.echo(
        f"{'player':<24}{'role':<10}{'poss':>6}{'net':>7}{'WAR':>6}{'value':>10}{'price':>10}{'surplus':>10}  position"
    )
    for row in table.iter_rows(named=True):
        typer.echo(
            f"{row['name']:<24}{row['role'] or '':<10}{row['possession_share']:>6.0%}{row['net']:>+7.1f}{row['war']:>6.1f}"
            f"{money(row['program_value']):>10}{money(row['price']):>10}{money(row['surplus']):>10}  {row['quadrant'] or ''}"
        )


@app.command("market-fit")
def market_fit(
    ctx: typer.Context,
    labels: Path | None = typer.Option(None, help="CSV of disclosed deals (deal-registry schema)."),
    assumptions: AssumptionsOption = None,
) -> None:
    """Fit the roster-market model on disclosed deals and report whether it beats the allocation."""
    session = _options(ctx).session(assumptions)
    try:
        fit = session.fit_market(labels=labels)
    except InsufficientLabelsError as error:
        typer.echo(f"not fitted: {error}")
        raise typer.Exit(code=1) from error
    model = fit.model
    usable, note = fit.usable(session.registry)
    typer.echo(
        f"{model.n_labels} labeled player-seasons from {model.n_schools} schools; ridge penalty {model.lam:g}"
    )
    for lam, mae in model.cv_by_lambda:
        typer.echo(f"  penalty {lam:>6g}  leave-one-school-out log MAE {mae:.3f}")
    typer.echo(
        f"  nested (penalty chosen without the held-out school) log MAE {model.nested_log_mae:.3f}; "
        f"tier median {model.baseline_log_mae:.3f}"
    )
    typer.echo(f"{'USABLE' if usable else 'NOT USED'}: {note}")
    for feature, coef in zip(model.features, model.coef[1:], strict=True):
        typer.echo(f"  {feature:<18} {coef:+.3f} log dollars per SD")
    for line in fit.unmatched:
        typer.echo(f"unmatched: {line}")


@app.command("assumptions")
def list_assumptions(assumptions: AssumptionsOption = None) -> None:
    """List every model assumption, its value, basis and status."""
    for item in AssumptionRegistry.load(extra_paths=tuple(assumptions or ())):
        typer.echo(
            f"{item.status.glyph} {item.assumption_id:<45} {item.basis.value:<22} {item.value!r}"
        )


@app.command("cache-path")
def cache_path(ctx: typer.Context) -> None:
    """Print the download cache directory."""
    typer.echo(str(_options(ctx).cache().root))


@cache_app.command("info")
def cache_info(ctx: typer.Context) -> None:
    """Files, sizes and how many raw files match their pinned digests."""
    summary = _options(ctx).cache().summary()
    typer.echo(f"cache: {summary.root}")
    typer.echo(
        f"raw files      {summary.raw_files:>5}  {summary.raw_bytes / 1e6:9.1f} MB  "
        f"({summary.pinned} match a pin, {summary.unpinned} unpinned)"
    )
    typer.echo(f"derived files  {summary.derived_files:>5}  {summary.derived_bytes / 1e6:9.1f} MB")


@cache_app.command("verify")
def cache_verify(ctx: typer.Context) -> None:
    """Hash every cached file and report any that no longer match."""
    issues = _options(ctx).cache().verify()
    for issue in issues:
        typer.echo(f"{issue.relative_path}: {issue.problem}")
    if issues:
        typer.echo(
            f"{len(issues)} problems. Stale derived files are rebuilt on next use; "
            "`athletevalue cache clear --derived-only` removes them now."
        )
        raise typer.Exit(code=1)
    typer.echo("all cached files match the manifest and their pins")


@cache_app.command("clear")
def cache_clear(
    ctx: typer.Context,
    derived_only: bool = typer.Option(
        False, "--derived-only", help="Keep downloads; remove only locally computed files."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
) -> None:
    """Delete cached files."""
    cache = _options(ctx).cache()
    what = "derived files" if derived_only else "all downloads and derived files"
    if not yes and not typer.confirm(f"Delete {what} under {cache.root}?"):
        raise typer.Exit(code=1)
    removed = cache.clear(derived_only=derived_only)
    typer.echo(f"removed {removed} files")


_EXPECTED_ERRORS = (OfflineError, ArtifactMismatchError, SourceUnavailableError)


def run() -> None:
    """Console entry point: expected data errors print one line instead of a traceback."""
    try:
        app()
    except _EXPECTED_ERRORS as error:
        hint = ""
        if isinstance(error, OfflineError):
            hint = " Run `athletevalue fetch --season <year>` while online first."
        typer.echo(f"error: {error}{hint}", err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    run()
