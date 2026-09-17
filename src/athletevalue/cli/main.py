"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from athletevalue import api
from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.valuation.report import money

app = typer.Typer(
    help="Separate what a college basketball player is worth to a program from what the market pays.",
    no_args_is_help=True,
)

SeasonOption = Annotated[
    int, typer.Option("--season", "-s", help="Ending year, e.g. 2026 for 2025-26.")
]
AssumptionsOption = Annotated[
    list[Path] | None,
    typer.Option("--assumptions", "-a", help="TOML file overriding registry entries. Repeatable."),
]


def _registry(paths: list[Path] | None) -> AssumptionRegistry:
    return AssumptionRegistry.load(extra_paths=tuple(paths or ()))


@app.command()
def fetch(
    season: SeasonOption,
    economics: bool = typer.Option(True, help="Also fetch EADA and past seasons."),
) -> None:
    """Download and cache one season's inputs."""
    cache = ArtifactCache.default()
    client = SdvClient(cache)
    for dataset in (
        SdvDataset.POSSESSIONS,
        SdvDataset.TEAM_IDS,
        SdvDataset.SCHEDULE,
        SdvDataset.ESPN_SCHEDULE,
        SdvDataset.REFERENCE_RAPM,
    ):
        artifact = client.artifact(dataset, season)
        typer.echo(
            f"{dataset.value:<45} {artifact.byte_count / 1e6:7.1f} MB  sha256 {artifact.sha256[:12]}"
        )
    if economics:
        frames = api.load_economics_frames(cache, last_season=season)
        typer.echo(
            f"economics inputs: {frames.outcomes.height} team-seasons, {frames.eada.height} EADA rows"
        )
    typer.echo(f"cache: {cache.root}")


@app.command()
def fit(
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
    assumptions: AssumptionsOption = None,
) -> None:
    """Fit RAPM for a season and print the top players."""
    model = api.fit_season(
        season, registry=_registry(assumptions), choose_lambda_by_cv=cv, prior=prior
    )
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
    season: SeasonOption,
    torvik: bool = typer.Option(
        True, help="Compare team ratings with Bart Torvik's published CSV."
    ),
    assumptions: AssumptionsOption = None,
) -> None:
    """Check a season fit against published references."""
    registry = _registry(assumptions)
    report = api.validate(
        api.fit_season(season, registry=registry), registry=registry, use_torvik=torvik
    )
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
    name: Annotated[str, typer.Argument(help="Player name as listed in box scores.")],
    season: SeasonOption,
    team: str | None = typer.Option(
        None, "--team", "-t", help="Disambiguate players with the same name."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the full valuation as JSON."),
    seed: int = typer.Option(0, help="Random seed for the Monte Carlo draws."),
    assumptions: AssumptionsOption = None,
) -> None:
    """Value one player-season."""
    valuation = api.value_player(
        name, season, team=team, registry=_registry(assumptions), seed=seed
    )
    if as_json:
        typer.echo(json.dumps(valuation.model_dump(mode="json"), indent=2))
    else:
        typer.echo(valuation.summary())


@app.command("team")
def team_command(
    team: Annotated[str, typer.Argument(help="Team name, e.g. 'Michigan St.'")],
    season: SeasonOption,
    seed: int = typer.Option(0, help="Random seed for the Monte Carlo draws."),
    assumptions: AssumptionsOption = None,
) -> None:
    """Value vs. price for every player on a team (medians)."""
    table = api.value_team(team, season, registry=_registry(assumptions), seed=seed)
    typer.echo(
        f"{'player':<24}{'role':<10}{'poss':>6}{'net':>7}{'WAR':>6}{'value':>10}{'price':>10}{'surplus':>10}  quadrant (relative to teammates)"
    )
    for row in table.iter_rows(named=True):
        typer.echo(
            f"{row['name']:<24}{row['role'] or '':<10}{row['possession_share']:>6.0%}{row['net']:>+7.1f}{row['war']:>6.1f}"
            f"{money(row['program_value']):>10}{money(row['price']):>10}{money(row['surplus']):>10}  {row['quadrant'] or ''}"
        )


@app.command("assumptions")
def list_assumptions(assumptions: AssumptionsOption = None) -> None:
    """List every model assumption, its value, basis and status."""
    for item in _registry(assumptions):
        typer.echo(
            f"{item.status.glyph} {item.assumption_id:<45} {item.basis.value:<22} {item.value!r}"
        )


@app.command("cache-path")
def cache_path() -> None:
    """Print the download cache directory."""
    typer.echo(str(ArtifactCache.default().root))


if __name__ == "__main__":
    app()
