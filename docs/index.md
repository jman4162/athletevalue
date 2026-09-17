# athletevalue

[![PyPI](https://img.shields.io/pypi/v/athletevalue.svg)](https://pypi.org/project/athletevalue/)
[![Python](https://img.shields.io/pypi/pyversions/athletevalue.svg)](https://pypi.org/project/athletevalue/)
[![CI](https://github.com/jman4162/athletevalue/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/athletevalue/actions/workflows/ci.yml)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](https://github.com/jman4162/athletevalue/blob/main/LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey.svg)](https://github.com/jman4162/athletevalue/blob/main/LICENSE-DATA)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-informational.svg)](https://mypy-lang.org/)

What an NCAA men's basketball player's wins are worth to his school, and what a
published roster budget would pay him. A Python package and CLI that rates players
from every lineup they played, converts impact into wins and into the revenue schools
attribute to basketball in federal filings, and sets that against an allocation of
reported roster budgets. Every number carries an interval and a stated basis.

**What the price side is and is not.** No public dataset records what individual
college basketball players are paid. The "roster market value" this package prints is
an allocation: a published conference-tier average budget, spread across a roster by
role and rating. It contains no information about any player's contract and is
labelled a scenario for that reason. A model that learns pay from disclosed deals is
included but ships dormant, because there are no public deals to train it on.

**Estimates, not reports.** Outputs are research estimates about public game data
and public filings. They are not reports of anyone's pay and should not be read or
repeated as such. See [DISCLAIMER](https://github.com/jman4162/athletevalue/blob/main/DISCLAIMER.md)
and [PRIVACY](privacy.md).

## Install

```bash
pip install athletevalue          # or: uv add athletevalue
```

Python 3.12 or later. The first valuation downloads about 500 MB of public data for
16 seasons and parses 15 years of EADA spreadsheets; allow several minutes and about
5 GB of memory for a season fit (7 GB with `--cv`). Later runs read the cache.
`athletevalue cache-path` prints the cache location.

## Quick start

```bash
athletevalue value "Player Name" --season 2026 --team "School"
```

The example below is a synthetic player on a synthetic season, produced by
`scripts/readme_example.py`; the README does not name real athletes.

```text
A. Guard · State U (Example Conf) · 2025-26 · starter

Athletic impact            +4.6 /100   80%: +0.9 to +8.3            ▲ estimated
  offense / defense      +2.6 / +2.0
Wins above replacement           0.6   80%: 0.2 to 1.1              ○ scenario
  replacement         bench median (-0.3/100); under nba convention 0.9 / pooled 0.6
Program value                   $61k   80%: $12k to $158k           ○ scenario
  win revenue, this season          $31k   80%: $8k to $55k             ○ scenario
  bid revenue, this season           $8k   80%: $1k to $28k             ○ scenario
  tournament units              $22k   80%: $3k to $75k             ○ scenario
  with next season              $82k   80%: $17k to $204k           ○ scenario
Roster market value           $1.45M   80%: $476k to $2.55M         ○ scenario
Surplus                      -$1.37M   80%: -$2.41M to -$458k       ○ scenario
Among paid teammates  both above median   (relative to team medians)

Drivers
+ on-court impact ranks 1 of 72 rated players
~ box-score prior +3.4 per 100; his possessions moved the rating +1.1
+ on the floor for 58% of team possessions
~ team went 10-4, net rating +12.1
+ program reports $12.00M basketball revenue (D1 median $7.00M)
+ Example Conf roster budget tier: power (published average, collectives included); the House cap for direct revenue sharing across all sports is $20.50M

Price basis: allocation. Status: ● reported ◆ derived ▲ estimated ○ scenario
As of 2026-09-17 · games through 2025-12-29 · model mbb-v0.3.1
Estimates, not reports of pay. An allocated market value spreads a published conference-tier budget by role and rating; it says nothing about this player's contract.
```

```python
from athletevalue import api

valuation = api.value_player("Player Name", season=2026, team="School")
print(valuation.summary())
valuation.war.value, valuation.war.lower, valuation.war.upper

table = api.value_team("School", season=2026)  # polars DataFrame, one row per player
```

Other commands:

| Command | What it does |
| --- | --- |
| `athletevalue fit --season 2026 [--cv] [--no-prior]` | Fit ratings and print the top players |
| `athletevalue validate --season 2026` | Check the fit against published references |
| `athletevalue team "School" --season 2026` | Value and allocated price for a roster |
| `athletevalue market-fit --labels deals.csv` | Fit the market model on deals you supply |
| `athletevalue assumptions` | List every model constant, its value, basis and status |
| `athletevalue fetch --season 2026` | Download inputs without fitting |

Seasons are keyed by ending year: 2026 is the 2025-26 season.

## What each number means

| Output | Question | Method |
| --- | --- | --- |
| Athletic impact | Points per 100 possessions a player adds over an average D1 player | Possession-weighted ridge regression (RAPM) on every lineup, shrunk toward a box-score prior fitted on earlier seasons |
| Wins above replacement | Wins the team gains versus a replacement-level player | Per-game win model over the team's actual schedule; three replacement definitions, one chosen |
| Program value | Revenue this season that the fitted model associates with those wins | School fixed-effects model of EADA revenue, plus a tournament-bid model and conference units |
| Roster market value | What a published tier budget would pay under a stated split | Allocation by role and rating (scenario); disclosed pay when the registry has it |
| Surplus | Program value minus price, annual on both sides | Difference of the two, on shared rating draws |

Every estimate carries an 80% interval and a status:

- ● **reported**: stated by a primary source.
- ◆ **derived**: computed from reported data by a fixed procedure.
- ▲ **estimated**: depends on a statistical fit or a published third-party figure.
- ○ **scenario**: depends on an assumption a user should question, such as which
  replacement level to use or how a conference splits tournament money.

A result takes the status of its weakest input. Because the replacement level is a
choice, wins and every dollar figure are scenarios; the summary prints wins under all
three definitions.

## Validation

Each season fit is checked against the SportsDataverse league-wide RAPM, which is fit
on the same possessions and shrinks toward zero (a reimplementation check), and
against Bart Torvik's team ratings, which are not (an external check).

| Gate | 2025 | 2026 | Threshold |
| --- | --- | --- | --- |
| Spearman vs. reference RAPM, players with 500+ possessions, no-prior fit | 0.963 (2,817) | 0.975 (3,036) | ≥ 0.90 |
| Spearman of team net rating vs. Torvik AdjOE − AdjDE | 0.953 (321 teams) | 0.971 (324 teams) | ≥ 0.93 |
| Held-out error with box prior ÷ without, games held out, prior rebuilt per fold | 0.998 | 0.998 | ≤ 1.00 |
| Intercept, points per 100 | 103.9 | 106.0 | 95–112 |
| Home-court advantage per side, per 100 | 2.93 | 2.61 | 1–4 |
| Residual variance | 13,315 | 13,330 | 11,000–15,000 |

`athletevalue validate --season 2026` reproduces this table. Torvik's site refuses
some cloud address ranges; when that happens the Torvik rows are skipped and reported.

**What the box-score prior buys.** Held-out squared error per 100 possessions, games
held out, box totals and the team adjustment rebuilt from training games:

| Prior | 2025 | 2026 |
| --- | --- | --- |
| None (λ = 1000) | 5,055.5 | 5,186.6 |
| Zero box information plus the team adjustment (λ = 3000) | 5,050.6 | 5,182.1 |
| Box model plus the team adjustment (λ = 3000) | 5,046.8 | 5,177.6 |

Most of the gain is the team adjustment restoring the scale of team strength; the box
model itself adds about 0.1%. On 1,735 players who appear in both 2025 and 2026, the
2025 rating with the prior (trained on 2021–2024) correlates 0.52 with the 2026
no-prior rating, against 0.40 without it. `scripts/returning_players.py 2025 2026`
reproduces that check.

## Data and licensing

| Source | Used for | Terms |
| --- | --- | --- |
| [SportsDataverse releases](https://github.com/sportsdataverse/sportsdataverse-data/releases) | Possessions, box scores, schedules, reference RAPM | Published from an MIT-licensed repository. The NCAA files are parsed from stats.ncaa.org by a producer repository with no license file, and the ESPN files from ESPN's public API; neither origin has granted rights and stats.ncaa.org's terms restrict automated access. MIT covers SportsDataverse's work, not those rights. |
| [EADA](https://ope.ed.gov/athletics/) | Men's basketball revenue by school, 2011–2025 | U.S. government work |
| [Bart Torvik](https://barttorvik.com/) team results | Validation only | Season CSVs permitted by the site's robots.txt; not bundled, not used to fit |
| Press and methodology pages | Roster budgets, tournament unit value, conventions | Short quoted facts with URLs; see `athletevalue assumptions` |

Code is MIT. Curated data in `src/athletevalue/data` (the team crosswalk and the deal
registry) is CC BY 4.0; see [LICENSE-DATA](https://github.com/jman4162/athletevalue/blob/main/LICENSE-DATA).
Data downloaded at runtime stays under its own terms and is never redistributed. A
commercial user should do their own diligence on the SportsDataverse chain.

KenPom and the Knight-Newhouse College Athletics Database restrict redistribution. The
package has a reader for a user's own KenPom export, for local comparison, and an
import-linter contract that stops fitting code from importing it.

## Limitations

- **Program value is an association, not a causal estimate.** The win coefficient is
  identified from a school winning more or less than its own norm, and schools that
  spend more win more and earn more in the same years, so the coefficient can be
  biased upward. It counts only revenue schools attribute to men's basketball in EADA,
  which omits conference media money, donations and brand effects, a bias downward.
  The net direction is unknown. Sixty-one percent of school-seasons report revenue
  equal to expense and are excluded from the fit but kept in each school's revenue
  base, so for most programs the dollar figure is an elasticity from revenue-tracking
  schools applied to a budget number.
- **The replacement level is a choice.** The NBA convention is −2.0 per 100. In these
  data the pooled low-minute players rate about −12, and the 9th–12th men on each
  team about −1.2. The default uses the bench median; the summary shows all three.
- **Market value is an allocation for 2025-26 only.** It spreads reported average
  roster budgets by role and rating and sums to the budget in every draw. Every
  conference outside the report's named top and middle tiers gets the bottom tier,
  including some programs whose real budgets are far higher.
- **Ratings are noisy and their intervals condition on the prior.** A starter's net
  rating has a posterior SD near 2.6 points per 100 with the prior and near 4 without.
  The prior mean is treated as known.
- **Players are rated within a season.** Cross-season linkage exists only in the
  returning-player script.
- **Results reproduce to about 1e-15 relative, not bitwise**, because the dense
  Cholesky solve is multithreaded; ranks on near-ties can differ between runs.

[METHODOLOGY.md](methodology.md)
gives the equations, fitted values and caveats for each layer.

## Roadmap

- **0.4.0**: pinned upstream hashes and version-keyed caches, cross-season identity in
  the package, validation gates on every layer with a committed snapshot, a session
  cache so one `value` call does not refit everything, CLI tests, a release workflow.
- **Later**: multi-season ratings with recency weights, in-season updates from
  CollegeBasketballData, then football.

## Contributing

See [CONTRIBUTING](contributing.md).
The deal registry is not accepting rows until its correction and takedown process is
in place; see [PRIVACY](privacy.md).

## Citation

If you use athletevalue in a publication, cite the software:

```bibtex
@software{hodge_athletevalue_2026,
  author  = {Hodge, John},
  title   = {athletevalue: open models of college basketball player impact,
             program value and roster-market pay},
  year    = {2026},
  version = {0.3.1},
  url     = {https://github.com/jman4162/athletevalue},
  note    = {Python package, MIT license; data files CC BY 4.0}
}
```

[CITATION.cff](https://github.com/jman4162/athletevalue/blob/main/CITATION.cff) carries
the same metadata in a machine-readable form. Estimates are research outputs; read
[DISCLAIMER](https://github.com/jman4162/athletevalue/blob/main/DISCLAIMER.md) before
using them for decisions.
