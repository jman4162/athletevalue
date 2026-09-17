# athletevalue

[![PyPI](https://img.shields.io/pypi/v/athletevalue.svg)](https://pypi.org/project/athletevalue/)
[![Python](https://img.shields.io/pypi/pyversions/athletevalue.svg)](https://pypi.org/project/athletevalue/)
[![CI](https://github.com/jman4162/athletevalue/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/athletevalue/actions/workflows/ci.yml)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](https://github.com/jman4162/athletevalue/blob/main/LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey.svg)](https://github.com/jman4162/athletevalue/blob/main/LICENSE-DATA)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-informational.svg)](https://mypy-lang.org/)

**Price ≠ value.** A Python package and CLI that rates NCAA men's basketball players
from every lineup they played, converts their impact into wins and program revenue,
and compares that value with what the roster market pays, with intervals and cited
assumptions on every number.

`#CollegeBasketball` `#MarchMadness` `#NIL` `#RevenueSharing` `#SportsAnalytics`
`#SportsEconomics` `#RAPM` `#PlusMinus` `#Python` `#OpenSource` `#OpenData`

Rating sites publish one "NIL value" per athlete from a model nobody outside can
inspect. That single number mixes on-court impact, school revenue-sharing budgets,
collective payments and endorsement income. `athletevalue` keeps those apart,
attaches an interval and an evidence status to each, and cites every constant it
uses. The comparison between value and price is the point:

> In 2025-26 the model prices Michigan's Yaxel Lendeborg at $1.55M and credits his
> wins with $912k of program revenue. It prices Gonzaga's Graham Ike at $394k and
> credits him with $569k. Both medians carry wide intervals.

Version 0.1 covers NCAA Division I men's basketball.

## Install

```bash
pip install athletevalue          # or: uv add athletevalue
```

Python 3.12 or later. The first valuation downloads about 400 MB of public data
into a local cache and parses 15 years of EADA spreadsheets, which takes several
minutes. Later runs read the cache and finish in seconds. `athletevalue cache-path`
prints the cache location.

## Quick start

```bash
athletevalue value "Yaxel Lendeborg" --season 2026
```

```text
Yaxel Lendeborg · Michigan (Big Ten) · 2025-26 · starter

Athletic impact           +14.6 /100   80%: +11.3 to +17.9          ▲ estimated
  offense / defense      +8.2 / +6.4
Wins above replacement           6.8   80%: 5.3 to 8.6              ▲ estimated
Program value                  $912k   80%: $494k to $1.40M         ○ scenario
  win revenue                  $889k   80%: $479k to $1.37M         ▲ estimated
  bid revenue                   $12k   80%: $2k to $52k             ▲ estimated
  tournament units               $3k   80%: $814 to $9k             ○ scenario
Roster market value           $1.55M   80%: $1.19M to $2.01M        ○ scenario
Surplus                       -$656k   80%: -$1.25M to -$42k        ○ scenario
Value vs. price            fair star

Drivers
+ on-court impact ranks 1 of 4030 rated players
~ box-score prior +10.8 per 100; his possessions moved the rating +3.8
+ on the floor for 78% of team possessions
~ team went 37-3, net rating +42.8
+ program reports $21.02M basketball revenue (D1 median $3.04M)
+ Big Ten roster budget tier: power

Warnings
! fitted market model not used: 0 disclosed deals available; at least 40 labeled player-seasons needed

Price basis: allocation. Status: ● reported ◆ derived ▲ estimated ○ scenario
As of 2026-09-16 · games through 2026-04-06 · model mbb-v0.3.0
```

```python
from athletevalue import api

valuation = api.value_player("Yaxel Lendeborg", season=2026)
print(valuation.summary())
valuation.war.value, valuation.war.lower, valuation.war.upper

table = api.value_team("Gonzaga", season=2026)  # polars DataFrame, one row per player
```

Other commands:

| Command | What it does |
| --- | --- |
| `athletevalue fit --season 2026 [--cv]` | Fit RAPM and print the top players |
| `athletevalue validate --season 2026` | Check the fit against published references |
| `athletevalue team "Duke" --season 2026` | Value vs. price for a roster |
| `athletevalue market-fit [--labels deals.csv]` | Fit the market model on disclosed deals and report whether it beats the allocation |
| `athletevalue assumptions` | List every model constant, its basis and status |
| `athletevalue fetch --season 2026` | Download inputs without fitting |

Seasons are keyed by ending year: 2026 is the 2025-26 season.

## What each number means

| Output | Question | Method |
| --- | --- | --- |
| Athletic impact | Points per 100 possessions a player adds over an average D1 player | Possession-weighted ridge regression (RAPM) on every lineup, shrunk toward a box-score prior |
| Wins above replacement | Wins the team gains versus a replacement-level player | Per-game win model over the team's actual schedule |
| Program value | Revenue those wins bring the school, this season and next | School fixed-effects model of EADA revenue, plus tournament bids |
| Roster market value | What the market would plausibly pay | Disclosed pay if the registry has it; else a model fitted on disclosed deals once it beats the allocation; else reported 2025-26 roster budgets split by role and impact |
| Surplus | Program value minus price | Difference of the two simulations |

Every estimate carries an 80% interval and a status:

- ● **reported**: stated by a primary source.
- ◆ **derived**: computed from reported data by a fixed procedure.
- ▲ **estimated**: depends on a statistical fit or a published third-party figure.
- ○ **scenario**: depends on an assumption a user should question, such as how a
  conference splits tournament money.

A result takes the status of its weakest input. The allocated market value is a
scenario because no public dataset records individual college basketball pay.

## Market model

Version 0.3 adds a model that learns roster pay from disclosed deals: ridge regression
in log dollars on 14 player and program features, with the penalty chosen by
leave-one-school-out cross-validation and CV+ prediction intervals. It sets the price
only when it has at least 40 labeled player-seasons from 10 schools and its held-out
error beats both a tier median and the allocation. The package ships no fitted model.
The deal registry is empty, so today every valuation reports why the model was not
used and keeps the allocation. To train on deals you are allowed to use:

```bash
athletevalue market-fit --labels my_deals.csv
athletevalue value "Player Name" --season 2026 --labels my_deals.csv
```

The CSV uses the deal-registry columns. Fitted models stay on your machine.

## Validation

Each season fit is checked against the SportsDataverse league-wide RAPM, which is
fit on the same possessions, and against Bart Torvik's team ratings, which are not.

| Gate | 2025 | 2026 | Threshold |
| --- | --- | --- | --- |
| Spearman vs. reference RAPM, players with 500+ possessions (no-prior fit) | 0.963 (2,817) | 0.975 (3,036) | ≥ 0.90 |
| Spearman of team net rating vs. Torvik AdjOE − AdjDE | 0.953 (321 teams) | 0.971 (324 teams) | ≥ 0.93 |
| Held-out error with box prior ÷ without, games held out | 0.997 | 0.997 | ≤ 1.00 |
| Intercept, points per 100 | 103.9 | 106.0 | 95–112 |
| Home-court advantage per side, per 100 | 2.93 | 2.61 | 1–4 |
| Residual variance | 13,315 | 13,330 | 11,000–15,000 |

Two checks outside the gates support the box-score prior:

| Check | No prior | With prior |
| --- | --- | --- |
| 2025 rating vs. the same player's 2026 rating, 1,735 returning players, correlation | 0.40 | 0.52 |
| Team net rating vs. Torvik AdjEM, 2026 slope (1.0 means same scale) | 0.81 | 0.96 |

The 2025 prior in the first check was trained on 2021-2024, so nothing from 2026
reaches it. Run `athletevalue validate --season 2026` to reproduce the gates.

## Data and licensing

| Source | Used for | Terms |
| --- | --- | --- |
| [SportsDataverse releases](https://github.com/sportsdataverse/sportsdataverse-data/releases) | Possessions, schedules, rosters, reference RAPM | MIT-licensed repository |
| [EADA](https://ope.ed.gov/athletics/) | Men's basketball revenue by school, 2011-2025 | U.S. government work |
| [Bart Torvik](https://barttorvik.com/) team results | Validation only | Not bundled, not used to fit |
| Press and methodology pages | Roster budgets, tournament unit value, conventions | Cited facts, see `athletevalue assumptions` |

Code is MIT. Curated data in `src/athletevalue/data` (the team crosswalk and the
deal registry) is CC BY 4.0; see [LICENSE-DATA](https://github.com/jman4162/athletevalue/blob/main/LICENSE-DATA). Data downloaded at
runtime stays under its own terms and is never redistributed.

KenPom and the Knight-Newhouse College Athletics Database restrict redistribution.
The package has a reader for a user's own KenPom export, for local comparison, and
an import-linter contract that stops fitting code from importing it.

## Limitations

- **Revenue is what schools attribute to men's basketball in EADA.** Conference
  media money, donations credited to the athletic department and brand effects are
  not in it, so program value is a lower bound on a player's value to the university.
  61% of D1 rows report revenue equal to expense and are excluded from the fit.
- **Without labels, market value is an allocation, not a prediction.** It spreads
  reported average roster budgets across a roster using reported pay ratios by role,
  and is only available for 2025-26, the season those figures describe. The fitted
  market model needs disclosed deals the public record does not yet contain.
- **College ratings are noisy.** A starter's net rating has a posterior SD near 2.6
  points per 100 with the box-score prior and near 4 without it. The prior's own
  coefficient uncertainty is not propagated, so intervals are slightly too narrow.
- **Players are rated within a season.** Transfers are not linked across seasons.
- **Tournament flags before 2023 come from matching ESPN games by date and team
  name.** A few First Four games each season go unmatched.

[METHODOLOGY.md](methodology.md) gives the equations, estimates and caveats for each layer.

## Roadmap

- **v0.2**: box-score prior for RAPM with a team adjustment.
- **v0.3 (this release)**: fitted roster-market model with CV+ intervals, dormant until
  enough disclosed deals exist.
- **Next**: seed the deal registry from public records, multi-season ratings with
  recency weights, CollegeBasketballData for in-season updates, optional private
  KenPom calibration, then football.

## Contributing

See [CONTRIBUTING.md](contributing.md). Sourced deals for the registry are the most
useful contribution; the rules are in
[src/athletevalue/data/deal_registry/README.md](https://github.com/jman4162/athletevalue/blob/main/src/athletevalue/data/deal_registry/README.md).

## Citation

See [CITATION.cff](https://github.com/jman4162/athletevalue/blob/main/CITATION.cff). Estimates are research outputs; read
[DISCLAIMER.md](https://github.com/jman4162/athletevalue/blob/main/DISCLAIMER.md) before using them for decisions.
