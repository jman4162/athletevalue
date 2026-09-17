# Methodology

This document describes model `mbb-v0.2.0`. Every constant named here is an entry
in the assumption registry (`athletevalue assumptions` lists them with their basis,
citation or rationale). Figures quoted are from the 2025 and 2026 season fits.

## 1. Data

**Possessions.** SportsDataverse publishes one row per possession for every D1
men's game from 2011, parsed from stats.ncaa.org play-by-play. Each row names the
offense team, the points scored, a garbage-time flag and the stats.ncaa.org ids of
all ten players on the floor. For 2026 the file has 878,168 possessions; 840,040
remain after dropping garbage time and rows where a D1 team's lineup is incomplete.

**Non-D1 opponents.** Their lineups carry no player ids. Each non-D1 lineup enters
the model as one shared column, so those games still inform D1 players' ratings.

**Venues.** ESPN schedules mark neutral sites and NCAA tournament games
(`tournament_id` 22). From 2023 the possession file carries ESPN game ids. For
earlier seasons neutral-site and postseason ESPN games are matched to NCAA games
within one day using token similarity of both team names. Matching recovers 93-99%
of neutral-site games and 62-67 of 67 tournament games per season.

**Program finances.** EADA sport-level files give each institution's reported men's
basketball revenue (`REV_MEN`) and expense (`EXP_MEN`) by academic year. A curated
crosswalk maps 369 of 373 stats.ncaa.org team names to IPEDS unit ids. The service
academies are not in EADA, and New Haven joined D1 after the latest file.

## 2. Player impact (RAPM)

Possessions are grouped into rows that share a game, an offense team, both
lineups and a venue. For row *r* with offense lineup *O* and defense lineup *D*:

    y_r = 100 · pts_r / poss_r
        = μ + h·v_r + Σ_{i∈O} off_i − Σ_{j∈D} def_j + ε_r,   Var(ε_r) = σ² / poss_r

where *v_r* is +1 when the offense is at home, −1 away and 0 neutral. A positive
`def_j` means player *j* lowers opponent scoring, so a player's net rating is
`off + def`. Players with fewer than 100 average possessions share one pooled
offense and one pooled defense column.

Coefficients minimize

    Σ_r poss_r (y_r − x_r β)² + λ Σ_k p_k (β_k − b0_k)²

with *p_k* = 1 for player columns and 0 for the intercept, home court, pooled and
non-D1 columns. `b0` is a prior mean; v0.1 uses zero. Weighting rows by possession
count makes `X'WX` identical to the possession-level Gram matrix, so λ means the
same thing as in a possession-level fit.

**Penalty.** λ = 1000, the value in the SportsDataverse model card. Cross-validation
that holds out whole games picks the same value for 2026:

| λ | 100 | 300 | 1000 | 3000 | 10000 |
| --- | --- | --- | --- | --- | --- |
| held-out weighted MSE | 5,246 | 5,207 | 5,187 | 5,192 | 5,210 |

**Uncertainty.** Posterior variances are exact: `σ̂² · diag(A⁻¹)` with
`A = X'WX + λP`, from the Cholesky factor. The residual variance is
`σ̂² = Σ poss_r r_r² / (n_rows − df)` with `df = n_cols − λ Σ_penalized (A⁻¹)_kk`.
Net-rating SEs include the offense-defense covariance. The fit refuses to report a
result if `df` is not below the column count, which catches a penalty that was
silently dropped. Without a box prior a starter's net-rating SE is about 4 points
per 100: with λ = 1000 the prior SD is `sqrt(13,300 / 1000) ≈ 3.6` per component,
and one season of college possessions does not move far from it.

**Team ratings.** A team's adjusted offense is the possession-weighted average of
the offensive coefficients it put on the floor; defense likewise. Their sum is the
team's adjusted net rating relative to an average D1 lineup.

**Checks.** See the validation table in the README. Two findings from them:

- The pooled low-minute players rate −12.1 (2026) and −9.3 (2025) per 100, far
  below the −2.0 replacement convention borrowed from NBA Box Plus/Minus.
- The home-court coefficient is 2.6-3.2 per side, a net home edge of about 6 points
  per 100 possessions, or 4 points in a 70-possession game.

## 2b. Box-score prior

Shrinking every player toward zero treats a high-scoring starter and a walk-on the
same until possessions say otherwise. Since v0.2 the ridge instead shrinks toward
what a player's box score predicts.

**Box model.** For each player-season, 13 counting stats (points, field-goal,
three-point, free-throw, rim and mid-range attempts, assists, turnovers, offensive
and defensive rebounds, steals, blocks, fouls) become rates per 100 offensive
possessions after adding 100 possessions at the league-average rate. A
possession-weighted ridge regression on standardized rates predicts no-prior
offensive and defensive RAPM, separately. It is fitted on the four nearest other
seasons (for 2026: 2022-2025), never on the season it is applied to. In-sample R²
is 0.33 for offense and 0.12 for defense.

**Team adjustment.** Box rates ignore opponents, so a low-major star's rates
overstate him. As in Box Plus/Minus, each team's offensive and defensive
predictions are shifted by a constant so their possession-weighted sum equals the
team's rating from the no-prior fit. Without this step 2026 team ratings match
Torvik at Spearman 0.925; with it, 0.971.

**Fit.** The adjusted predictions are the prior mean `b0` in the ridge objective
above, with λ = 3000. Cross-validation that holds out whole games, and recomputes the
team adjustment from training games only, gives:

| λ with prior | 1000 | 3000 | 10000 | 30000 |
| --- | --- | --- | --- | --- |
| 2026 held-out MSE | 5,177.6 | 5,171.9 | 5,171.9 | 5,172.4 |
| 2025 held-out MSE | 5,048.3 | 5,041.4 | 5,040.6 | 5,040.8 |

against 5,186.6 (2026) and 5,055.5 (2025) with no prior at λ = 1000. An earlier
version computed the team adjustment from the full season inside cross-validation;
held-out games then leaked into the prior and error kept falling as λ grew without
bound. The leak-free version above is what the `prior_cv_error_ratio` gate runs.

**Effects.**

- Correlation between a returning player's 2025 rating and his 2026 no-prior rating
  rises from 0.40 to 0.52 (1,735 players; 2025 prior trained on 2021-2024).
- The no-prior fit compresses team strength: regressing its team ratings on Torvik
  AdjEM gives a slope of 0.81 (2026). With the prior the slope is 0.96 and RMSE
  falls from 3.8 to 3.2 points per 100.
- A starter's posterior SD falls from about 4.0 to about 2.6 points per 100. That
  figure treats the prior mean as known; box-model coefficient uncertainty, small
  with ~13,000 training rows, is not added.

The no-prior fit is still computed and is what the reference-RAPM gate compares,
because the published reference also shrinks toward zero. `athletevalue fit
--no-prior` reproduces it.

## 3. Wins above replacement

A player's share of team possessions is `s = (off_poss + def_poss) / team lineup
possessions`. Replacing him with a replacement-level player lowers the team's net
rating by `Δ = (net − R) · s` with `R = −2.0`.

Win probability in each game is

    P(win) = Φ( (net_team − net_opp + 2h·v) · poss_game / 100 / σ_margin )

`σ_margin` is fitted each season as the root-mean-square gap between actual and
expected margins: 11.5 points in 2026 and 12.1 in 2025 with the box prior. WAR is the sum over the
team's games of `P(win | net_team) − P(win | net_team − Δ)`, computed for 4,000
draws of the player's net rating from its posterior. The headline figure is the
median of those draws.

`war_linear` is the derivative of the same sum times Δ. It agrees with the
simulation for small impacts and understates it for a star on a dominant team,
where removing the player moves close games into toss-up territory.

A Pythagorean exponent fitted on raw season efficiencies is 7.6 (2026) and 7.8
(2025). Published exponents of 10.25-11.5 use schedule-adjusted efficiencies. The
package reports the fitted exponent as a check and does not use it for WAR.

## 4. Program value

**Revenue response.** On school-seasons 2012-2025, excluding 2020 (tournament
cancelled) and 2021 (attendance limits):

    log R_st = a_s + g_t + b0·W_st + b1·W_s,t−1 + d0·Bid_st + d1·Bid_s,t−1 + e_st

School effects absorb brand, market and conference; season effects absorb league
growth. Only rows where reported revenue differs from expense enter the fit: 61% of
D1 rows report the two as equal, which means the school allocates revenue to the
sport rather than measuring it. The fit uses 1,370 school-seasons from 167 schools
with at least four usable seasons. A bootstrap over schools gives 400 replicates.

| Effect (current + next season) | Median | 80% interval |
| --- | --- | --- |
| One win, log revenue | 0.0063 | 0.0037 to 0.0089 |
| NCAA bid, log revenue | 0.053 | 0.015 to 0.091 |

A win is worth `(b0 + b1) · R̄` where `R̄` is the school's mean reported revenue over
its latest three seasons. For Michigan (`R̄` = $21.0M) that is about $130k a win.

**Tournament bids.** A logistic model of bid on win percentage, power-conference
membership and their interaction, fitted on 2011-2026 except 2020 and 2021. A player's WAR changes his
team's win percentage by `WAR / games`, which changes the bid probability.

**Tournament units.** A conference earns one unit per game a member plays, except
the championship game; the data give 1.94 units per bid. A unit is worth about
$2M paid over six years (undiscounted). The school's share defaults to an equal
split among conference members, a user assumption, which makes this component and
total program value a scenario.

**What is missing.** Conference media distributions, donations and licensing that
EADA does not attribute to men's basketball, and effects beyond one season. Program
value is a lower bound on value to the university.

## 5. Roster market value

No public dataset records what individual college basketball players are paid, so
v0.1 allocates a reported budget rather than predicting pay.

**Budget.** Opendorse's 2025-26 study, as reported in March 2026, gives average
men's basketball roster spending of $7-10M in the ACC, Big 12, Big East, Big Ten
and SEC; $2-5M for a "B tier"; and $1-2M for a "C tier". The study does not list
B-tier members; the package assigns the AAC, Atlantic 10, Mountain West, WCC and
MVC as a user assumption. Budgets are lognormal with the cited range as the central
80%. For context, the House settlement caps school revenue sharing at about $20.5M
across all sports in 2025-26; the budgets above also include collective money.

**Roles.** Players are ranked by possessions. The top five are starters, the next
three are rotation, the rest are bench, and anyone on the floor for under 5% of
possessions is unpaid. The same study reports rotation players earn 15-25% less than
starters and bench players 30-55% less; each draw samples a ratio from those ranges.

**Impact within a role.** Pay within a role is proportional to
`max(net − R, 0.5)^κ` with κ = 1, normalized to the role mean so the role ratios
still hold. Every draw samples the budget, ratios and player ratings.

The allocation sums to the budget in every draw. It says what a team spending an
average budget in its tier would pay if it paid by role and on-court impact. It
does not see recruiting rank, draft stock, transfer leverage or social following,
which the study and other research say drive real contracts. That is why it is
labelled a scenario and why the deal registry exists.

## 6. Surplus and quadrant

Surplus is program value minus price, draw by draw. When the deal registry holds a
disclosed figure for the player-season it replaces the allocation as the price.

The value-vs-price quadrant compares a player's median program value and price with
the medians of his paid teammates. It is a within-team comparison, not a league
ranking.

## 7. Reproducing

```bash
uv sync --group dev
uv run pytest                 # offline suite, synthetic seasons
uv run pytest -m network      # full 2025 and 2026 fits against live data
athletevalue validate --season 2026
athletevalue fit --season 2026 --cv
```

Every download is recorded in `manifest.json` in the cache with its URL, SHA-256,
size, retrieval time and license.
