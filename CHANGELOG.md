# Changelog

## Unreleased

Fixes from a code review of the market model:

- `athletevalue value --labels` now passes the labels file through; it was ignored.
- CV+ intervals used half the miscoverage in each tail, so "80%" intervals covered about
  90%. Each bound now uses the full `1 - level`; synthetic coverage is 0.79-0.83.
  Monte Carlo draws sample the CV+ set itself instead of a narrower version of it.
- The must-beat-the-allocation gate was skipped whenever any label lacked an allocated
  price. The model and the allocation are now compared on the labels the allocation
  prices, and the gate says when there is nothing to compare.
- With `economics=False` the model was trained with revenue features but priced
  players as if revenue were missing. Economics are now fitted for market features.
- Deals are grouped by normalized name and school, so spelling variants of one
  player-season become one label.
- Deal CSVs with numeric ids load; ids are read as text.
- A fitted price takes the status of its weakest input, including WAR and the tier
  lists, so it now reports as a scenario. The allocation stays visible whenever it is
  not the price.
- The insufficient-labels message counts only men's basketball roster-pay deals.

## 0.3.0 (2026-09-17)

- Fitted roster-market model: ridge regression of log annual pay on 14 player and
  program features, penalty chosen by leave-one-school-out cross-validation, CV+
  prediction intervals with schools as folds.
- Use gate: at least 40 labeled player-seasons from 10 schools, and held-out error
  below both a tier median and the allocation. Disclosed pay still outranks the model.
- `athletevalue market-fit [--labels deals.csv]` and `value --labels`; valuations keep
  the allocation as `allocated_market_value` when the model sets the price and warn
  when the model is not used.
- No fitted model ships: the deal registry is empty.
- README badges, description and hashtags. Model version `mbb-v0.3.0`.

## 0.2.0 (2026-09-17)

- Box-score prior for RAPM: a ridge model of per-100 box rates fitted on the four
  nearest other seasons, with a Box Plus/Minus style team adjustment, used as the
  shrinkage target (λ = 3000). Held-out error falls 0.3% in 2025 and 2026, next-season
  correlation for returning players rises from 0.40 to 0.52, and team ratings match
  Torvik's scale (slope 0.96 vs 0.81).
- New validation gate `prior_cv_error_ratio`, computed with the team adjustment refit
  on training games only so held-out games cannot leak into the prior.
- `athletevalue fit --prior/--no-prior`; rating tables gain a `prior_net` column and
  valuations a driver showing how far possessions moved a player from his prior.
- The reference-RAPM gate compares the no-prior fit, which matches the reference's
  estimand.
- Model version `mbb-v0.2.0`.

## 0.1.0 (2026-09-16)

First release, men's basketball.

- RAPM from SportsDataverse possession data with exact posterior standard errors,
  pooled low-minute players, non-D1 opponents, neutral sites and grouped
  cross-validation.
- Validation gates against the SportsDataverse league-wide RAPM and Bart Torvik
  team ratings.
- Wins above replacement from a per-game win model over each team's schedule.
- Program value from a school fixed-effects model of EADA revenue, a tournament-bid
  model and conference tournament units.
- Roster market value for 2025-26 from reported roster budgets allocated by role
  and impact.
- Assumption registry with citations and evidence status on every estimate.
- CLI: `fetch`, `fit`, `validate`, `value`, `team`, `assumptions`, `cache-path`.
