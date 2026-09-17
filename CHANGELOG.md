# Changelog

## Unreleased (0.4.0)

Reproducibility
- Upstream files behind the documented numbers are pinned by SHA-256 in
  `athletevalue/data/pinned_artifacts.json`. A download or cached copy that differs
  from its pin raises `ArtifactMismatchError`; `allow_unpinned=True`
  (`--allow-unpinned`) accepts it. `scripts/pin_artifacts.py` regenerates the file.
- Derived files (EADA extracts, no-prior baselines, economics outcomes) are keyed on
  the model version and settings, and record the digest of every input. A changed
  input, release or setting makes them stale instead of silently reused.
- Downloads retry HTTP 429, 5xx and connection failures with backoff, honouring
  `Retry-After`. The EADA file list is refreshed after 30 days; `fetch --refresh`
  downloads a season again.
- `ATHLETEVALUE_DETERMINISTIC=1` runs BLAS on one thread, so repeated runs agree bit
  for bit on one machine.
- Seasons skipped because a source has not published them are reported as notes on
  the economics model.

Fixes
- The revenue bootstrap and the market model numbered schools with polars categorical
  codes. In recent polars those codes are shared across the process, so a second fit
  in one session could see non-contiguous codes and resample empty schools. Codes are
  now computed locally in order of first appearance, which gives the same numbering
  the published fits used.
- Cross-validation subtracts the held-out Gram in place and reuses one factor buffer,
  cutting peak memory; results are bit-identical.

Publishing
- Releases publish from GitHub Actions through PyPI Trusted Publishing. The package
  metadata no longer carries an email address.

## 0.3.1 (2026-09-17)

Fixes and corrections from an adversarial review. 0.2.0 and 0.3.0 are yanked on PyPI.

Model
- The box-score prior trains only on seasons before the target season. Every season
  but the latest previously shrank toward a prior that had seen later seasons.
- `fit --cv` no longer leaks held-out games into the prior. The penalty is chosen with
  the box totals and team adjustment rebuilt from each fold's training games; the
  validation gate uses the same procedure, which the earlier gate did only for the
  adjustment. The penalty stays at 3000.
- WAR, program value and the allocated price read one shared draw of each player's
  rating, so the surplus interval no longer samples the same rating twice.
- Program value is annual: this season's win and bid effects plus tournament units.
  The two-season figure with next season's carry-over is reported separately and is
  not set against one season of pay.
- The replacement level is a scenario input with three definitions (NBA convention,
  pooled coefficient, bench median); the default is the bench median (about −1.2) and
  every summary prints wins under all three.
- The game-margin spread is fitted on D1-vs-D1 games; non-D1 opponents are rated no
  worse than the worst D1 team.
- Fitted-model price point, interval and surplus draws come from one CV+ set, with
  ranks that guarantee the stated level.

Claims and citations
- The Pythagorean-exponent citation pointed at a page that did not contain the
  figures; it now quotes Pomeroy's ratings page (archived copy linked).
- Two paraphrased citations are verbatim quotes; market sources credit On3 and name
  the Opendorse study; the units source credits Deseret News; the House cap is a
  registry entry.
- The box-prior evidence is restated against a zero-box control: most of the gain is
  the team adjustment. The returning-player check ships as a script.
- "Lower bound" language is replaced with the bias discussion; "cites every constant"
  with "states the basis for every constant".

Public posture
- The README example is a synthetic player; real names appear only in tests.
- The team table's labels are neutral positions against team medians, with the
  medians printed, and unpaid players get none. Every summary ends with a disclaimer.
- The deal registry is closed until the correction and removal process in the new
  PRIVACY.md is staffed; the unnamed-source tier is gone; rows carry a status.
- LICENSE-DATA and the README describe the real chain of title for SportsDataverse
  data; ESPN-derived and NCAA-derived assets are tagged as such in the cache manifest.
- DCO text vendored; `--last-season` on `fit`, `validate` and `value`; the citation
  suite fetches every cited page in the network run.

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
