# Round-number effects in RMB gold prices — an exploratory study

A personal empirical-finance project testing whether RMB-denominated Au99.99 gold
prices behave differently near round 50-yuan levels. Using daily data, I found no
detectable occupancy or next-day volatility effect under an autocorrelation-aware
block-bootstrap null. Because daily prices are too coarse to identify the mechanism
well, the project is paused rather than extended with additional specifications.

## Research question

Do daily gold prices cluster at, avoid, or react to round 50-yuan levels?

## Main findings

- A naive independent-observation test suggested avoidance near round levels
  (`p ≈ 0.005`), but daily price persistence makes that inference unreliable.
- A moving-block bootstrap of daily log returns (20-day blocks, 2,000 simulations,
  fixed seed) placed the observed occupancy statistic inside the null distribution
  (`p ≈ 0.45`).
- Proximity to a level did not distinguishably predict next-day absolute returns
  under the same null framework (`p ≈ 0.32`).
- These are null results under the specified design, not proof that round-number
  effects never exist.

The fuller interpretation is in [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

## Data and public-repository policy

The study used daily Au99.99 OHLC observations originally collected from a
Sina-hosted source. Source and row-level derived data are not redistributed here.
To rebuild the project, obtain Au99.99 daily OHLC data from a source whose terms
permit your intended use and save it as:

`RMBGoldRoundNumberEffect/data/raw/Au9999.csv`

Expected fields are `date`, `open`, `high`, `low`, and `close`. A volume
field may be present, but volume is excluded from the final analysis because its
quality could not be validated. The public repository retains only code and
aggregate tables, figures, and reports.

## Reproducing

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r RMBGoldRoundNumberEffect/requirements.txt

python RMBGoldRoundNumberEffect/src/audit_sina_au9999.py
python RMBGoldRoundNumberEffect/src/construct_research_variables.py
python RMBGoldRoundNumberEffect/src/identify_stage2_5_anomaly_candidates.py
python RMBGoldRoundNumberEffect/src/build_analysis_dataset.py
python RMBGoldRoundNumberEffect/src/analyze_training_distance_distribution.py
python RMBGoldRoundNumberEffect/src/test_roundnumber_avoidance.py
python RMBGoldRoundNumberEffect/src/test_level_proximity_volatility.py
```

The scripts use project-relative paths and fixed random seeds. Generated files are
written under `outputs/`. Row-level outputs are intentionally ignored by Git and
should remain local; aggregate Stage 3–5 results are retained for inspection.

## Validation

Stage 2 (`construct_research_variables.py`) runs an automated check suite on every
execution and writes the results to
`outputs/stage2_variable_construction/variable_construction_validation_report.md`.
The checks test consequences of correct variable construction rather than restating
the formulas, so a genuine bug trips them.

- The strongest check, `volume_20d_baseline_lagged_no_current_or_future`, recomputes
  the 20-day rolling volume baseline by explicit row slicing (t−1 back through t−20)
  and asserts it matches the vectorised `.shift(1).rolling(20)` result — an automated
  guard against look-ahead bias in variable construction.
- Each check's pass/fail and detail are recorded in the report rather than raised, so
  the artefacts always show which checks ran and what they found.

## Repository structure

- `RMBGoldRoundNumberEffect/src/` — data audit, variable construction, and tests
- `outputs/stage3_distance_distribution/` — descriptive aggregate results
- `outputs/stage4_roundnumber_avoidance_test/` — occupancy-test results
- `outputs/stage5_level_proximity_volatility/` — next-day-volatility-test results
- `RESEARCH_REPORT.md` — concise research write-up

## Known limitations and possible refinements

**Additive level grid against multiplicative price dynamics.** Levels are spaced every
50 RMB, a fixed additive grid, while the null resamples log returns, which are
multiplicative. Over the training window the price roughly tripled, so a 50-RMB gap
falls from about 33% of price at 150 to about 11% at 440. The price sweeps across levels
far more quickly in the later years, and the pooled 14-year statistic is weighted toward
that period rather than being a uniform average over the sample.

If a round-number effect exists only where the grid is coarse relative to daily moves,
pooling would obscure it. The natural refinement is to repeat the analysis split by price
band (for example 150–250, 250–350, 350–450) and check whether the conclusion holds
within each. This has not been done.

## Scope and status

Training window: 2006–2020. Observations from 2021 onward were held out and were not
used after the training-sample tests returned null results. Further work would
require credible intraday or order-level data; the current daily-data study is
therefore paused.
