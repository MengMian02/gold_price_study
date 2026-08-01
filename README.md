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

## Repository structure

- `RMBGoldRoundNumberEffect/src/` — data audit, variable construction, and tests
- `outputs/stage3_distance_distribution/` — descriptive aggregate results
- `outputs/stage4_roundnumber_avoidance_test/` — occupancy-test results
- `outputs/stage5_level_proximity_volatility/` — next-day-volatility-test results
- `RESEARCH_REPORT.md` — concise research write-up

## Scope and status

Training window: 2006–2020. Observations from 2021 onward were held out and were not
used after the training-sample tests returned null results. Further work would
require credible intraday or order-level data; the current daily-data study is
therefore paused.
