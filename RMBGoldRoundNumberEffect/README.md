# RMBGoldRoundNumberEffect

Python source code for the exploratory study of round-number effects in
RMB-denominated Au99.99 daily prices. See the repository-level
[README](../README.md) and [research report](../RESEARCH_REPORT.md) for the question,
results, limitations, and full run order.

## Local input

Place a locally obtained Au99.99 daily OHLC file at:

`data/raw/Au9999.csv`

The file is intentionally excluded from version control. Expected fields are
`date`, `open`, `high`, `low`, and `close`. Generated row-level datasets
also remain local.

## Source files

- `audit_sina_au9999.py` — validates the input schema and flags anomalies
- `construct_research_variables.py` — builds price-distance and volatility fields
- `identify_stage2_5_anomaly_candidates.py` — produces records for manual review
- `build_analysis_dataset.py` — applies the documented exclusions and removes volume
- `analyze_training_distance_distribution.py` — produces descriptive aggregates
- `test_roundnumber_avoidance.py` — runs the occupancy block-bootstrap test
- `test_level_proximity_volatility.py` — runs the next-day-volatility test

Install dependencies from `requirements.txt` and run these scripts from the
repository root in the order shown above.
