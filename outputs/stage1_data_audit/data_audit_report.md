# Stage 1 Data Audit Report: Sina Au99.99 Daily OHLCV

## Scope

This is a data audit only. It detects and reports; it never cleans, filters,
imputes, corrects, or reorders the raw CSV, and it returns its input unmodified.
Disposition of any finding (keep, exclude, set-nan) is recorded separately in
`data/decisions.csv`, not here.

## Source and Provenance

- Source file used: `RMBGoldRoundNumberEffect/data/raw/Au9999.csv`
- SHA-256: `77e01ca0a9a5e35106c66cc2fc3b9c7e67ba22807c9b2cf2a7f9a8d05dba9540`
- Size (bytes): 236353
- Notes: Requested filename Au9999_Sina_Daily_OHLCV.csv was not found; using actual file RMBGoldRoundNumberEffect/data/raw/Au9999.csv.
- Rows: 5788
- Columns: date, open, high, low, close, volume_kg
- Expected columns: date, open, high, low, close, volume_kg
- Missing expected columns: none
- Extra columns: none

The SHA-256 above is also written to `source_manifest.json`; Stage 2 compares it
against the file it reads, so the audited file and the analysed file cannot
silently diverge.

## Why These Checks Matter

- File and column checks ensure the audit is being applied to the intended dataset.
- Date parsing, sorting, and duplicate checks protect time-series calculations from hidden ordering or identity errors. Rows are sorted by date before any change statistic is computed, so a mis-ordered file cannot corrupt the jump statistics; a mis-ordered raw file is itself reported as an error.
- Calendar-gap checks compare adjacent trading rows and count skipped weekdays, surfacing runs of potentially missing trading days.
- Missing, non-numeric, zero, and negative value checks prevent invalid prices or volumes from contaminating later returns and volatility calculations.
- OHLC logic checks catch hard data errors where open or close falls outside the reported daily low-high range.
- Extreme close-to-close, volume, and intraday-range changes are warning flags only; they may reflect real market moves, source glitches, unit changes, or data-entry problems.

## Date Audit

- Date range: 2002-10-30 to 2026-07-28
- Date parse failures: 0
- Duplicate date rows: 0
- Sorted by parsed date (original file order): True

## Missing Values

| field | missing_count |
| --- | --- |
| date | 0 |
| open | 0 |
| high | 0 |
| low | 0 |
| close | 0 |
| volume_kg | 0 |

## Numeric Validity

### Non-Numeric Values

| field | non_numeric_count |
| --- | --- |
| open | 0 |
| high | 0 |
| low | 0 |
| close | 0 |
| volume_kg | 0 |

### Zero or Negative Values

| field | non_positive_or_zero_count |
| --- | --- |
| open | 0 |
| high | 0 |
| low | 0 |
| close | 0 |
| volume_kg | 0 |

## Anomaly Summary

- Total anomaly table rows: 279
- Error-level flags: 0
- Warning-level flags: 279

| severity | rule | count |
| --- | --- | --- |
| warning | extreme_intraday_range | 63 |
| warning | large_close_change_warning | 15 |
| warning | large_volume_change_warning | 41 |
| warning | weekday_gap | 160 |

## Rules and Thresholds

- `date_not_sorted` (error): raw file dates not in ascending order.
- `duplicate_date` (error): a date appears on more than one row.
- `date_parse_failed` / `missing_*` / `non_numeric_*` / `non_positive_*` (error): field-level validity.
- `open_outside_low_high` / `close_outside_low_high` (error): OHLC bracket violated.
- `weekday_gap` (warning): consecutive rows skip one or more weekdays.
- `large_unexplained_gap` (error): a gap skips more than 10 intervening weekdays.
- `large_close_change_warning` (warning): absolute daily close-to-close change > 10.00%.
- `large_volume_change_warning` (warning): absolute daily volume change > 5.00 times previous day.
- `extreme_intraday_range` (warning): `(high-low)/close` > 10.00%.
- Robust outlier warning: robust z-score absolute value > 8.00 (applied to close change, volume change, and intraday range).

## Coverage and Limitations

A clean report does not prove clean data. Stage 1 does **not**:

- consult an exchange trading calendar — calendar-gap detection is a weekday-count
  proxy and cannot by itself tell a genuine SGE holiday from a scraping failure;
- verify prices against any second source — an internally consistent but wrong
  price passes every check here;
- detect a unit change (e.g. a currency or scale change) applied uniformly across
  the whole series, since nothing about it looks anomalous row-to-row.

## Interpretation

The dataset is not altered by this audit. Error-level rows should be reviewed before downstream cleaning. Warning-level rows should not be automatically removed; they require source or market-context review.

Generated files:

- `data_audit_report.md`
- `data_audit_anomalies.csv`
- `source_manifest.json`
