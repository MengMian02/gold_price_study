# Threshold-change diff: `(2.5, 5.0, 7.5, 10.0)` → `(2.0, 3.0, 5.0, 10.0)`

This records the effect of standardising the robustness near-thresholds on
`(2, 3, 5, 10)` (Stage 2 task 7). `main_near_threshold = 5` is the pre-specified
headline and is unchanged; the other thresholds are robustness checks.

The pre-change outputs (old thresholds) are preserved in git history at the commit
immediately before this one; the live `outputs/stage3_*`, `stage4_*`, `stage5_*` are
the **new** outputs. The tables below record old vs new for every statistic. (Local
copies of the pre-change outputs were kept alongside this file during the migration
but are intentionally not committed, since git history already preserves them.)

## Verification headline

- **The threshold-5.0 rows are numerically identical in every stage** (see bold
  rows below). Because each threshold's statistic is computed independently and
  the Monte Carlo null uses a fixed seed, changing the *list* of thresholds moves
  only the rows for thresholds that were added or removed.
- Threshold-**independent** outputs are unchanged: Stage 3
  `distance_description.csv` and `distance_1yuan_bins.csv`, Stage 4
  `mean_distance` and `per_bin_null_band.csv`, and all three `test_summary.json`
  conclusions.
- Every conclusion is still null (nothing escapes the 95% null band).

## Stage 3 — `distance_threshold_counts.csv`

OLD (removed rows `~~2.5~~`, `~~7.5~~`):

| threshold | obs | actual_prop | uniform_ref | actual − ref |
|---|---|---|---|---|
| ~~2.5~~ | 300 | 0.08187773 | 0.10 | −0.01812227 |
| **5.0** | **631** | **0.17221616** | **0.20** | **−0.02778384** |
| ~~7.5~~ | 993 | 0.27101528 | 0.30 | −0.02898472 |
| 10.0 | 1395 | 0.38073144 | 0.40 | −0.01926856 |

NEW (added rows `2.0`, `3.0`):

| threshold | obs | actual_prop | uniform_ref | actual − ref |
|---|---|---|---|---|
| 2.0 | 246 | 0.06713974 | 0.08 | −0.01286026 |
| 3.0 | 365 | 0.09961790 | 0.12 | −0.02038210 |
| **5.0** | **631** | **0.17221616** | **0.20** | **−0.02778384** |
| 10.0 | 1395 | 0.38073144 | 0.40 | −0.01926856 |

5.0 and 10.0 rows: **identical**.

## Stage 4 — `statistic_comparison.csv`

`mean_distance` (threshold-independent) is unchanged:
`empirical 12.784378, null_median 12.416983, p2.5 10.458997, p97.5 14.455598,
pct 65.5, two_sided_p 0.6907, outside_band False`.

OLD proportion rows:

| statistic | empirical | null_median | null_p2.5 | null_p97.5 | pct | two_sided_p | outside |
|---|---|---|---|---|---|---|---|
| ~~prop_distance_le_2.5~~ | 0.081878 | 0.100710 | 0.055131 | 0.152572 | 20.5 | 0.4198 | False |
| **prop_distance_le_5.0** | **0.172216** | **0.201146** | **0.119808** | **0.292856** | **22.4** | **0.4518** | **False** |
| ~~prop_distance_le_7.5~~ | 0.271015 | 0.301310 | 0.197591 | 0.416492 | 27.0 | 0.5467 | False |
| prop_distance_le_10.0 | 0.380731 | 0.402566 | 0.282983 | 0.526754 | 34.2 | 0.6857 | False |

NEW proportion rows:

| statistic | empirical | null_median | null_p2.5 | null_p97.5 | pct | two_sided_p | outside |
|---|---|---|---|---|---|---|---|
| prop_distance_le_2.0 | 0.067140 | 0.079967 | 0.042849 | 0.123090 | 22.8 | 0.4628 | False |
| prop_distance_le_3.0 | 0.099618 | 0.120906 | 0.067413 | 0.180411 | 21.1 | 0.4288 | False |
| **prop_distance_le_5.0** | **0.172216** | **0.201146** | **0.119808** | **0.292856** | **22.4** | **0.4518** | **False** |
| prop_distance_le_10.0 | 0.380731 | 0.402566 | 0.282983 | 0.526754 | 34.2 | 0.6857 | False |

5.0 and 10.0 rows: **identical**. `per_bin_null_band.csv`: **identical**.

## Stage 5 — `level_proximity_volatility_stats.csv`

OLD (thresholds 2.0, 5.0):

| thr | n_near | mean|r|_near | mean|r|_far | stat | null_med | p2.5 | p97.5 | pct | p | outside |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 246 | 0.007704 | 0.007331 | 0.000374 | −0.000012 | −0.001210 | 0.001987 | 68.7 | 0.6277 | False |
| **5.0** | **631** | **0.007848** | **0.007253** | **0.000595** | **0.000005** | **−0.001012** | **0.001413** | **84.0** | **0.3198** | **False** |

NEW (thresholds 2.0, 3.0, 5.0, 10.0):

| thr | n_near | mean|r|_near | mean|r|_far | stat | null_med | p2.5 | p97.5 | pct | p | outside |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 246 | 0.007704 | 0.007331 | 0.000374 | −0.000012 | −0.001210 | 0.001987 | 68.7 | 0.6277 | False |
| 3.0 | 365 | 0.007860 | 0.007300 | 0.000560 | 0.000003 | −0.001138 | 0.001626 | 78.8 | 0.4258 | False |
| **5.0** | **631** | **0.007848** | **0.007253** | **0.000595** | **0.000005** | **−0.001012** | **0.001413** | **84.0** | **0.3198** | **False** |
| 10.0 | 1394 | 0.007798 | 0.007084 | 0.000714 | 0.000006 | −0.000978 | 0.001035 | 91.5 | 0.1699 | False |

2.0 and 5.0 rows: **identical**. (Stage 5 counts prior-day distances over the
n−1 return days, so its 10.0 count is 1394 vs Stage 3's full-sample 1395; this is
by design, not a threshold effect.)
