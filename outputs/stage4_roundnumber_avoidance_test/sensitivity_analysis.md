# Stage 4 Sensitivity: Robustness of the Round-Number Occupancy Null

The observed statistic is fixed throughout: `prop_distance_le_5` = 0.1722 on the real training closes. Each variant below changes only how the *null* is built, then reports where that observed value falls. Simulations use n_sim = 2000 and seed 12345; the block-bootstrap engine is imported from `test_roundnumber_avoidance.py`. The block-20 rows are read from the existing `statistic_comparison.csv`, not recomputed.

## 1a. Block length

| variant | null_median | null_p2.5 | null_p97.5 | band_width | observed_percentile |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.1995 | 0.1239 | 0.2809 | 0.1570 | 19.9 |
| 20 (main run, read from output) | 0.2011 | 0.1198 | 0.2929 | 0.1730 | 22.4 |
| 50 | 0.1976 | 0.1135 | 0.2942 | 0.1807 | 25.4 |
| 100 | 0.1990 | 0.1141 | 0.2980 | 0.1840 | 26.2 |

Block length does not materially affect the conclusion: the observed value stays near the same percentile and well inside the band at every block length, because the persistence that matters comes from integrating returns into a price level, not from block ordering.

## 1b. What gets resampled

| variant | null_median | null_p2.5 | null_p97.5 | band_width | observed_percentile |
| --- | --- | --- | --- | --- | --- |
| log returns, multiplicative (current) | 0.2011 | 0.1198 | 0.2929 | 0.1730 | 22.4 |
| absolute RMB diffs, additive, floored at 1 | 0.2029 | 0.1427 | 0.3030 | 0.1603 | 18.4 |

Resampling absolute differences instead of log returns does not change the conclusion: the observed value remains inside the null band under both schemes.

## 1c. Terminal price plausibility

Derived from the 2,000 block-20 null paths the main run already produces (reproduced here from the same seed, so identical); no new simulation.

| quantity | value |
| --- | --- |
| terminal price p2.5 | 112.0 |
| terminal price p25 | 246.8 |
| terminal price p50 | 379.0 |
| terminal price p75 | 598.6 |
| terminal price p97.5 | 1329.9 |
| share of paths with max > 1000 RMB/g | 0.0905 |
| corr(prop_le_5, log terminal price) | -0.0073 |

The null paths stay in a plausible price range and occupancy is only weakly related to where a path ends up, so terminal-price drift does not materially distort the null.

## 1d. Start price

| variant | null_median | null_p2.5 | null_p97.5 | band_width | observed_percentile |
| --- | --- | --- | --- | --- | --- |
| fixed start = 138.21 (current) | 0.2011 | 0.1198 | 0.2929 | 0.1730 | 22.4 |
| start + U(0, 50) | 0.1990 | 0.1299 | 0.2825 | 0.1526 | 19.6 |

Jittering the start price by a full level spacing does not change the conclusion: sweeping the start across one grid spacing barely moves the null, and the observed value stays inside the band.
