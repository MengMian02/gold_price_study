# Stage 4: Round-Number Avoidance Test

## Scope

This stage tests whether training-sample closing prices avoid round 50-RMB levels
beyond what a mechanical price process would produce. It uses the training window
2006-01-01 to 2020-12-31 only; data from 2021 onward is held out.
It tests the distance distribution only, and does not study support/resistance,
next-day returns, volatility, or trading rules.

## Why a naive test is not enough

The distance-to-level series is highly persistent (lag-1 autocorrelation ~0.94;
the nearest 50-level changes only about 133 times across the sample). Treating
each day as an independent observation overstates significance. This stage instead
compares the empirical distance distribution to a Monte Carlo null.

## Null model

- Under the null, prices have NO round-number preference.
- Null paths start at the first training close and are built by moving-block
  bootstrap of the actual daily log returns.
- Block length: 20 trading days (preserves short-run
  autocorrelation and fat tails while breaking any price-level/return link).
- Simulations: 2000. Seed: 12345 (reproducible).

## Input

- File: `au9999_analysis_dataset.csv`
- Training observations: 3,664
- Start price for null paths: 138.21

## Statistic Comparison

| statistic | empirical | null_median | null_p2.5 | null_p97.5 | empirical_percentile_in_null | two_sided_p | outside_95_null_band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean_distance | 12.784378 | 12.416983 | 10.458997 | 14.455598 | 65.5 | 0.6907 | False |
| prop_distance_le_2.0 | 0.06714 | 0.079967 | 0.042849 | 0.12309 | 22.8 | 0.4628 | False |
| prop_distance_le_3.0 | 0.099618 | 0.120906 | 0.067413 | 0.180411 | 21.1 | 0.4288 | False |
| prop_distance_le_5.0 | 0.172216 | 0.201146 | 0.119808 | 0.292856 | 22.4 | 0.4518 | False |
| prop_distance_le_10.0 | 0.380731 | 0.402566 | 0.282983 | 0.526754 | 34.2 | 0.6857 | False |

`empirical_percentile_in_null` is where the empirical value sits within the null
distribution (50 = dead centre). `outside_95_null_band` is the decision flag.

## Per-Bin Empirical vs Null Band

| distance_bin_yuan | empirical_proportion | null_median | null_p2.5 | null_p97.5 | empirical_below_null_band | empirical_above_null_band |
| --- | --- | --- | --- | --- | --- | --- |
| [0,1) | 0.028657 | 0.039847 | 0.020463 | 0.063053 | False | False |
| [1,2) | 0.03548 | 0.04012 | 0.021288 | 0.0625 | False | False |
| [2,3) | 0.032478 | 0.04012 | 0.023465 | 0.060862 | False | False |
| [3,4) | 0.037937 | 0.039847 | 0.02429 | 0.059771 | False | False |
| [4,5) | 0.03357 | 0.039574 | 0.02429 | 0.05786 | False | False |
| [5,6) | 0.037118 | 0.040393 | 0.025102 | 0.056496 | False | False |
| [6,7) | 0.043668 | 0.04012 | 0.026201 | 0.056502 | False | False |
| [7,8) | 0.037118 | 0.039847 | 0.027293 | 0.055404 | False | False |
| [8,9) | 0.042303 | 0.04012 | 0.027566 | 0.055131 | False | False |
| [9,10) | 0.045579 | 0.04012 | 0.027838 | 0.054039 | False | False |
| [10,11) | 0.04476 | 0.040393 | 0.028377 | 0.054585 | False | False |
| [11,12) | 0.042031 | 0.040666 | 0.029476 | 0.054592 | False | False |
| [12,13) | 0.048308 | 0.040393 | 0.028657 | 0.054585 | False | False |
| [13,14) | 0.044487 | 0.04012 | 0.028657 | 0.053493 | False | False |
| [14,15) | 0.038755 | 0.039574 | 0.026747 | 0.052948 | False | False |
| [15,16) | 0.042849 | 0.039574 | 0.027566 | 0.053766 | False | False |
| [16,17) | 0.042849 | 0.039301 | 0.026201 | 0.054039 | False | False |
| [17,18) | 0.042576 | 0.039301 | 0.025928 | 0.053773 | False | False |
| [18,19) | 0.043395 | 0.039301 | 0.025375 | 0.055404 | False | False |
| [19,20) | 0.037391 | 0.039301 | 0.024556 | 0.05595 | False | False |
| [20,21) | 0.037937 | 0.039028 | 0.023745 | 0.057314 | False | False |
| [21,22) | 0.039028 | 0.038755 | 0.022926 | 0.058133 | False | False |
| [22,23) | 0.039028 | 0.038755 | 0.022653 | 0.059771 | False | False |
| [23,24) | 0.039301 | 0.038483 | 0.021288 | 0.062507 | False | False |
| [24,25) | 0.043395 | 0.038619 | 0.020469 | 0.062507 | False | False |

## Chart

![Empirical vs null band](distance_empirical_vs_null_band.png)

## Interpretation

The empirical distance statistics all fall INSIDE the 95% no-preference null band. The apparent avoidance of round 50-RMB levels is statistically indistinguishable from what a mechanical price process with the same return behaviour produces. We cannot claim a genuine round-number avoidance effect from this evidence.

- Mean distance: empirical 12.784378 vs null median 12.416983 (95% null band 10.458997 to 14.455598), two-sided p = 0.6907.
- Distance bin [0,1): empirical 0.028657 vs null median 0.039847 (band 0.020463 to 0.063053).
- Of the 5 bins nearest a level (0-5 yuan), 0 sit below the null band.

## Caveats

- A single empirical path is compared to a null band; this is a parametric-style
  bootstrap test, not a large-sample asymptotic test.
- The block bootstrap preserves the marginal return distribution and short-run
  dependence but assumes returns are otherwise level-independent under the null.
- Failing to reject the null is not proof of no effect; it means this dataset and
  test do not provide evidence for one.
