# Stage 5: Level-Proximity and Next-Day Volatility

## Scope

Single pre-committed confirmatory test of support/resistance, run after the
occupancy test (Stage 4) and a descriptive support/resistance look both came back
null. Training window 2006-01-01 to 2020-12-31; 2021+ held out.
One statistic, one null model, no further variants (to avoid data-dredging).

## Question

No look-ahead: is today's absolute log-return larger or smaller when YESTERDAY's
close sat near a 50-level? Statistic = mean|return| after a near-level close minus
mean|return| after a far-level close.

## Null model

Block bootstrap of daily log-returns (block 20, 2000 sims,
seed 12345). Return sizes are unlinked from price level, so any real link
shows as the statistic escaping the null band. The null also captures any purely
mechanical bias (its median need not be zero).

## Result

- Training observations: 3,664

| near_threshold_yuan | n_days_near | mean_abs_return_after_near | mean_abs_return_after_far | statistic_near_minus_far | null_median | null_p2.5 | null_p97.5 | empirical_percentile_in_null | two_sided_p | outside_95_null_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | 246 | 0.007704 | 0.007331 | 0.000374 | -1.2e-05 | -0.00121 | 0.001987 | 68.7 | 0.6277 | False |
| 3.0 | 365 | 0.00786 | 0.0073 | 0.00056 | 3e-06 | -0.001138 | 0.001626 | 78.8 | 0.4258 | False |
| 5.0 | 631 | 0.007848 | 0.007253 | 0.000595 | 5e-06 | -0.001012 | 0.001413 | 84.0 | 0.3198 | False |
| 10.0 | 1394 | 0.007798 | 0.007084 | 0.000714 | 6e-06 | -0.000978 | 0.001035 | 91.5 | 0.1699 | False |

`outside_95_null_band` is the decision flag.

## Chart

![Null distribution](level_proximity_null_distribution.png)

## Interpretation

INSIDE the null band -> no detectable link. Being near a round level yesterday does not predict a different-sized move today beyond what a mechanical random walk produces.

## Caveats

- One 15-year path compared to a null band (parametric-bootstrap-style test).
- Failing to reject is not proof of no effect; it means this test found no evidence.
- The block bootstrap breaks the price-level/return link under the null but assumes
  returns are otherwise level-independent.
