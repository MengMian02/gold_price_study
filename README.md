# Round-Number Effects in RMB Gold Prices

Do prices cluster at, or avoid, round numbers? This project tests that on 14 years of Shanghai Gold Exchange Au99.99 daily data, and finds that the answer depends almost entirely on how serial dependence is handled.

The standard independence-based test reports a 4.4σ deviation. A moving-block bootstrap on the same data, same statistic, reports p = 0.45.

| | Uncertainty on near-level occupancy | Verdict |
|---|---|---|
| Assuming independent daily observations | ±0.7pp | −4.4σ, p < 0.0001 |
| Moving-block bootstrap | ±4.4pp | p = 0.45 |

Daily closes are not independent draws. The distance from the nearest 50-RMB level has lag-1 autocorrelation of about 0.94, and the nearest level changes only about 133 times across 3,664 trading days. The nominal sample is 3,664 observations; the effective sample for this question is far smaller. Independence-based standard errors understate the true uncertainty by a factor of roughly 6.7.

The substantive conclusion — no detectable round-number effect at daily frequency — is reported below with the power analysis needed to interpret it.

## What is tested

**Stage 4 — occupancy.** Do closes spend less time near 50-RMB levels than a no-effect price process would produce?

**Stage 5 — volatility.** Is the next day's move larger after a close near a level? This is the support-and-resistance hypothesis, pre-committed as the single confirmatory dynamics test.

## How the null is built

The counterfactual — gold behaving identically but with no level preference — cannot be observed, so it is built from the data. The real return series is cut into 20-day blocks, resampled with replacement, and cumulated into 2,000 synthetic price paths.

Preserved: fat tails, volatility clustering, drift, within-block autocorrelation.
Destroyed: any relationship between price level and subsequent returns — the thing under test.

## Results

| Test | Observed | Null median | 95% null band | Percentile | p |
|---|---|---|---|---|---|
| Occupancy within 5 RMB | 17.2% | 20.1% | 12.0%–29.3% | 22.4 | 0.45 |
| Next-day volatility, near vs far | +8.2% | ~0 | −0.10 to +0.14pp | 84.0 | 0.32 |

Neither is distinguishable from a no-effect null.

## Power — what these nulls can and cannot support

| Test | Minimum detectable effect |
|---|---|
| Occupancy | ±61% |
| Next-day volatility | ±24% |

Near-level occupancy would have to fall from about 20% to under 8% before the occupancy test could reliably detect it. Round-number effects reported in the equity literature are typically a few percent.

These null results rule out large effects. They are uninformative about effects of the magnitude the literature reports. The observed volatility effect is roughly a third of what its own design can resolve. A null result without a power figure cannot be interpreted; that is why the power figure is reported.

## The two results are one finding, not two

Higher volatility near a level would cause faster escape from it, and therefore lower occupancy. The two tests fit a single mechanism — which is why they do not corroborate each other. If the volatility effect were real and large it would mechanically produce the occupancy effect. They are one finding seen from two angles. Both stages also share a random seed, so their null paths are identical; each test is individually valid, but the two are not statistically independent.

## Methodology notes

**Look-ahead is guarded.** Grouping days by their contemporaneous distance to a level would contaminate the comparison — the "near" group would absorb high-volatility days that jumped in from far away, diluting any real effect toward null. Stage 5 groups on the prior close. Stage 2's validation suite recomputes the 20-day volume baseline by explicit row slicing and asserts it matches the vectorised version — an automated guard against look-ahead in variable construction.

**The null median is reported, not assumed.** Stage 4's simulated paths have zero round-number preference by construction, yet their median mean-distance is 12.42, not the uniform 12.50. Testing against the theoretical value would attribute a mechanical artefact to round numbers. Stage 5's null median is approximately zero, confirming its design introduces no such bias.

**Construction choices are tested, not asserted.** `sensitivity_roundnumber_null.py` reports the effect of block length, resampling target, path plausibility, and start price on the null band. Block length does not matter — the persistence comes from integrating returns into a price level, not from autocorrelation in returns. Resampling absolute price differences instead of log returns matters and is rejected: it lets null paths drift toward zero, where a fixed 50-RMB grid becomes coarse relative to price and the occupancy statistic degenerates.

**Thresholds are pre-specified.** The main near-threshold is 5 RMB, fixed in configuration. Thresholds 2, 3, and 10 are robustness checks. They are nested, so consistency across them is one number reported four times, not four signals agreeing.

**Data decisions are logged, not hardcoded.** `data/decisions.csv` records every audit flag with a disposition, reason, reviewer, and date — including flags reviewed and retained. The audit detects; a human decides; the decision is versioned and applied by code.

## Known limitations

**Statistical power.** The binding constraint on the substantive conclusion. See above.

**Daily frequency.** A ±2 RMB band is roughly one typical daily move at these price levels. The state "price sitting near a level" is barely observable at daily frequency — the price can move from outside the band to outside the band within one session. A support-and-resistance mechanism operates intraday; daily closes see only a shadow of it.

**Additive level grid against multiplicative price dynamics.** Levels are spaced every 50 RMB, a fixed additive grid, while the null resamples log returns. The price roughly tripled over the training window, so a 50-RMB gap falls from about 33% of price to about 11%. The price sweeps across levels faster in the later years, and the pooled statistic is weighted toward that period. If an effect exists only where the grid is coarse relative to daily moves, pooling would obscure it. The natural refinement — repeating the analysis split by price band — has not been done.

**Volume was excluded.** Its quality could not be validated against a second source.

**Trading calendar.** Gap detection uses a weekday-count proxy. Distinguishing a genuine SGE holiday from a scraping failure requires an exchange trading calendar, which this project lacks.

**Single instrument, single market.** Au99.99 only.

## Pipeline

| Stage | Script | Purpose |
|---|---|---|
| 1 | `audit_sina_au9999.py` | Read-only data audit; flags anomalies for review |
| 2 | `construct_research_variables.py` | Level distances, returns, volatility, lagged states |
| 2.5 | `identify_stage2_5_anomaly_candidates.py` | Anomaly candidates for manual review |
| 2.6 | `build_analysis_dataset.py` | Applies logged decisions; final dataset |
| 3 | `analyze_training_distance_distribution.py` | Descriptive distance distribution |
| 4 | `test_roundnumber_avoidance.py` | Occupancy test, block bootstrap |
| 5 | `test_level_proximity_volatility.py` | Volatility test, pre-committed |

Stage 1 detects and reports; it never deletes, corrects, or filters. Disposition is a separate, logged, human decision.

## Reproduction

Data is not redistributed. Obtain Au99.99 daily OHLC from a source whose terms permit your use, place it at `data/raw/Au9999.csv`, and run the stages in order. Training window 2006-01-01 to 2020-12-31; data from 2021 onward is held out and was not used in any test.
