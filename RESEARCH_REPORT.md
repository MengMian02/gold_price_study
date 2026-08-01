# Round-number effects in RMB Au99.99 gold: an exploratory study

## Summary

This project examines whether daily RMB-denominated Au99.99 gold prices behave
differently near round 50-yuan levels. Under an autocorrelation-aware moving-block
bootstrap, I find no detectable occupancy or next-day volatility effect. The result
is conditional on this instrument, frequency, sample, level definition, and null
model; it is not evidence that round-number behaviour is absent in every market.

## Question

Do gold closing prices cluster at, avoid, or react to round 50-yuan levels?

## Data

The underlying dataset contains daily open, high, low, and close prices from 2002
to 2026. Four dates judged unreliable were excluded and volume was dropped because
I could not validate its quality. The confirmatory analysis uses 3,664 observations
from 2006–2020; observations from 2021 onward were held out.

Source and row-level derived data are not redistributed in the public repository.
The code documents the required schema and rebuilds all analysis files locally.

## Method

Daily price levels are highly persistent, so treating 3,664 days as independent
observations understates sampling uncertainty. I therefore construct a no-effect
null by resampling daily log returns in 20-day moving blocks, cumulating them into
synthetic price paths, and recalculating the same round-number statistics.

The reported tests use 2,000 simulations and a fixed random seed. The block
bootstrap preserves the empirical return distribution and short-run dependence,
while imposing no price-level rule. This is still a modelling choice: results are
properly read as conditional on the specified bootstrap null.

## Results

### Occupancy near round levels

Within the training sample, 17.2% of closes were within 5 yuan of a 50-yuan level,
compared with a 20% uniform benchmark. A naive independent-observation test gave
`p ≈ 0.005`, but that benchmark does not account for persistence.

Under the block-bootstrap null, the observed share lay near the 22nd percentile of
the simulated distribution; the two-sided p-value was approximately 0.45. The mean
distance and all tested proximity shares also remained inside their 95% null bands.

### Next-day volatility

The pre-specified statistic compared the next day's absolute log return after near-
level and far-from-level closes. At the 5-yuan threshold, the estimated difference
was about 0.00060. It lay near the 84th percentile of the null distribution, with a
two-sided p-value of approximately 0.32.

Neither test provides evidence of a daily round-number effect under the specified
design.

## Interpretation

The main methodological lesson is that a persistent price path can generate an
apparently non-uniform distance distribution even without a round-number mechanism.
A simulation-based null makes that dependence explicit and produces much wider,
more credible uncertainty than an independence-based test.

The null result may also reflect limited statistical power or inappropriate data
granularity. Behaviour around psychological levels is more plausibly visible in
intraday transactions, signed order flow, or order-book data than in one daily
observation.

## Limitations

- One instrument, one 50-yuan grid, and one training window were studied.
- The block length and return-resampling mechanism define the null and may not
  capture every feature of the data-generating process.
- Daily OHLC data cannot directly observe order placement or trader direction.
- A single persistent price path provides limited effective information.
- Volume was excluded because its quality could not be validated.

## Status

The project is paused. Continuing credibly would require intraday or order-level
data, which was not available at a reasonable cost. I therefore report the null
result and data limitation rather than add more specifications after seeing the
outcome.
