# Round-number effects in RMB Au99.99 gold: an exploratory study

See README.md for the summary. This document is the detailed record.

## Data

The underlying dataset contains daily open, high, low, and close prices from 2002
to 2026. Four dates judged unreliable were excluded and volume was dropped because
I could not validate its quality. The confirmatory analysis uses 3,664 observations
from 2006–2020; observations from 2021 onward were held out.

Source and row-level derived data are not redistributed in the public repository.
The code documents the required schema and rebuilds all analysis files locally.

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
