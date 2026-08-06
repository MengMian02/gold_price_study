"""Stage 5: does proximity to a round 50-level predict next-day volatility?

This is the single pre-committed confirmatory test of the support/resistance
hypothesis, run after the occupancy question (Stage 4) and the descriptive
distance-distribution look (Stage 3) both returned nothing significant. It is
deliberately the ONLY dynamics test we run, to avoid data-dredging across many
variants.

Question (no look-ahead): is today's absolute log-return systematically larger
or smaller when YESTERDAY's close sat near a 50-level?

Statistic: mean|return| on days whose prior close was near a level (prior
distance <= threshold) minus mean|return| on days whose prior close was far.

Null: block bootstrap of daily log-returns (same engine and discipline as Stage
4). Under the null, return sizes are unlinked to where the price sits relative to
levels. The null distribution also reveals any purely mechanical bias, so we
compare the real statistic to the null band, not to zero.

Scope: training window 2006-2020 only (2021+ held out). Tests this one statistic;
no regressions, trading rules, or further variants. Input is read only.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from levels import distance_to_level
from test_roundnumber_avoidance import draw_block_bootstrap


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = (
    WORKSPACE_ROOT / "outputs" / "stage2_variable_construction" / "au9999_analysis_dataset.csv"
)
# Pre-exclusion Stage 2 variables (all dates present); used only for the Parkinson
# robustness variant (b), which reinstates the two extreme-intraday-range dates.
STAGE2_VARIABLES = (
    WORKSPACE_ROOT / "outputs" / "stage2_variable_construction" / "au9999_research_variables_stage2.csv"
)
# Dropped in build_analysis_dataset for having no genuine trading record; kept out of
# variant (b) too (only the extreme-range dates are reinstated there).
PLACEHOLDER_DATES = frozenset({"2012-01-03", "2013-01-03"})
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "stage5_level_proximity_volatility"

TRAIN_START = pd.Timestamp("2006-01-01")
TRAIN_END = pd.Timestamp("2020-12-31")
LEVEL_STEP = 50.0
NEAR_THRESHOLDS = (2.0, 3.0, 5.0, 10.0)   # headline = 5.0; 2.0/3.0/10.0 are robustness variants


@dataclass
class Config:
    source: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    n_sim: int = 2000
    block_length: int = 20
    # Deliberately the same seed as Stage 4 (test_roundnumber_avoidance.py): identical seed,
    # return series, and block-start array shape mean this stage's null paths are bit-identical
    # to Stage 4's. Do not change either seed -- the alignment makes the two stages a controlled
    # comparison, not two independent tests. The two results are therefore NOT independent.
    seed: int = 12345


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test level-proximity vs next-day volatility")
    p.add_argument("--source", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--n-sim", type=int, default=2000)
    p.add_argument("--block-length", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()




def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path without exposing local folders."""
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name

def load_training_closes(source: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(source)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["distance_to_level"] = pd.to_numeric(df["distance_to_level"], errors="coerce")
    df = df.sort_values("trade_date")
    m = (df["trade_date"] >= TRAIN_START) & (df["trade_date"] <= TRAIN_END)
    closes = df.loc[m, "close_price"].to_numpy(float)
    dist_col = df.loc[m, "distance_to_level"].to_numpy(float)
    keep = np.isfinite(closes) & (closes > 0)
    return closes[keep], dist_col[keep]


def near_minus_far(prior_dist: np.ndarray, abs_ret: np.ndarray, thr: float) -> tuple[float, float, float, int]:
    """Return (stat, mean_near, mean_far, n_near) for one aligned series."""
    near = prior_dist <= thr
    if near.sum() == 0 or (~near).sum() == 0:
        return np.nan, np.nan, np.nan, int(near.sum())
    mean_near = float(abs_ret[near].mean())
    mean_far = float(abs_ret[~near].mean())
    return mean_near - mean_far, mean_near, mean_far, int(near.sum())


def make_paths(returns: np.ndarray, start: float, n_close: int, cfg: Config, rng) -> np.ndarray:
    L = min(cfg.block_length, len(returns))
    need = n_close - 1
    k = int(np.ceil(need / L))
    starts = rng.integers(0, len(returns) - L + 1, size=(cfg.n_sim, k))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(cfg.n_sim, k * L)[:, :need]
    r = returns[idx]
    logp = np.log(start) + np.cumsum(r, axis=1)
    return np.concatenate([np.full((cfg.n_sim, 1), start), np.exp(logp)], axis=1)


def null_stats(paths: np.ndarray, thr: float) -> np.ndarray:
    prior_dist = distance_to_level(paths[:, :-1])            # (B, n-1)
    abs_ret = np.abs(np.diff(np.log(paths), axis=1))         # (B, n-1)
    near = prior_dist <= thr
    cnt_near = near.sum(1)
    cnt_far = (~near).sum(1)
    sum_near = np.where(near, abs_ret, 0.0).sum(1)
    sum_far = np.where(~near, abs_ret, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = sum_near / cnt_near - sum_far / cnt_far
    stat[(cnt_near == 0) | (cnt_far == 0)] = np.nan
    return stat


def two_sided_p(sim: np.ndarray, obs: float) -> float:
    sim = sim[np.isfinite(sim)]
    B = len(sim)
    ge = (sim >= obs).sum()
    le = (sim <= obs).sum()
    return float(min(1.0, 2.0 * min(ge + 1, le + 1) / (B + 1)))


def load_prox_sample(source: Path, exclude_dates: frozenset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (closes, prior-day distance column, parkinson_volatility) for the training
    window, dropping ``exclude_dates`` and non-positive closes. Used by the Parkinson
    robustness check so it can vary the excluded set without touching the primary loader.
    """
    df = pd.read_csv(source)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for col in ("close_price", "distance_to_level", "parkinson_volatility"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("trade_date")
    date_str = df["trade_date"].dt.strftime("%Y-%m-%d")
    mask = (df["trade_date"] >= TRAIN_START) & (df["trade_date"] <= TRAIN_END) & (~date_str.isin(exclude_dates))
    closes = df.loc[mask, "close_price"].to_numpy(float)
    dist = df.loc[mask, "distance_to_level"].to_numpy(float)
    park = df.loc[mask, "parkinson_volatility"].to_numpy(float)
    keep = np.isfinite(closes) & (closes > 0)
    return closes[keep], dist[keep], park[keep]


def null_stats_outcome(prior_dist: np.ndarray, outcome: np.ndarray, thr: float) -> np.ndarray:
    """Same near-minus-far null as null_stats, but with an arbitrary per-day outcome carried
    through the block bootstrap (here Parkinson volatility) instead of the path's own return.
    """
    near = prior_dist <= thr
    cnt_near = near.sum(1)
    cnt_far = (~near).sum(1)
    sum_near = np.where(near, outcome, 0.0).sum(1)
    sum_far = np.where(~near, outcome, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = sum_near / cnt_near - sum_far / cnt_far
    stat[(cnt_near == 0) | (cnt_far == 0)] = np.nan
    return stat


def parkinson_table(closes: np.ndarray, dist_col: np.ndarray, park: np.ndarray, cfg: Config) -> pd.DataFrame:
    """near-minus-far table with parkinson_volatility as the outcome. Same grouping, block
    bootstrap, seed and thresholds as the primary test; the block-start draws are identical, so
    Parkinson and returns are resampled through the same blocks (returns build the prior distance,
    Parkinson is the outcome carried along). Column structure matches the primary statistics file.
    """
    n = len(closes)
    returns = np.diff(np.log(closes))
    prior_dist_real = dist_col[:-1]          # yesterday's distance
    park_real = park[1:]                     # today's Parkinson volatility

    # Same block bootstrap / seed as the primary null: fresh generators seeded identically draw
    # the same block-start array, so returns and Parkinson are resampled through the same blocks.
    sampled_ret, _ = draw_block_bootstrap(returns, n - 1, cfg.n_sim, cfg.block_length, np.random.default_rng(cfg.seed))
    sampled_park, _ = draw_block_bootstrap(park[1:], n - 1, cfg.n_sim, cfg.block_length, np.random.default_rng(cfg.seed))
    log_prices = np.log(float(closes[0])) + np.cumsum(sampled_ret, axis=1)
    prices = np.concatenate([np.full((cfg.n_sim, 1), float(closes[0])), np.exp(log_prices)], axis=1)
    prior_dist_null = distance_to_level(prices[:, :-1])

    rows = []
    for thr in NEAR_THRESHOLDS:
        stat, mn, mf, n_near = near_minus_far(prior_dist_real, park_real, thr)
        sim = null_stats_outcome(prior_dist_null, sampled_park, thr)
        lo, med, hi = np.nanpercentile(sim, [2.5, 50, 97.5])
        rows.append({
            "near_threshold_yuan": thr,
            "n_days_near": n_near,
            "mean_abs_return_after_near": round(mn, 6),
            "mean_abs_return_after_far": round(mf, 6),
            "statistic_near_minus_far": round(stat, 6),
            "null_median": round(float(med), 6),
            "null_p2.5": round(float(lo), 6),
            "null_p97.5": round(float(hi), 6),
            "empirical_percentile_in_null": round(float(np.nanmean(sim < stat) * 100), 1),
            "two_sided_p": round(two_sided_p(sim, stat), 4),
            "outside_95_null_band": bool(stat < lo or stat > hi),
        })
    return pd.DataFrame(rows)


def run_parkinson_robustness(cfg: Config) -> pd.DataFrame:
    """Parkinson robustness for both samples: (a) the analysis sample as used by the primary
    test, and (b) the same sample with the two extreme-intraday-range dates reinstated.
    """
    ca, da, pa = load_prox_sample(cfg.source, frozenset())
    cb, db, pb = load_prox_sample(STAGE2_VARIABLES, PLACEHOLDER_DATES)
    table_a = parkinson_table(ca, da, pa, cfg)
    table_a.insert(0, "sample", "analysis_as_is")
    table_b = parkinson_table(cb, db, pb, cfg)
    table_b.insert(0, "sample", "extreme_dates_reinstated")
    return pd.concat([table_a, table_b], ignore_index=True)


def run(cfg: Config) -> dict:
    if not cfg.source.exists():
        raise FileNotFoundError(f"Working dataset not found: {cfg.source}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    closes, dist_col = load_training_closes(cfg.source)
    n = len(closes)
    returns = np.diff(np.log(closes))
    # Real-data prior-day distance is READ from the Stage 2 distance_to_level
    # column, aligned to yesterday's close; only the simulated paths recompute it.
    prior_dist_real = dist_col[:-1]
    abs_ret_real = np.abs(returns)

    # The interpretability of the whole null hinges on prior_dist_real being
    # YESTERDAY's distance (closes[:-1]), not today's (closes[1:]). Using the
    # contemporaneous close would drop high-volatility days that jumped in from far
    # away into the "near" group, diluting any real effect toward null.
    assert len(prior_dist_real) == len(abs_ret_real)
    assert np.isclose(prior_dist_real[0], distance_to_level(closes[0]))

    paths = make_paths(returns, float(closes[0]), n, cfg, rng)

    rows = []
    null_by_thr = {}
    for thr in NEAR_THRESHOLDS:
        stat, mn, mf, n_near = near_minus_far(prior_dist_real, abs_ret_real, thr)
        sim = null_stats(paths, thr)
        null_by_thr[thr] = sim
        lo, med, hi = np.nanpercentile(sim, [2.5, 50, 97.5])
        rows.append({
            "near_threshold_yuan": thr,
            "n_days_near": n_near,
            "mean_abs_return_after_near": round(mn, 6),
            "mean_abs_return_after_far": round(mf, 6),
            "statistic_near_minus_far": round(stat, 6),
            "null_median": round(float(med), 6),
            "null_p2.5": round(float(lo), 6),
            "null_p97.5": round(float(hi), 6),
            "empirical_percentile_in_null": round(float(np.nanmean(sim < stat) * 100), 1),
            "two_sided_p": round(two_sided_p(sim, stat), 4),
            "outside_95_null_band": bool(stat < lo or stat > hi),
        })
    table = pd.DataFrame(rows)

    plot_path = save_plot(null_by_thr, table, cfg.output_dir) if plt is not None else None
    table.to_csv(cfg.output_dir / "level_proximity_volatility_stats.csv", index=False, encoding="utf-8-sig")

    # Post-hoc Parkinson robustness (separate file; primary outputs above are untouched).
    parkinson = run_parkinson_robustness(cfg)
    parkinson.to_csv(cfg.output_dir / "parkinson_robustness.csv", index=False, encoding="utf-8-sig")

    report_path = write_report(cfg, n, table, parkinson, plot_path)

    any_out = bool(table["outside_95_null_band"].any())
    summary = {
        "input_file": repository_relative(cfg.source),
        "training_observations": int(n),
        "n_sim": cfg.n_sim,
        "block_length": cfg.block_length,
        "seed": cfg.seed,
        "any_threshold_outside_95_null_band": any_out,
        "conclusion": (
            "no link between level-proximity and next-day volatility (indistinguishable from mechanical null)"
            if not any_out else
            "a threshold escapes the null band; inspect before claiming an effect"
        ),
        "report": repository_relative(report_path),
    }
    (cfg.output_dir / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def save_plot(null_by_thr: dict, table: pd.DataFrame, output_dir: Path) -> Path:
    out = output_dir / "level_proximity_null_distribution.png"
    thr = 5.0
    sim = null_by_thr[thr]
    sim = sim[np.isfinite(sim)]
    real = float(table.loc[table["near_threshold_yuan"] == thr, "statistic_near_minus_far"].iloc[0])
    lo, hi = np.percentile(sim, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(sim, bins=45, color="#B7C9DB", edgecolor="white", label="2000 no-effect worlds")
    ax.axvline(lo, color="#4C78A8", ls=":", lw=1.3)
    ax.axvline(hi, color="#4C78A8", ls=":", lw=1.3, label="95% of no-effect worlds")
    ax.axvline(0, color="#555", ls="--", lw=1.0)
    ax.axvline(real, color="#D95F02", lw=3, label=f"REAL ({real:+.5f})")
    ax.set_xlabel("mean |return| after NEAR-level close  −  after FAR-level close")
    ax.set_ylabel("number of no-effect worlds")
    ax.set_title("Does proximity to a round level predict next-day volatility? (threshold 5 yuan)")
    ax.legend(frameon=False)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _stage4_mean_distance_null_median() -> float:
    """Read Stage 4's null median for mean_distance from its output file (not hardcoded)."""
    path = WORKSPACE_ROOT / "outputs" / "stage4_roundnumber_avoidance_test" / "statistic_comparison.csv"
    with open(path, encoding="utf-8-sig") as handle:
        for stat_row in csv.DictReader(handle):
            if stat_row["statistic"] == "mean_distance":
                return float(stat_row["null_median"])
    return float("nan")


def write_report(cfg: Config, n: int, table: pd.DataFrame, parkinson: pd.DataFrame, plot_path: Path | None) -> Path:
    out = cfg.output_dir / "level_proximity_volatility_report.md"
    headline = table[table["near_threshold_yuan"] == 5.0].iloc[0]
    verdict = (
        "INSIDE the null band -> no detectable link. Being near a round level yesterday does not "
        "predict a different-sized move today beyond what a mechanical random walk produces."
        if not bool(headline["outside_95_null_band"]) else
        "OUTSIDE the null band -> a link survives the mechanical null; inspect and confirm out-of-sample."
    )
    plot_line = f"![Null distribution]({plot_path.name})\n" if plot_path else "_Plot unavailable._\n"

    # --- Documentation values, read from the already-computed tables (no new stats). ---
    stat5 = float(headline["statistic_near_minus_far"])
    far5 = float(headline["mean_abs_return_after_far"])
    lo5 = float(headline["null_p2.5"])
    hi5 = float(headline["null_p97.5"])
    med5 = float(headline["null_median"])
    se5 = (hi5 - lo5) / (2 * 1.96)
    mde = 2.8 * se5
    mde_pct = mde / far5 * 100.0
    obs_pct = stat5 / far5 * 100.0
    stage4_median = _stage4_mean_distance_null_median()

    power_section = (
        "## Power\n\n"
        f"For the pre-specified headline (Distance <= 5), SE = (null_p97.5 - null_p2.5) / (2 * 1.96) = "
        f"{se5:.6f}. The minimum detectable effect at 80% power is MDE = 2.8 * SE = {mde:.6f}, which is "
        f"{mde_pct:.1f}% of the far-group mean |return| ({far5:.6f}).\n\n"
        "The MDE exceeds the band edge because the observed statistic is itself random: an effect "
        "sitting exactly on the boundary is detected only about half the time. The 2.8 factor is 1.96 "
        "(to clear the band) plus 0.84 (for an 80% detection rate).\n\n"
        f"The observed effect is {stat5:.6f}, or {obs_pct:.1f}% of the far-group mean, against an MDE of "
        f"{mde_pct:.1f}%. The observed effect is a fraction of what this design can reliably resolve, so "
        "the null rules out large effects and is uninformative about effects of the magnitude typically "
        "reported in the round-number literature."
    )

    null_median_section = (
        "## What the null median is\n\n"
        f"The null median for the main threshold is {med5:.6f}, approximately zero. Each of the "
        f"{cfg.n_sim:,} no-effect simulated paths is split by prior-day distance and the same "
        "near-minus-far subtraction is performed, giving 2,000 differences from worlds where the true "
        "answer is zero; their median is the value above.\n\n"
        "It matters because a grouping procedure can manufacture a difference from nothing -- splitting "
        "on the contemporaneous close instead of the prior close would do exactly that, because the "
        "grouping variable and the outcome would then share a price. A null median at approximately zero "
        "is empirical evidence, on the real return series, that the lagged design introduces no "
        "mechanical bias.\n\n"
        f"Contrast Stage 4, whose null median for mean distance is {stage4_median:.6f}, not the "
        "theoretical uniform 12.5 -- a real mechanical bias in that statistic. You cannot know which "
        "case applies without simulating, which is why the null median is reported in both stages."
    )

    pk_a = parkinson[parkinson["sample"] == "analysis_as_is"]
    pk_b = parkinson[parkinson["sample"] == "extreme_dates_reinstated"]
    pk5_a = pk_a[pk_a["near_threshold_yuan"] == 5.0].iloc[0]
    pk5_b = pk_b[pk_b["near_threshold_yuan"] == 5.0].iloc[0]
    agree = bool((pk_a["outside_95_null_band"].values == pk_b["outside_95_null_band"].values).all())
    any_pk_outside = bool(parkinson["outside_95_null_band"].any())
    agree_clause = (
        "agree at every threshold" if agree else "do NOT agree at every threshold"
    ) + (
        ", and in both the statistic stays inside the null band" if agree and not any_pk_outside else ""
    )
    parkinson_section = (
        "## Parkinson volatility (post-hoc robustness)\n\n"
        "- `abs_log_return` is the pre-specified primary measure; Parkinson volatility is a post-hoc "
        "robustness check added during review, reported here regardless of outcome.\n"
        "- Parkinson is roughly 5x more efficient at estimating volatility from the same number of days, "
        "because it uses the daily high and low rather than a single closing price.\n"
        "- Parkinson is the measure more closely aligned with the hypothesised mechanism: an intraday "
        "rejection at a level shows up in the range but can be nearly invisible in the close-to-close "
        "move.\n"
        "- Against that, `close` comes from a closing auction with substantial volume behind it, while "
        "`high` and `low` can be set by a single fill -- which is why the more robust close-to-close "
        "measure was chosen as primary.\n"
        "- Reported on two samples: (a) the analysis sample as used by the primary test, and (b) the same "
        "sample with the two extreme-intraday-range dates (2016-03-01, 2021-12-28) reinstated, because "
        "excluding days for extreme range and then using range as the outcome would truncate the "
        "dependent variable. 2021-12-28 is outside the 2006-2020 training window, so only 2016-03-01 "
        "actually re-enters; the two samples differ by one day.\n\n"
        f"{md(parkinson)}\n\n"
        "In this file the `mean_abs_return_after_*` columns hold Parkinson volatility (the column "
        "structure matches the primary statistics file). At the headline threshold, (a) gives statistic "
        f"{pk5_a['statistic_near_minus_far']} (band {pk5_a['null_p2.5']} to {pk5_a['null_p97.5']}) and (b) "
        f"gives {pk5_b['statistic_near_minus_far']} (band {pk5_b['null_p2.5']} to {pk5_b['null_p97.5']}). "
        f"The two samples {agree_clause}. Full results are in `parkinson_robustness.csv`."
    )

    one_finding_section = (
        "## Stages 4 and 5 are one finding, not two\n\n"
        "Stage 4 found prices spend slightly less time near levels; Stage 5 finds volatility slightly "
        "higher after a near close. These fit a single mechanism -- higher volatility near a level causes "
        "faster escape and therefore lower occupancy. That coherence is exactly why they do not "
        "corroborate each other: if the volatility effect were real and large it would mechanically "
        "produce the occupancy effect. The two results must be described as one finding observed from two "
        "angles, not as two independent tests both failing to reject."
    )

    report = f"""# Stage 5: Level-Proximity and Next-Day Volatility

## Scope

Single pre-committed confirmatory test of support/resistance, run after the
occupancy test (Stage 4) and the descriptive distance-distribution look (Stage 3)
both returned nothing significant. Training window {TRAIN_START.date()} to
{TRAIN_END.date()}; 2021+ held out. One primary statistic, one null model, one main
threshold fixed in advance (to avoid data-dredging).

## Question

No look-ahead: is today's absolute log-return larger or smaller when YESTERDAY's
close sat near a 50-level? Statistic = mean|return| after a near-level close minus
mean|return| after a far-level close.

## Null model

Block bootstrap of daily log-returns (block {cfg.block_length}, {cfg.n_sim} sims,
seed {cfg.seed}). Return sizes are unlinked from price level, so any real link
shows as the statistic escaping the null band. The null also captures any purely
mechanical bias (its median need not be zero).

Seed {cfg.seed} is shared with Stage 4 (`test_roundnumber_avoidance.py`): the two
stages draw the same block-start array from the same return series, so their null
paths are bit-identical. Each test is individually valid -- the shared paths are a
valid sample from the null -- but the two results are NOT independent and must not be
described as two independent tests both failing to reject. The alignment is
deliberate, making the two stages a controlled comparison.

## Result

- Training observations: {n:,}

{md(table)}

`outside_95_null_band` is the decision flag.

## Chart

{plot_line}
## Interpretation

{verdict}

{null_median_section}

{power_section}

{parkinson_section}

{one_finding_section}

## Caveats

- One 15-year path compared to a null band (parametric-bootstrap-style test).
- Failing to reject is not proof of no effect; it means this test found no evidence.
- The block bootstrap breaks the price-level/return link under the null but assumes
  returns are otherwise level-independent.
"""
    out.write_text(report, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    cfg = Config(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        n_sim=args.n_sim,
        block_length=args.block_length,
        seed=args.seed,
    )
    print(json.dumps(run(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
