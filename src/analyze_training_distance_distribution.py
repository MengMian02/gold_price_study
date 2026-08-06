from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal runtimes.
    plt = None

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "outputs" / "stage2_variable_construction" / (
    "au9999_analysis_dataset.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_distance_distribution"

TRAIN_START = pd.Timestamp("2006-01-01")
TRAIN_END = pd.Timestamp("2020-12-31")
LEVEL_STEP = 50.0
NEAR_THRESHOLDS = [2.0, 3.0, 5.0, 10.0]


def load_training_sample() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    required = {"trade_date", "distance_to_level"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["distance_to_level"] = pd.to_numeric(df["distance_to_level"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)

    train = df[(df["trade_date"] >= TRAIN_START) & (df["trade_date"] <= TRAIN_END)].copy()
    # Read the distance-to-level column written by Stage 2 rather than recomputing it.
    train["distance_recomputed"] = train["distance_to_level"]
    return train


def make_description(train: pd.DataFrame) -> pd.DataFrame:
    distance = train["distance_recomputed"].dropna()
    stats = {
        "count": distance.count(),
        "mean": distance.mean(),
        "std": distance.std(ddof=1),
        "min": distance.min(),
        "25%": distance.quantile(0.25),
        "50%": distance.quantile(0.50),
        "75%": distance.quantile(0.75),
        "max": distance.max(),
    }
    return pd.DataFrame({"metric": list(stats.keys()), "value": list(stats.values())})


def make_one_yuan_bins(train: pd.DataFrame) -> pd.DataFrame:
    distance = train["distance_recomputed"]
    bins = list(np.arange(0, 26, 1))
    labels = [f"[{i},{i + 1})" for i in range(25)]
    out = pd.cut(distance, bins=bins, right=False, include_lowest=True, labels=labels)

    # The theoretical maximum 25 is included in the final interval for completeness.
    out = out.astype("object")
    out.loc[distance == 25] = "[24,25]"

    counts = out.value_counts().reindex(labels, fill_value=0)
    total = int(distance.notna().sum())
    return pd.DataFrame(
        {
            "distance_bin_yuan": labels,
            "observation_count": counts.values,
            "proportion": counts.values / total if total else np.nan,
        }
    )


def make_threshold_summary(train: pd.DataFrame) -> pd.DataFrame:
    distance = train["distance_recomputed"]
    total = int(distance.notna().sum())
    rows = []
    for threshold in NEAR_THRESHOLDS:
        count = int((distance <= threshold).sum())
        expected = threshold / 25.0
        rows.append(
            {
                "threshold_yuan": threshold,
                "observation_count": count,
                "actual_proportion": count / total if total else np.nan,
                "uniform_reference_proportion": expected,
                "actual_minus_uniform_reference": (count / total - expected) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_histogram(train: pd.DataFrame) -> Path:
    out = OUTPUT_DIR / "distance_histogram.png"
    if plt is None:
        save_histogram_with_pillow(train, out)
        return out

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(
        train["distance_recomputed"].dropna(),
        bins=np.arange(0, 26, 1),
        color="#4C78A8",
        edgecolor="white",
    )
    ax.set_title("Training Sample Distance to Nearest 50 RMB Level")
    ax.set_xlabel("Distance to nearest 50 RMB level (yuan)")
    ax.set_ylabel("Observation count")
    ax.set_xlim(0, 25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_histogram_with_pillow(train: pd.DataFrame, out: Path) -> None:
    width, height = 1200, 720
    margin_left, margin_right, margin_top, margin_bottom = 95, 45, 70, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(18)
    small = get_font(14)
    title = get_font(24)

    values = train["distance_recomputed"].dropna()
    counts, edges = np.histogram(values, bins=np.arange(0, 26, 1))
    ymax = max(int(counts.max()), 1)

    draw.text((margin_left, 25), "Training Sample Distance to Nearest 50 RMB Level", fill="#111111", font=title)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="#333333", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="#333333", width=2)

    for y_frac in np.linspace(0, 1, 6):
        y = margin_top + plot_h - int(y_frac * plot_h)
        val = int(round(y_frac * ymax))
        draw.line((margin_left, y, margin_left + plot_w, y), fill="#E6E6E6", width=1)
        draw.text((18, y - 8), str(val), fill="#333333", font=small)

    bar_w = plot_w / 25
    for idx, count in enumerate(counts):
        x0 = margin_left + idx * bar_w + 1
        x1 = margin_left + (idx + 1) * bar_w - 1
        y0 = margin_top + plot_h - (count / ymax) * plot_h
        draw.rectangle((x0, y0, x1, margin_top + plot_h), fill="#4C78A8")

    for tick in [0, 5, 10, 15, 20, 25]:
        x = margin_left + (tick / 25) * plot_w
        draw.line((x, margin_top + plot_h, x, margin_top + plot_h + 6), fill="#333333", width=1)
        draw.text((x - 8, margin_top + plot_h + 12), str(tick), fill="#333333", font=small)

    draw.text((width // 2 - 185, height - 45), "Distance to nearest 50 RMB level (yuan)", fill="#111111", font=font)
    draw.text((18, 36), "Count", fill="#111111", font=font)
    image.save(out)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}"


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int = 6) -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    headers = [str(col) for col in df.columns]
    rows = [[fmt(value) for value in row] for row in df.to_numpy()]
    widths = []
    for idx, header in enumerate(headers):
        max_row_width = max([len(row[idx]) for row in rows], default=0)
        widths.append(max(len(header), max_row_width))

    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def write_report(
    train: pd.DataFrame,
    description: pd.DataFrame,
    bins: pd.DataFrame,
    thresholds: pd.DataFrame,
    histogram_path: Path,
) -> Path:
    out = OUTPUT_DIR / "distance_distribution_report.md"
    valid_distance = train["distance_recomputed"].dropna()
    sample_start = train["trade_date"].min().date()
    sample_end = train["trade_date"].max().date()

    near5_row = thresholds.loc[thresholds["threshold_yuan"] == 5.0].iloc[0]
    near5_prop = near5_row["actual_proportion"]
    near5_diff = near5_row["actual_minus_uniform_reference"]

    max_distance = valid_distance.max()
    min_distance = valid_distance.min()
    range_check = "pass" if min_distance >= 0 and max_distance <= 25 else "fail"

    lines = [
        "# Training Sample Distance Distribution",
        "",
        "The uniform line is a visual benchmark, not a formal maintained null hypothesis.",
        "",
        "- The uniform reference assumes independent observations, which this series does not satisfy: the distance series is highly persistent and the nearest level changes only about 133 times across 3,664 days.",
        "- Deviations from the uniform line must not be read as evidence of a round-number effect.",
        "- The correct null is constructed in Stage 4 by moving-block bootstrap, and its centre is not the uniform value.",
        "- Stage 3 is descriptive only: it measures the distribution; Stage 4 supplies the standard of comparison.",
        "",
        "## Scope",
        "",
        f"- Input file: `{INPUT_FILE.relative_to(PROJECT_ROOT).as_posix()}`",
        "- Training sample: 2006-01-01 to 2020-12-31, inclusive.",
        "- Data from 2021-01-01 onward was not used for charts, summary statistics, threshold choice, or interpretation.",
        "- Task scope: only the distribution of closing-price distance to the nearest 50 RMB level.",
        "- No return, volatility, volume comparison, regression, or significance test was performed.",
        "",
        "## Training Sample",
        "",
        f"- Actual start date: {sample_start}",
        f"- Actual end date: {sample_end}",
        f"- Observations: {len(train):,}",
        f"- Valid Distance observations: {valid_distance.count():,}",
        f"- Theoretical Distance range check, 0 to 25 yuan: {range_check}",
        "",
        "## Distance Formula",
        "",
        "`Distance_t = abs(Close_t - 50 * round(Close_t / 50))`",
        "",
        "This script uses transparent half-up rounding: when a close is exactly at the midpoint between two 50-yuan levels, the higher level is selected.",
        "",
        "## Basic Descriptive Statistics",
        "",
        dataframe_to_markdown(description),
        "",
        "## One-Yuan Distance Bins",
        "",
        dataframe_to_markdown(bins),
        "",
        "## Near-Threshold Counts",
        "",
        dataframe_to_markdown(thresholds),
        "",
        "## Charts",
        "",
        f"![Distance histogram]({histogram_path.name})",
        "",
        "## Direct Visual Reading",
        "",
        f"- Main near definition, Distance <= 5: {int(near5_row['observation_count']):,} observations, {format_pct(near5_prop)} of the training sample.",
        f"- Uniform visual reference for Distance <= 5 is 20.00%; actual minus reference is {near5_diff:.2%}.",
        "- This is exploratory description only.",
        "- Based on this chart/table alone, do not claim a round-number effect, support/resistance, or tradable regularity.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_training_sample()

    description = make_description(train)
    bins = make_one_yuan_bins(train)
    thresholds = make_threshold_summary(train)

    train_out = OUTPUT_DIR / "training_distance_data.csv"
    description_out = OUTPUT_DIR / "distance_description.csv"
    bins_out = OUTPUT_DIR / "distance_1yuan_bins.csv"
    thresholds_out = OUTPUT_DIR / "distance_threshold_counts.csv"

    train.to_csv(train_out, index=False, encoding="utf-8-sig")
    description.to_csv(description_out, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_out, index=False, encoding="utf-8-sig")
    thresholds.to_csv(thresholds_out, index=False, encoding="utf-8-sig")

    histogram_path = save_histogram(train)
    report_path = write_report(train, description, bins, thresholds, histogram_path)

    print(f"Input: {INPUT_FILE}")
    print(f"Training rows: {len(train)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
