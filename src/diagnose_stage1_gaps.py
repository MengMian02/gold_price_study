"""Phase 1 diagnostics for the Stage 1 data audit.

This is a standalone, read-only diagnostic. It reads ``data/raw/Au9999.csv`` and
reports facts needed to decide the later Stage 1 fixes. It does not modify the raw
CSV, does not change the pipeline, and applies no thresholds or flags of its own
(the intraday-range section deliberately reports the raw distribution only).

Outputs: prints a markdown report to stdout and writes it to
``outputs/stage1_data_audit/diagnostics.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume_kg"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume_kg"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "Au9999.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "stage1_data_audit"


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path without exposing local folders."""
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name

# Calendar-day gap size above which a consecutive-row gap is listed in detail.
# This is a reporting cut-off only (it does not flag or exclude anything); a
# normal Friday->Monday weekend is 3 calendar days, so >4 surfaces gaps that
# skip at least one weekday.
GAP_LISTING_MIN_CALENDAR_DAYS = 4


@dataclass
class DiagnosticsConfig:
    source: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    gap_listing_min_calendar_days: int = GAP_LISTING_MIN_CALENDAR_DAYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 diagnostics for the Stage 1 audit")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="CSV path. Read only.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gap-listing-min-calendar-days", type=int, default=GAP_LISTING_MIN_CALENDAR_DAYS)
    return parser.parse_args()


def load(source: Path) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"Source CSV not found: {source}")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    df = raw.copy()
    # Original file position: header is line 1, so the first data row is line 2.
    df["row_number"] = np.arange(2, len(df) + 2)
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for col in NUMERIC_COLUMNS:
        df[f"{col}_num"] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
    return df


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_none_"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def fmt_date(value: object) -> str:
    if pd.isna(value):
        return "NaT"
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def fmt_num(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def section_duplicates(df: pd.DataFrame) -> str:
    lines = ["## 1a. Duplicate dates", ""]
    parse_failures = int(df["parsed_date"].isna().sum())
    dup_mask = df["parsed_date"].duplicated(keep=False) & df["parsed_date"].notna()
    dup_dates = sorted(df.loc[dup_mask, "parsed_date"].unique())
    lines.append(f"- Unparseable date rows (excluded from duplicate check): {parse_failures}")
    lines.append(f"- Distinct duplicated dates: {len(dup_dates)}")
    lines.append(f"- Rows involved in a duplicate: {int(dup_mask.sum())}")
    lines.append("")
    if dup_dates:
        lines.append("Duplicated date values: " + ", ".join(fmt_date(d) for d in dup_dates))
        lines.append("")
        rows = []
        for d in dup_dates:
            for _, r in df[df["parsed_date"] == d].iterrows():
                rows.append([
                    r["row_number"], r["date"], r["open"], r["high"],
                    r["low"], r["close"], r["volume_kg"],
                ])
        lines.append(md_table(
            ["row_number", "date", "open", "high", "low", "close", "volume_kg"], rows
        ))
    else:
        lines.append("_No duplicate dates found._")
    return "\n".join(lines)


def section_gaps(df: pd.DataFrame, config: DiagnosticsConfig) -> str:
    lines = ["## 1b. Calendar gaps", ""]
    valid = df[df["parsed_date"].notna()].reset_index(drop=True)
    dates = valid["parsed_date"]

    calendar_diff = dates.diff().dt.days  # difference vs previous row, in calendar days

    # Intervening weekdays between consecutive valid-date rows.
    # np.busday_count(a, b) counts Mon-Fri in [a, b); subtracting 1 gives the
    # weekdays strictly between the two dates = potentially missing trading days.
    d_days = dates.to_numpy().astype("datetime64[D]")
    prev_d, next_d = d_days[:-1], d_days[1:]
    busday = np.busday_count(prev_d, next_d)
    intervening = np.full(len(valid), np.nan)
    intervening[1:] = busday - 1
    valid["calendar_diff"] = calendar_diff
    valid["intervening_weekdays"] = intervening

    # Distribution of calendar-day differences (value counts).
    diff_counts = calendar_diff.dropna().astype(int).value_counts().sort_index()
    lines.append("### Distribution of calendar-day differences between consecutive rows")
    lines.append("")
    lines.append(md_table(
        ["calendar_day_diff", "count"],
        [[int(k), int(v)] for k, v in diff_counts.items()],
    ))
    lines.append("")

    # Gaps exceeding the listing cut-off.
    gap_mask = valid["calendar_diff"] > config.gap_listing_min_calendar_days
    lines.append(f"### Gaps exceeding {config.gap_listing_min_calendar_days} calendar days")
    lines.append("")
    gap_rows = []
    for i in valid.index[gap_mask]:
        prev = valid.loc[i - 1]
        cur = valid.loc[i]
        gap_rows.append([
            fmt_date(prev["parsed_date"]), fmt_date(cur["parsed_date"]),
            int(cur["calendar_diff"]), int(cur["intervening_weekdays"]),
        ])
    lines.append(md_table(
        ["prev_date", "next_date", "gap_calendar_days", "intervening_weekdays"], gap_rows
    ))
    lines.append("")

    # Summary over ALL consecutive pairs: a "weekday-gap" is any pair with >=1
    # intervening weekday (i.e. at least one weekday skipped).
    weekday_gap_mask = valid["intervening_weekdays"] >= 1
    total_weekday_gaps = int(weekday_gap_mask.sum())
    total_unexplained_weekdays = int(valid.loc[weekday_gap_mask, "intervening_weekdays"].sum())
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- Total weekday-gaps (consecutive rows skipping >=1 weekday): {total_weekday_gaps}")
    lines.append(f"- Total unexplained weekdays (summed intervening weekdays): {total_unexplained_weekdays}")
    lines.append(f"- Weekday-gaps larger than 10 intervening weekdays: "
                 f"{int((valid['intervening_weekdays'] > 10).sum())}")
    return "\n".join(lines)


def section_zero_volume(df: pd.DataFrame) -> str:
    lines = ["## 1c. Zero-volume days", ""]
    close = df["close_num"]
    prev_close = close.shift(1)
    next_close = close.shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_return_day = np.log(close / prev_close)
        log_return_next = np.log(next_close / close)

    vol = df["volume_kg_num"]
    zero_or_missing = vol.isna() | (vol == 0)
    lines.append(f"- Rows with volume 0 or missing: {int(zero_or_missing.sum())}")
    lines.append("")
    if not zero_or_missing.any():
        lines.append("_No zero-volume or missing-volume rows found._")
        return "\n".join(lines)

    rows = []
    for i in df.index[zero_or_missing]:
        prev_c = prev_close.iloc[i]
        stale = bool(pd.notna(prev_c) and df["close_num"].iloc[i] == prev_c)
        rows.append([
            df["row_number"].iloc[i], fmt_date(df["parsed_date"].iloc[i]),
            df["volume_kg"].iloc[i] if df["volume_kg"].iloc[i] != "" else "(missing)",
            fmt_num(df["close_num"].iloc[i]), fmt_num(prev_c),
            "yes" if stale else "no",
            fmt_num(log_return_day.iloc[i]), fmt_num(log_return_next.iloc[i]),
        ])
    lines.append(md_table(
        ["row_number", "date", "volume", "close", "prev_close",
         "close==prev_close", "log_return_day", "log_return_next_day"],
        rows,
    ))
    return "\n".join(lines)


def section_sort_order(df: pd.DataFrame) -> str:
    lines = ["## 1d. Sort order", ""]
    valid_dates = df["parsed_date"].dropna()
    monotonic = bool(valid_dates.is_monotonic_increasing)
    lines.append(f"- `parsed_date` monotonically increasing in file order: {monotonic}")
    if not monotonic:
        backward = df["parsed_date"].notna() & (df["parsed_date"].diff() < pd.Timedelta(0))
        rows = [[df["row_number"].iloc[i], fmt_date(df["parsed_date"].iloc[i]),
                 fmt_date(df["parsed_date"].iloc[i - 1])] for i in df.index[backward]]
        lines.append("")
        lines.append(md_table(["row_number", "date", "previous_row_date"], rows))
    return "\n".join(lines)


def section_intraday_range(df: pd.DataFrame) -> str:
    lines = ["## 1e. Intraday range distribution", "",
             "Metric: `(high - low) / close`. No threshold is applied here.", ""]
    valid = (df["high_num"] > 0) & (df["low_num"] > 0) & (df["close_num"] > 0)
    rng = ((df.loc[valid, "high_num"] - df.loc[valid, "low_num"]) / df.loc[valid, "close_num"]).dropna()
    desc = rng.describe()
    lines.append(md_table(
        ["statistic", "value"],
        [[k, fmt_num(v)] for k, v in desc.items()],
    ))
    lines.append("")
    quantiles = [0.90, 0.95, 0.99, 0.995, 0.999]
    lines.append(md_table(
        ["percentile", "value"],
        [[f"{q:.1%}", fmt_num(rng.quantile(q))] for q in quantiles],
    ))
    lines.append("")
    lines.append(f"- Valid observations: {int(rng.count())}")
    lines.append(f"- Max `(high-low)/close`: {fmt_num(rng.max())} "
                 f"(on {fmt_date(df.loc[rng.idxmax(), 'parsed_date'])})")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, config: DiagnosticsConfig) -> str:
    header = [
        "# Stage 1 Diagnostics (Phase 1)",
        "",
        f"- Source file: `{repository_relative(config.source)}`",
        f"- Rows (excluding header): {len(df)}",
        f"- Date range: {fmt_date(df['parsed_date'].min())} to {fmt_date(df['parsed_date'].max())}",
        "",
        "This report is diagnostic only. It applies no thresholds and flags nothing; "
        "it exists to inform the Stage 1 fixes.",
        "",
    ]
    sections = [
        section_duplicates(df),
        section_gaps(df, config),
        section_zero_volume(df),
        section_sort_order(df),
        section_intraday_range(df),
    ]
    return "\n".join(header) + "\n" + "\n\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()
    config = DiagnosticsConfig(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        gap_listing_min_calendar_days=args.gap_listing_min_calendar_days,
    )
    df = load(config.source)
    report = build_report(df, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / "diagnostics.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
