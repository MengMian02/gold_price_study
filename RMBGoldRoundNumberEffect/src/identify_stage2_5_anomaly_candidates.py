"""Stage 2.5 anomaly-candidate identification from processed variable data.

This script reads only the Stage 2 processed dataset. It does not read raw data,
clean values, overwrite existing files, compare Near/Not Near groups, run
statistical tests, or access external sources.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = WORKSPACE_ROOT / "outputs" / "stage2_variable_construction" / "au9999_research_variables_stage2.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "stage2_5_anomaly_candidates"

REQUIRED_COLUMNS = [
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume_kg_raw",
    "log_return",
    "abs_log_return",
    "parkinson_volatility",
    "abnormal_volume_20d",
]
KNOWN_DATES = ["2016-03-01", "2021-12-28", "2012-01-03", "2013-01-03"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify Stage 2.5 manual-review anomaly candidates")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(Path(args.input), Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run(input_path: Path, output_dir: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed Stage 2 data not found: {input_path}")
    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.sort_values(["trade_date", "source_row_number"], kind="mergesort").reset_index(drop=True)

    add_neighbors(df)
    add_auxiliary_metrics(df)
    thresholds = compute_thresholds(df)
    flags = build_flags(df, thresholds)
    candidates = build_candidate_table(df, flags)
    report = build_report(df, candidates, thresholds, input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    script_copy = output_dir / "identify_stage2_5_anomaly_candidates.py"
    candidate_path = output_dir / "stage2_5_anomaly_candidates.csv"
    report_path = output_dir / "stage2_5_anomaly_candidate_report.md"
    audit_copy_path = output_dir / "stage2_5_flagged_audit_copy.csv"

    shutil.copy2(Path(__file__), script_copy)
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    df.to_csv(audit_copy_path, index=False, encoding="utf-8-sig")
    report_path.write_text(report, encoding="utf-8")

    return {
        "input_file": str(input_path),
        "rows_read": int(len(df)),
        "candidate_rows": int(len(candidates)),
        "script": str(script_copy),
        "candidate_table": str(candidate_path),
        "flagged_audit_copy": str(audit_copy_path),
        "report": str(report_path),
    }


def add_neighbors(df: pd.DataFrame) -> None:
    base = ["trade_date", "open_price", "high_price", "low_price", "close_price", "volume_kg_raw"]
    for col in base:
        df[f"prev_{col}"] = df[col].shift(1)
        df[f"next_{col}"] = df[col].shift(-1)
    for k in range(1, 6):
        df[f"volume_lag{k}"] = df["volume_kg_raw"].shift(k)
        df[f"volume_lead{k}"] = df["volume_kg_raw"].shift(-k)
        df[f"date_lag{k}"] = df["trade_date"].shift(k)
        df[f"date_lead{k}"] = df["trade_date"].shift(-k)


def add_auxiliary_metrics(df: pd.DataFrame) -> None:
    valid_hl = (df["high_price"] > 0) & (df["low_price"] > 0)
    df["log_range"] = np.nan
    df.loc[valid_hl, "log_range"] = np.log(df.loc[valid_hl, "high_price"] / df.loc[valid_hl, "low_price"])
    df["relative_range"] = (df["high_price"] - df["low_price"]) / df["close_price"]
    df["open_gap"] = np.log(df["open_price"] / df["prev_close_price"])
    df["abs_open_gap"] = df["open_gap"].abs()

    for col in ["open_price", "high_price", "low_price", "close_price"]:
        df[f"{col}_log_change"] = np.log(df[col] / df[f"prev_{col}"])
        df[f"abs_{col}_log_change"] = df[f"{col}_log_change"].abs()

    intraday_anchor_low = pd.concat(
        [df["open_price"], df["close_price"], df["prev_close_price"], df["next_close_price"]], axis=1
    ).median(axis=1, skipna=True)
    intraday_anchor_high = intraday_anchor_low
    df["low_anchor_deviation"] = np.log(intraday_anchor_low / df["low_price"])
    df["high_anchor_deviation"] = np.log(df["high_price"] / intraday_anchor_high)

    neighbor_close_median = pd.concat([df["prev_close_price"], df["next_close_price"]], axis=1).median(axis=1)
    df["low_neighbor_deviation"] = np.log(neighbor_close_median / df["low_price"])
    df["high_neighbor_deviation"] = np.log(df["high_price"] / neighbor_close_median)

    prev60 = df["volume_kg_raw"].shift(1).rolling(60, min_periods=40).median()
    next60 = df["volume_kg_raw"].shift(-60).rolling(60, min_periods=40).median()
    df["volume_prev60_median"] = prev60
    df["volume_next60_median"] = next60
    df["volume_regime_ratio_next60_to_prev60"] = next60 / prev60
    df["volume_prev_next_median"] = pd.concat([df["prev_volume_kg_raw"], df["next_volume_kg_raw"]], axis=1).median(axis=1)
    df["volume_vs_neighbor_median_ratio"] = df["volume_kg_raw"] / df["volume_prev_next_median"]


def q(s: pd.Series, p: float) -> float:
    return float(s.dropna().quantile(p))


def compute_thresholds(df: pd.DataFrame) -> dict:
    positive_volume = df.loc[df["volume_kg_raw"] > 0, "volume_kg_raw"]
    thresholds = {
        "log_range_p99": q(df["log_range"], 0.99),
        "relative_range_p99": q(df["relative_range"], 0.99),
        "parkinson_volatility_p99": q(df["parkinson_volatility"], 0.99),
        "abs_log_return_p99": q(df["abs_log_return"], 0.99),
        "abs_open_gap_p99": q(df["abs_open_gap"], 0.99),
        "volume_positive_p01": q(positive_volume, 0.01),
        "volume_positive_p99": q(positive_volume, 0.99),
        "abnormal_volume_p01": q(df["abnormal_volume_20d"], 0.01),
        "abnormal_volume_p99": q(df["abnormal_volume_20d"], 0.99),
        "low_anchor_deviation_p99": q(df["low_anchor_deviation"], 0.99),
        "high_anchor_deviation_p99": q(df["high_anchor_deviation"], 0.99),
        "low_neighbor_deviation_p99": q(df["low_neighbor_deviation"], 0.99),
        "high_neighbor_deviation_p99": q(df["high_neighbor_deviation"], 0.99),
    }
    for col in ["open_price", "high_price", "low_price", "close_price"]:
        thresholds[f"abs_{col}_log_change_p99"] = q(df[f"abs_{col}_log_change"], 0.99)
    return thresholds


def build_flags(df: pd.DataFrame, t: dict) -> dict[str, pd.Series]:
    flags: dict[str, pd.Series] = {}
    flags["known_issue_date"] = df["trade_date"].dt.strftime("%Y-%m-%d").isin(KNOWN_DATES)
    flags["nonpositive_ohlc"] = (df[["open_price", "high_price", "low_price", "close_price"]] <= 0).any(axis=1)
    flags["high_below_open_close"] = df["high_price"] < df[["open_price", "close_price"]].max(axis=1)
    flags["low_above_open_close"] = df["low_price"] > df[["open_price", "close_price"]].min(axis=1)
    flags["ohlc_logic_error"] = flags["nonpositive_ohlc"] | flags["high_below_open_close"] | flags["low_above_open_close"]

    flags["log_range_top_1pct"] = df["log_range"] >= t["log_range_p99"]
    flags["relative_range_top_1pct"] = df["relative_range"] >= t["relative_range_p99"]
    flags["parkinson_volatility_top_1pct"] = df["parkinson_volatility"] >= t["parkinson_volatility_p99"]
    flags["low_far_below_open_close"] = df["low_anchor_deviation"] >= t["low_anchor_deviation_p99"]
    flags["high_far_above_open_close"] = df["high_anchor_deviation"] >= t["high_anchor_deviation_p99"]
    flags["low_far_below_neighbor_days"] = df["low_neighbor_deviation"] >= t["low_neighbor_deviation_p99"]
    flags["high_far_above_neighbor_days"] = df["high_neighbor_deviation"] >= t["high_neighbor_deviation_p99"]

    flags["abs_log_return_gt_0p2"] = df["abs_log_return"] > 0.2
    flags["abs_log_return_top_1pct"] = df["abs_log_return"] >= t["abs_log_return_p99"]
    flags["abs_open_gap_top_1pct"] = df["abs_open_gap"] >= t["abs_open_gap_p99"]
    for col in ["open_price", "high_price", "low_price", "close_price"]:
        flags[f"{col}_change_top_1pct"] = df[f"abs_{col}_log_change"] >= t[f"abs_{col}_log_change_p99"]

    flags["volume_missing_zero_negative"] = df["volume_kg_raw"].isna() | (df["volume_kg_raw"] <= 0)
    flags["volume_fixed_10_or_20"] = df["volume_kg_raw"].isin([10, 20])
    flags["volume_lowest_1pct_positive"] = (df["volume_kg_raw"] > 0) & (df["volume_kg_raw"] <= t["volume_positive_p01"])
    flags["volume_highest_1pct"] = df["volume_kg_raw"] >= t["volume_positive_p99"]
    flags["abnormal_volume_lowest_1pct"] = df["abnormal_volume_20d"] <= t["abnormal_volume_p01"]
    flags["abnormal_volume_highest_1pct"] = df["abnormal_volume_20d"] >= t["abnormal_volume_p99"]
    flags["volume_neighbor_collapse"] = df["volume_vs_neighbor_median_ratio"] <= 0.1
    flags["volume_neighbor_surge"] = df["volume_vs_neighbor_median_ratio"] >= 10
    flags["possible_volume_regime_change"] = (
        (df["volume_regime_ratio_next60_to_prev60"] >= 5)
        | (df["volume_regime_ratio_next60_to_prev60"] <= 0.2)
    )
    return flags


def classify_types(row_flags: list[str]) -> list[str]:
    types = []
    if any(x in row_flags for x in ["ohlc_logic_error", "nonpositive_ohlc", "high_below_open_close", "low_above_open_close"]):
        types.append("ohlc_logic_error")
    if any(x in row_flags for x in ["log_range_top_1pct", "relative_range_top_1pct", "parkinson_volatility_top_1pct"]):
        types.append("extreme_intraday_range")
    if any(x in row_flags for x in ["low_far_below_open_close", "high_far_above_open_close", "low_far_below_neighbor_days", "high_far_above_neighbor_days"]):
        types.append("isolated_high_low_spike")
    if any(x in row_flags for x in ["abs_log_return_gt_0p2", "abs_log_return_top_1pct", "abs_open_gap_top_1pct", "open_price_change_top_1pct", "high_price_change_top_1pct", "low_price_change_top_1pct", "close_price_change_top_1pct"]):
        types.append("extreme_return")
    if any(x in row_flags for x in ["volume_highest_1pct", "abnormal_volume_highest_1pct", "volume_neighbor_surge"]):
        types.append("extreme_volume")
    if any(x in row_flags for x in ["volume_missing_zero_negative", "volume_fixed_10_or_20", "volume_lowest_1pct_positive", "abnormal_volume_lowest_1pct", "volume_neighbor_collapse"]):
        types.append("volume_collapse")
    if "possible_volume_regime_change" in row_flags:
        types.append("possible_volume_regime_change")
    if not types:
        types.append("other")
    return sorted(set(types))


def severity(row_flags: list[str], types: list[str]) -> str:
    high_rules = {
        "known_issue_date",
        "ohlc_logic_error",
        "abs_log_return_gt_0p2",
        "low_far_below_open_close",
        "high_far_above_open_close",
        "low_far_below_neighbor_days",
        "high_far_above_neighbor_days",
        "volume_fixed_10_or_20",
        "volume_missing_zero_negative",
        "volume_neighbor_collapse",
        "volume_neighbor_surge",
    }
    if high_rules.intersection(row_flags):
        return "high"
    if len(row_flags) >= 2 or len(types) >= 2:
        return "medium"
    return "low"


def volume_context(row: pd.Series) -> str:
    parts = []
    for k in range(5, 0, -1):
        d = row.get(f"date_lag{k}")
        v = row.get(f"volume_lag{k}")
        if pd.notna(d):
            parts.append(f"{pd.to_datetime(d).date()}:{v}")
    parts.append(f"{pd.to_datetime(row['trade_date']).date()}:{row['volume_kg_raw']}")
    for k in range(1, 6):
        d = row.get(f"date_lead{k}")
        v = row.get(f"volume_lead{k}")
        if pd.notna(d):
            parts.append(f"{pd.to_datetime(d).date()}:{v}")
    return " | ".join(parts)


def explanation(row_flags: list[str], types: list[str]) -> str:
    if "isolated_high_low_spike" in types:
        return "High/Low is unusually far from same-day open/close or adjacent trading days; resembles isolated field spike."
    if "volume_collapse" in types:
        return "Volume is extremely low or collapsed relative to nearby/history levels; manual review needed."
    if "extreme_return" in types:
        return "Large cross-day price movement; distinguish real market move from data break."
    if "possible_volume_regime_change" in types:
        return "Rolling 60-day volume median changes sharply; possible unit or coverage shift."
    if "extreme_intraday_range" in types:
        return "Intraday range is in the top 1% by at least one range metric."
    return "Flagged by one or more screening rules; pending manual review."


def build_candidate_table(df: pd.DataFrame, flags: dict[str, pd.Series]) -> pd.DataFrame:
    flag_df = pd.DataFrame(flags)
    candidate_mask = flag_df.any(axis=1)
    rows = []
    for idx, row in df.loc[candidate_mask].iterrows():
        triggered = [name for name, val in flag_df.loc[idx].items() if bool(val)]
        types = classify_types(triggered)
        rows.append(
            {
                "date": row["trade_date"].strftime("%Y-%m-%d"),
                "open": row["open_price"],
                "high": row["high_price"],
                "low": row["low_price"],
                "close": row["close_price"],
                "volume": row["volume_kg_raw"],
                "log_return": row["log_return"],
                "abs_log_return": row["abs_log_return"],
                "log_range": row["log_range"],
                "relative_range": row["relative_range"],
                "parkinson_volatility": row["parkinson_volatility"],
                "abnormal_volume_20d": row["abnormal_volume_20d"],
                "prev_date": date_or_blank(row["prev_trade_date"]),
                "prev_open": row["prev_open_price"],
                "prev_high": row["prev_high_price"],
                "prev_low": row["prev_low_price"],
                "prev_close": row["prev_close_price"],
                "prev_volume": row["prev_volume_kg_raw"],
                "next_date": date_or_blank(row["next_trade_date"]),
                "next_open": row["next_open_price"],
                "next_high": row["next_high_price"],
                "next_low": row["next_low_price"],
                "next_close": row["next_close_price"],
                "next_volume": row["next_volume_kg_raw"],
                "volume_context_pm5": volume_context(row),
                "triggered_rules": ";".join(triggered),
                "anomaly_type": ";".join(types),
                "severity": severity(triggered, types),
                "preliminary_explanation": explanation(triggered, types),
                "manual_review_status": "pending_review",
                "manual_review_note": "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sev_order = {"high": 0, "medium": 1, "low": 2}
    out["_sev"] = out["severity"].map(sev_order)
    out["_rule_count"] = out["triggered_rules"].str.count(";") + 1
    out = out.sort_values(["_sev", "_rule_count", "date"], ascending=[True, False, True]).drop(columns=["_sev", "_rule_count"])
    return out


def date_or_blank(value) -> str:
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def build_report(df: pd.DataFrame, candidates: pd.DataFrame, thresholds: dict, input_path: Path) -> str:
    type_counts = explode_counts(candidates, "anomaly_type")
    severity_counts = candidates["severity"].value_counts().rename_axis("severity").reset_index(name="count")
    known = candidates[candidates["date"].isin(KNOWN_DATES)][["date", "triggered_rules", "anomaly_type", "severity"]]
    lr20 = candidates[candidates["triggered_rules"].str.contains("abs_log_return_gt_0p2", na=False)][
        ["date", "open", "high", "low", "close", "volume", "log_return", "triggered_rules", "severity"]
    ]
    severe = candidates.head(20)
    threshold_rows = pd.DataFrame([{"metric": k, "threshold": v} for k, v in thresholds.items()])
    multi = candidates[candidates["triggered_rules"].str.contains(";", na=False)]
    priority = candidates[candidates["severity"].eq("high")].head(20)
    return f"""# Stage 2.5 Anomaly Candidate Report

## Scope

This stage reads only the processed Stage 2 variable dataset. It does not read raw data, access external sources, modify values, delete records, create a cleaned dataset, compare Near and Not Near groups, run tests, or run regressions.

## Input

- Processed data file: `{input_path}`
- Total rows: {len(df)}
- Frequency: daily trading records
- Fields used: `trade_date`, `open_price`, `high_price`, `low_price`, `close_price`, `volume_kg_raw`, `log_return`, `abs_log_return`, `parkinson_volatility`, `abnormal_volume_20d`

## Summary

- Candidate records flagged: {len(candidates)}
- Records triggering multiple rules: {len(multi)}
- Manual review status for all candidates: `pending_review`

## Counts by Anomaly Type

{md(type_counts)}

## Counts by Severity

{md(severity_counts)}

## Known Issue Dates

{md(known)}

## `abs(log_return) > 0.2` Records

{md(lr20)}

## Most Severe Candidate Records

{md(severe)}

## Thresholds and Rules

Rules used:

- OHLC positivity and basic logic: `High >= max(Open, Close)`, `Low <= min(Open, Close)`.
- Intraday range: `log_range`, `relative_range`, and `parkinson_volatility` in the top 1% of the full processed sample.
- Isolated high/low spike: high/low deviation from same-day open/close anchor or adjacent-day close anchor in the top 1%.
- Cross-day price: `abs(log_return) > 0.2`, `abs_log_return` top 1%, `abs_open_gap` top 1%, and individual OHLC field changes top 1%.
- Volume: missing/zero/negative, fixed 10 or 20, positive volume bottom/top 1%, abnormal volume bottom/top 1%, neighbor collapse/surge by a 0.1x/10x ratio, and possible 60-day median regime shift by a 0.2x/5x ratio.

Actual numeric thresholds:

{md(threshold_rows)}

## Priority Manual Review

{md(priority)}
"""


def explode_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[col, "count"])
    return df[col].str.split(";").explode().value_counts().rename_axis(col).reset_index(name="count")


def md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.copy().replace({np.nan: ""})
    cols = list(display.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
