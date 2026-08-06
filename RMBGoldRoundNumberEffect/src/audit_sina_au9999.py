"""Stage 1 data audit for Sina Au99.99 daily OHLCV data.

This script is intentionally read-only with respect to the source CSV. It does
not clean, overwrite, filter, impute, or modify raw data; it only detects and
reports, and writes its outputs under the audit output directory. Detection and
disposition are kept strictly separate: this stage never decides what to do
about a finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume_kg"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "Au9999_Sina_Daily_OHLCV.csv",
    PROJECT_ROOT / "data" / "raw" / "Au9999.csv",
    PROJECT_ROOT / "data" / "Au9999_Sina_Daily_OHLCV.csv",
    PROJECT_ROOT / "data" / "Au9999.csv",
]


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path without exposing local folders."""
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


@dataclass
class AuditConfig:
    source: Path | None
    output_dir: Path
    price_jump_threshold: float = 0.10
    volume_jump_threshold: float = 5.0
    robust_z_threshold: float = 8.0
    # Screening threshold for daily intraday range (high-low)/close. Chosen on the
    # same principle as price_jump_threshold: a coarse cut-off that keeps the human
    # review list short by targeting order-of-magnitude anomalies, not genuinely
    # volatile sessions. 0.10 sits above the 99.5th percentile of the observed
    # range distribution (see outputs/stage1_data_audit/diagnostics.md). It is a
    # warning only and is not tuned to catch any predetermined date.
    intraday_range_threshold: float = 0.10
    # A gap that skips more intervening weekdays than this is escalated to an
    # error. 10 comfortably covers the longest Chinese exchange closures (Spring
    # Festival, National Day / Golden Week), which skip at most ~7 weekdays.
    max_expected_holiday_weekdays: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Sina Au99.99 daily OHLCV CSV")
    parser.add_argument("--source", default=None, help="CSV path. If omitted, common data filenames are searched.")
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "stage1_data_audit"))
    parser.add_argument("--price-jump-threshold", type=float, default=0.10)
    parser.add_argument("--volume-jump-threshold", type=float, default=5.0)
    parser.add_argument("--robust-z-threshold", type=float, default=8.0)
    parser.add_argument("--intraday-range-threshold", type=float, default=0.10)
    parser.add_argument("--max-expected-holiday-weekdays", type=int, default=10)
    return parser.parse_args()


def resolve_source(source_arg: str | None) -> tuple[Path, list[str]]:
    notes: list[str] = []
    if source_arg:
        source = Path(source_arg)
        if not source.exists():
            raise FileNotFoundError(f"Source CSV not found: {source}")
        return source, notes

    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            if candidate.name != "Au9999_Sina_Daily_OHLCV.csv":
                notes.append(
                    "Requested filename Au9999_Sina_Daily_OHLCV.csv was not found; "
                    f"using actual file {repository_relative(candidate)}."
                )
            return candidate, notes
    raise FileNotFoundError("No source CSV found in data/. Expected Au9999_Sina_Daily_OHLCV.csv or Au9999.csv.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_anomaly(records: list[dict], row: pd.Series, row_number: int, rule: str, severity: str, message: str) -> None:
    records.append(
        {
            "row_number_in_csv_including_header": row_number,
            "date_raw": row.get("date", ""),
            "parsed_date": row.get("parsed_date", pd.NaT),
            "rule": rule,
            "severity": severity,
            "message": message,
            "open": row.get("open", ""),
            "high": row.get("high", ""),
            "low": row.get("low", ""),
            "close": row.get("close", ""),
            "volume_kg": row.get("volume_kg", ""),
        }
    )


def robust_z(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index)
    median = clean.median()
    mad = (clean - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(np.nan, index=series.index)
    return 0.6745 * (series - median) / mad


def audit(config: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    source, source_notes = resolve_source(str(config.source) if config.source else None)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    source_sha256 = sha256_file(source)
    source_bytes = int(source.stat().st_size)

    df = raw.copy()
    # Original file position (header is line 1). Carried through the sort below so
    # every reported row number refers to the ORIGINAL file position, not the
    # sorted position.
    df["orig_row_number"] = np.arange(2, len(df) + 2)
    df["parsed_date"] = pd.to_datetime(df.get("date", pd.Series(dtype=str)), errors="coerce", dayfirst=True)
    for col in ["open", "high", "low", "close", "volume_kg"]:
        if col in df.columns:
            df[f"{col}_num"] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")

    anomalies: list[dict] = []
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in raw.columns]
    extra_columns = [c for c in raw.columns if c not in EXPECTED_COLUMNS]

    # --- Sort-order check on the ORIGINAL file order, before sorting (2a). ---
    # date_not_sorted is now an error: the third-pass pct_change / robust-z
    # statistics follow row order, so a mis-ordered file would silently corrupt
    # every jump statistic. We sort below to make those statistics correct, but a
    # mis-ordered raw file is still a hard problem to surface.
    valid_dates_original = df["parsed_date"].dropna()
    original_is_sorted = bool(valid_dates_original.is_monotonic_increasing)
    if len(valid_dates_original) > 1 and not original_is_sorted:
        out_of_order = df["parsed_date"].notna() & (df["parsed_date"].diff() < pd.Timedelta(0))
        for _, row in df.loc[out_of_order].iterrows():
            add_anomaly(
                anomalies, row, int(row["orig_row_number"]), "date_not_sorted", "error",
                "日期顺序倒退；时间序列统计已在排序后计算，但原始文件顺序有误，需修正。",
            )

    # Stable sort by parsed date (ties keep original file order) so all downstream
    # pct_change / robust-z statistics follow DATE order rather than row order.
    df = df.sort_values(["parsed_date", "orig_row_number"], kind="mergesort").reset_index(drop=True)

    # --- Duplicate dates (report original row numbers). ---
    duplicated = df["parsed_date"].duplicated(keep=False) & df["parsed_date"].notna()
    for _, row in df.loc[duplicated].iterrows():
        add_anomaly(anomalies, row, int(row["orig_row_number"]), "duplicate_date", "error", "日期重复。")

    # --- Per-row checks (order-independent). ---
    for _, row in df.iterrows():
        row_number = int(row["orig_row_number"])

        if pd.isna(row["parsed_date"]):
            add_anomaly(anomalies, row, row_number, "date_parse_failed", "error", "日期无法解析。")

        for col in EXPECTED_COLUMNS:
            if col in df.columns and str(row[col]).strip() == "":
                add_anomaly(anomalies, row, row_number, f"missing_{col}", "error", f"{col} 缺失。")

        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            value = row.get(f"{col}_num")
            if pd.isna(value) and str(row[col]).strip() != "":
                add_anomaly(anomalies, row, row_number, f"non_numeric_{col}", "error", f"{col} 不是可解析数值。")
            elif pd.notna(value) and value <= 0:
                add_anomaly(anomalies, row, row_number, f"non_positive_{col}", "error", f"{col} 为零或负数。")

        if "volume_kg" in df.columns:
            volume = row.get("volume_kg_num")
            if pd.isna(volume) and str(row["volume_kg"]).strip() != "":
                add_anomaly(anomalies, row, row_number, "non_numeric_volume_kg", "error", "volume_kg 不是可解析数值。")
            elif pd.notna(volume) and volume == 0:
                add_anomaly(anomalies, row, row_number, "zero_volume_kg", "warning", "volume_kg 为零；可能是停牌/无成交/数据问题，需确认。")
            elif pd.notna(volume) and volume < 0:
                add_anomaly(anomalies, row, row_number, "negative_volume_kg", "error", "volume_kg 为负数。")

        if all(f"{c}_num" in df.columns for c in ["open", "high", "low", "close"]):
            open_, high, low, close = row["open_num"], row["high_num"], row["low_num"], row["close_num"]
            if pd.notna(low) and pd.notna(open_) and pd.notna(high) and not (low <= open_ <= high):
                add_anomaly(anomalies, row, row_number, "open_outside_low_high", "error", "不满足 low <= open <= high。")
            if pd.notna(low) and pd.notna(close) and pd.notna(high) and not (low <= close <= high):
                add_anomaly(anomalies, row, row_number, "close_outside_low_high", "error", "不满足 low <= close <= high。")

    # --- Calendar gap detection (2b). ---
    # PROXY: this counts weekdays (Mon-Fri) skipped between consecutive trading
    # rows. It cannot, by itself, distinguish a genuine SGE holiday closure from a
    # scraping failure -- both look like skipped weekdays.
    # TODO: replace this weekday-count proxy with a real SGE exchange trading
    # calendar, which is the correct way to tell a holiday from a missing day.
    prev_date = None
    for _, row in df.iterrows():
        cur = row["parsed_date"]
        if pd.notna(cur) and prev_date is not None and pd.notna(prev_date):
            calendar_days = int((cur - prev_date).days)
            intervening = int(np.busday_count(np.datetime64(prev_date, "D"), np.datetime64(cur, "D"))) - 1
            if intervening >= 1:
                add_anomaly(
                    anomalies, row, int(row["orig_row_number"]), "weekday_gap", "warning",
                    f"与上一交易日相隔 {calendar_days} 个日历日，其间有 {intervening} 个工作日缺失（可能为交易所假期或抓取缺口）。",
                )
                if intervening > config.max_expected_holiday_weekdays:
                    add_anomaly(
                        anomalies, row, int(row["orig_row_number"]), "large_unexplained_gap", "error",
                        f"其间有 {intervening} 个工作日缺失，超过预期假期上限 {config.max_expected_holiday_weekdays} 个工作日，疑似抓取缺失。",
                    )
        prev_date = cur

    # --- Extreme close-to-close change (warning). ---
    if "close_num" in df.columns:
        df["close_pct_change"] = df["close_num"].pct_change()
        price_jump = df["close_pct_change"].abs() > config.price_jump_threshold
        price_rz = robust_z(df["close_pct_change"])
        robust_price_jump = price_rz.abs() > config.robust_z_threshold
        for _, row in df.loc[price_jump | robust_price_jump].iterrows():
            pct = row.get("close_pct_change")
            add_anomaly(
                anomalies, row, int(row["orig_row_number"]),
                "large_close_change_warning", "warning",
                f"close 单日变化 {pct:.2%}，属于极端变化警告，不自动判定为错误。",
            )

    # --- Extreme volume change (warning). ---
    if "volume_kg_num" in df.columns:
        prev_volume = df["volume_kg_num"].shift(1)
        df["volume_pct_change"] = (df["volume_kg_num"] - prev_volume) / prev_volume.replace(0, np.nan)
        volume_jump = df["volume_pct_change"].abs() > config.volume_jump_threshold
        volume_rz = robust_z(np.log(df["volume_kg_num"].replace(0, np.nan)) - np.log(prev_volume.replace(0, np.nan)))
        robust_volume_jump = volume_rz.abs() > config.robust_z_threshold
        for _, row in df.loc[volume_jump | robust_volume_jump].iterrows():
            pct = row.get("volume_pct_change")
            add_anomaly(
                anomalies, row, int(row["orig_row_number"]),
                "large_volume_change_warning", "warning",
                f"volume_kg 单日变化 {pct:.2%}，属于极端变化警告，不自动判定为错误。",
            )

    # --- Extreme intraday range (warning) (2c). ---
    # Stage 1 previously inspected only close-to-close movement; a bad fill that
    # produced a spurious intraday extreme while leaving close normal was invisible.
    if all(f"{c}_num" in df.columns for c in ["high", "low", "close"]):
        valid_range = (df["high_num"] > 0) & (df["low_num"] > 0) & (df["close_num"] > 0)
        df["intraday_range"] = np.nan
        df.loc[valid_range, "intraday_range"] = (
            (df.loc[valid_range, "high_num"] - df.loc[valid_range, "low_num"]) / df.loc[valid_range, "close_num"]
        )
        range_jump = df["intraday_range"].abs() > config.intraday_range_threshold
        range_rz = robust_z(df["intraday_range"])
        robust_range_jump = range_rz.abs() > config.robust_z_threshold
        for _, row in df.loc[range_jump | robust_range_jump].iterrows():
            val = row.get("intraday_range")
            add_anomaly(
                anomalies, row, int(row["orig_row_number"]),
                "extreme_intraday_range", "warning",
                f"日内振幅 (high-low)/close = {val:.2%}，属于极端日内振幅警告，需人工复核。",
            )

    anomalies_df = pd.DataFrame(anomalies)
    if not anomalies_df.empty:
        anomalies_df["parsed_date"] = pd.to_datetime(anomalies_df["parsed_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    summary = build_summary(
        raw, df, source, source_notes, missing_expected, extra_columns, anomalies_df, config,
        original_is_sorted, source_sha256, source_bytes,
    )
    return raw, anomalies_df, summary, source


def build_summary(
    raw: pd.DataFrame,
    df: pd.DataFrame,
    source: Path,
    source_notes: list[str],
    missing_expected: list[str],
    extra_columns: list[str],
    anomalies_df: pd.DataFrame,
    config: AuditConfig,
    original_is_sorted: bool,
    source_sha256: str,
    source_bytes: int,
) -> dict:
    date_series = df["parsed_date"] if "parsed_date" in df else pd.Series(dtype="datetime64[ns]")
    numeric_cols = [c for c in ["open", "high", "low", "close", "volume_kg"] if f"{c}_num" in df.columns]
    missing_counts = {c: int((raw[c].astype(str).str.strip() == "").sum()) for c in raw.columns}
    non_numeric_counts = {
        c: int(df[f"{c}_num"].isna().sum() - (raw[c].astype(str).str.strip() == "").sum())
        for c in numeric_cols
    }
    non_positive_counts = {
        c: int((df[f"{c}_num"].dropna() <= 0).sum())
        for c in numeric_cols
    }
    rule_counts = (
        anomalies_df.groupby(["severity", "rule"]).size().reset_index(name="count").to_dict("records")
        if not anomalies_df.empty
        else []
    )
    return {
        "source_file": repository_relative(source),
        "source_notes": source_notes,
        "source_sha256": source_sha256,
        "source_bytes": source_bytes,
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "raw_dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "expected_columns": EXPECTED_COLUMNS,
        "missing_expected_columns": missing_expected,
        "extra_columns": extra_columns,
        "date_min": date_series.min().strftime("%Y-%m-%d") if date_series.notna().any() else "n/a",
        "date_max": date_series.max().strftime("%Y-%m-%d") if date_series.notna().any() else "n/a",
        "date_parse_failures": int(date_series.isna().sum()),
        "duplicate_dates": int(date_series.duplicated(keep=False).sum()),
        # Reflects the ORIGINAL file order, not the sorted order used for statistics.
        "is_sorted_by_date": original_is_sorted,
        "missing_counts": missing_counts,
        "non_numeric_counts": non_numeric_counts,
        "non_positive_counts": non_positive_counts,
        "anomaly_rows_total": int(len(anomalies_df)),
        "error_rows_total": int((anomalies_df["severity"] == "error").sum()) if not anomalies_df.empty else 0,
        "warning_rows_total": int((anomalies_df["severity"] == "warning").sum()) if not anomalies_df.empty else 0,
        "rule_counts": rule_counts,
        "thresholds": {
            "large_close_abs_pct_change": config.price_jump_threshold,
            "large_volume_abs_pct_change": config.volume_jump_threshold,
            "robust_z_threshold": config.robust_z_threshold,
            "extreme_intraday_range": config.intraday_range_threshold,
            "max_expected_holiday_weekdays": config.max_expected_holiday_weekdays,
        },
    }


def write_source_manifest(summary: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "path": summary["source_file"],
        "sha256": summary["source_sha256"],
        "bytes": summary["source_bytes"],
        "n_rows": summary["rows"],
        "audit_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_无_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def write_report(summary: dict, anomalies_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "data_audit_report.md"
    anomaly_path = output_dir / "data_audit_anomalies.csv"
    anomalies_df.to_csv(anomaly_path, index=False, encoding="utf-8-sig")

    missing_rows = [{"field": k, "missing_count": v} for k, v in summary["missing_counts"].items()]
    non_numeric_rows = [{"field": k, "non_numeric_count": v} for k, v in summary["non_numeric_counts"].items()]
    non_positive_rows = [{"field": k, "non_positive_or_zero_count": v} for k, v in summary["non_positive_counts"].items()]

    report = f"""# Stage 1 Data Audit Report: Sina Au99.99 Daily OHLCV

## Scope

This is a data audit only. It detects and reports; it never cleans, filters,
imputes, corrects, or reorders the raw CSV, and it returns its input unmodified.
Disposition of any finding (keep, exclude, set-nan) is recorded separately in
`data/decisions.csv`, not here.

## Source and Provenance

- Source file used: `{summary["source_file"]}`
- SHA-256: `{summary["source_sha256"]}`
- Size (bytes): {summary["source_bytes"]}
- Notes: {"; ".join(summary["source_notes"]) if summary["source_notes"] else "none"}
- Rows: {summary["rows"]}
- Columns: {", ".join(summary["columns"])}
- Expected columns: {", ".join(summary["expected_columns"])}
- Missing expected columns: {", ".join(summary["missing_expected_columns"]) if summary["missing_expected_columns"] else "none"}
- Extra columns: {", ".join(summary["extra_columns"]) if summary["extra_columns"] else "none"}

The SHA-256 above is also written to `source_manifest.json`; Stage 2 compares it
against the file it reads, so the audited file and the analysed file cannot
silently diverge.

## Why These Checks Matter

- File and column checks ensure the audit is being applied to the intended dataset.
- Date parsing, sorting, and duplicate checks protect time-series calculations from hidden ordering or identity errors. Rows are sorted by date before any change statistic is computed, so a mis-ordered file cannot corrupt the jump statistics; a mis-ordered raw file is itself reported as an error.
- Calendar-gap checks compare adjacent trading rows and count skipped weekdays, surfacing runs of potentially missing trading days.
- Missing, non-numeric, zero, and negative value checks prevent invalid prices or volumes from contaminating later returns and volatility calculations.
- OHLC logic checks catch hard data errors where open or close falls outside the reported daily low-high range.
- Extreme close-to-close, volume, and intraday-range changes are warning flags only; they may reflect real market moves, source glitches, unit changes, or data-entry problems.

## Date Audit

- Date range: {summary["date_min"]} to {summary["date_max"]}
- Date parse failures: {summary["date_parse_failures"]}
- Duplicate date rows: {summary["duplicate_dates"]}
- Sorted by parsed date (original file order): {summary["is_sorted_by_date"]}

## Missing Values

{markdown_table(missing_rows, ["field", "missing_count"])}

## Numeric Validity

### Non-Numeric Values

{markdown_table(non_numeric_rows, ["field", "non_numeric_count"])}

### Zero or Negative Values

{markdown_table(non_positive_rows, ["field", "non_positive_or_zero_count"])}

## Anomaly Summary

- Total anomaly table rows: {summary["anomaly_rows_total"]}
- Error-level flags: {summary["error_rows_total"]}
- Warning-level flags: {summary["warning_rows_total"]}

{markdown_table(summary["rule_counts"], ["severity", "rule", "count"])}

## Rules and Thresholds

- `date_not_sorted` (error): raw file dates not in ascending order.
- `duplicate_date` (error): a date appears on more than one row.
- `date_parse_failed` / `missing_*` / `non_numeric_*` / `non_positive_*` (error): field-level validity.
- `open_outside_low_high` / `close_outside_low_high` (error): OHLC bracket violated.
- `weekday_gap` (warning): consecutive rows skip one or more weekdays.
- `large_unexplained_gap` (error): a gap skips more than {summary["thresholds"]["max_expected_holiday_weekdays"]} intervening weekdays.
- `large_close_change_warning` (warning): absolute daily close-to-close change > {summary["thresholds"]["large_close_abs_pct_change"]:.2%}.
- `large_volume_change_warning` (warning): absolute daily volume change > {summary["thresholds"]["large_volume_abs_pct_change"]:.2f} times previous day.
- `extreme_intraday_range` (warning): `(high-low)/close` > {summary["thresholds"]["extreme_intraday_range"]:.2%}.
- Robust outlier warning: robust z-score absolute value > {summary["thresholds"]["robust_z_threshold"]:.2f} (applied to close change, volume change, and intraday range).

## Coverage and Limitations

A clean report does not prove clean data. Stage 1 does **not**:

- consult an exchange trading calendar — calendar-gap detection is a weekday-count
  proxy and cannot by itself tell a genuine SGE holiday from a scraping failure;
- verify prices against any second source — an internally consistent but wrong
  price passes every check here;
- detect a unit change (e.g. a currency or scale change) applied uniformly across
  the whole series, since nothing about it looks anomalous row-to-row.

## Interpretation

The dataset is not altered by this audit. Error-level rows should be reviewed before downstream cleaning. Warning-level rows should not be automatically removed; they require source or market-context review.

Generated files:

- `data_audit_report.md`
- `data_audit_anomalies.csv`
- `source_manifest.json`
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    config = AuditConfig(
        source=Path(args.source) if args.source else None,
        output_dir=Path(args.output_dir),
        price_jump_threshold=args.price_jump_threshold,
        volume_jump_threshold=args.volume_jump_threshold,
        robust_z_threshold=args.robust_z_threshold,
        intraday_range_threshold=args.intraday_range_threshold,
        max_expected_holiday_weekdays=args.max_expected_holiday_weekdays,
    )
    _raw, anomalies_df, summary, _source = audit(config)
    report_path = write_report(summary, anomalies_df, config.output_dir)
    manifest_path = write_source_manifest(summary, config.output_dir)
    print(f"Audit complete. Report: {report_path}")
    print(f"Anomaly table: {config.output_dir / 'data_audit_anomalies.csv'}")
    print(f"Source manifest: {manifest_path}")
    print(f"Rows: {summary['rows']}; errors: {summary['error_rows_total']}; warnings: {summary['warning_rows_total']}")


if __name__ == "__main__":
    main()
