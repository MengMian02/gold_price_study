"""Construct stage-2 research variables for RMB Au99.99 round-level study.

This stage only constructs, validates, and saves variables. It does not compare
groups, plot effects, run tests, estimate regressions, or draw research
conclusions. The input CSV is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "Au9999.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "stage2_variable_construction"
STAGE1_MANIFEST = WORKSPACE_ROOT / "outputs" / "stage1_data_audit" / "source_manifest.json"

DATE_COL = "date"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VOLUME_COL = "volume_kg"
REQUIRED_COLUMNS = [DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]
STATE_COLUMNS = [
    "nearest_level",
    "signed_distance",
    "distance_to_level",
    "near_2",
    "near_3",
    "near_5",
    "near_10",
    "side",
    "side_near_5",
]
NUMERIC_TOLERANCE = 1e-10


@dataclass
class Config:
    source: Path = DEFAULT_SOURCE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    level_step: float = 50.0
    near_thresholds: tuple[int, ...] = (2, 3, 5, 10)
    main_near_threshold: int = 5
    normal_volume_window: int = 20
    potential_break_return_threshold: float = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct Au99.99 research variables")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input CSV path. This file is read only.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated files.")
    parser.add_argument("--level-step", type=float, default=50.0)
    parser.add_argument("--normal-volume-window", type=int, default=20)
    parser.add_argument("--break-return-threshold", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        level_step=args.level_step,
        normal_volume_window=args.normal_volume_window,
        potential_break_return_threshold=args.break_return_threshold,
    )
    result = construct_and_save(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_against_manifest(source: Path) -> None:
    """Gate: the file Stage 1 audited must be the file this stage analyses.

    Stage 1 records the SHA-256 of the audited file in source_manifest.json. If it
    exists and disagrees with the file read here, the audited and analysed data
    have diverged and we refuse to run. If it is absent (e.g. a first-time user
    who has not run Stage 1 yet), we warn and continue.
    """
    if not STAGE1_MANIFEST.exists():
        print(
            "WARNING: Stage 1 source_manifest.json not found; skipping the source-hash gate. "
            "Run audit_sina_au9999.py first to enable it."
        )
        return
    manifest = json.loads(STAGE1_MANIFEST.read_text(encoding="utf-8"))
    audited_hash = manifest.get("sha256")
    current_hash = _sha256_file(source)
    if audited_hash != current_hash:
        raise ValueError(
            "Source-hash gate failed: the file audited by Stage 1 differs from the file being analysed.\n"
            f"  Stage 1 audited : {manifest.get('path')}  sha256={audited_hash}\n"
            f"  Stage 2 reads   : {source}  sha256={current_hash}\n"
            "Re-run Stage 1 on the same file before constructing variables."
        )


def run_input_gates(raw: pd.DataFrame, source: Path) -> None:
    """Refuse to run on data that is wrong without needing any external information.

    These are gates, not decisions: duplicate dates, out-of-order dates, and OHLC
    bracket violations can be judged wrong from the file alone, so Stage 2 stops
    rather than silently constructing variables on broken data. (Phase 1
    diagnostics confirmed the current file has none of these.) Judgement calls
    that need information outside the file are recorded in data/decisions.csv and
    applied later, not gated here.
    """
    parsed = pd.to_datetime(raw[DATE_COL], errors="coerce", dayfirst=True)

    duplicated = parsed.duplicated(keep=False) & parsed.notna()
    if duplicated.any():
        dates = sorted(parsed[duplicated].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(f"Input gate failed: duplicate dates present in the raw file: {dates}")

    valid = parsed.dropna()
    if len(valid) > 1 and not valid.is_monotonic_increasing:
        raise ValueError("Input gate failed: raw file dates are not in ascending order.")

    open_ = pd.to_numeric(raw[OPEN_COL].replace("", pd.NA), errors="coerce")
    high = pd.to_numeric(raw[HIGH_COL].replace("", pd.NA), errors="coerce")
    low = pd.to_numeric(raw[LOW_COL].replace("", pd.NA), errors="coerce")
    close = pd.to_numeric(raw[CLOSE_COL].replace("", pd.NA), errors="coerce")
    complete = open_.notna() & high.notna() & low.notna() & close.notna()
    bracket_ok = (high >= low) & (low <= open_) & (open_ <= high) & (low <= close) & (close <= high)
    bad = complete & ~bracket_ok
    if bad.any():
        rows = (np.nonzero(bad.to_numpy())[0] + 2).tolist()
        shown = rows[:20]
        raise ValueError(
            "Input gate failed: OHLC bracket violated (need low<=open<=high, low<=close<=high, high>=low) at "
            f"CSV rows {shown}" + (" ..." if len(rows) > len(shown) else "")
        )

    verify_source_against_manifest(source)


def construct_and_save(config: Config) -> dict:
    if not config.source.exists():
        raise FileNotFoundError(f"Input file not found: {config.source}")

    raw = pd.read_csv(config.source, dtype=str, keep_default_na=False)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    run_input_gates(raw, config.source)

    original_row_count = len(raw)
    df = raw.copy()
    df["source_row_number"] = np.arange(2, len(df) + 2)
    df["trade_date"] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)

    for source_col, output_col in [
        (OPEN_COL, "open_price"),
        (HIGH_COL, "high_price"),
        (LOW_COL, "low_price"),
        (CLOSE_COL, "close_price"),
        (VOLUME_COL, "volume_kg_raw"),
    ]:
        df[output_col] = pd.to_numeric(df[source_col].replace("", pd.NA), errors="coerce")

    df = df.sort_values(["trade_date", "source_row_number"], kind="mergesort").reset_index(drop=True)
    df["trading_index"] = np.arange(len(df))

    add_round_level_variables(df, config)
    add_return_variables(df)
    add_parkinson_variables(df)
    add_volume_variables(df, config)
    add_lagged_state_variables(df)
    add_quality_flags(df, config)

    validation = run_validations(df, original_row_count, config)
    sample_df = build_manual_check_sample(df)
    dictionary_df = build_variable_dictionary(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_data = config.output_dir / "au9999_research_variables_stage2.csv"
    dictionary_csv = config.output_dir / "variable_dictionary.csv"
    dictionary_md = config.output_dir / "variable_dictionary.md"
    report_path = config.output_dir / "variable_construction_validation_report.md"
    sample_path = config.output_dir / "manual_verification_examples.csv"

    df.to_csv(output_data, index=False, encoding="utf-8-sig")
    dictionary_df.to_csv(dictionary_csv, index=False, encoding="utf-8-sig")
    dictionary_md.write_text(dictionary_to_markdown(dictionary_df), encoding="utf-8")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_validation_report(df, sample_df, validation, config), encoding="utf-8")

    return {
        "input_file": str(config.source),
        "output_data": str(output_data),
        "variable_dictionary_csv": str(dictionary_csv),
        "variable_dictionary_md": str(dictionary_md),
        "validation_report": str(report_path),
        "manual_examples": str(sample_path),
        "original_rows": int(original_row_count),
        "output_rows": int(len(df)),
        "all_validations_passed": bool(all(item["passed"] for item in validation["checks"])),
        "manual_judgment_items": validation["manual_judgment_items"],
    }


def round_half_up_to_step(series: pd.Series, step: float) -> pd.Series:
    # Explicit half-up rule: exact midpoints between two levels choose the higher level.
    return step * np.floor((series / step) + 0.5)


def add_round_level_variables(df: pd.DataFrame, config: Config) -> None:
    valid_close = df["close_price"].notna()
    df["nearest_level"] = np.nan
    df.loc[valid_close, "nearest_level"] = round_half_up_to_step(df.loc[valid_close, "close_price"], config.level_step)
    df["signed_distance"] = df["close_price"] - df["nearest_level"]
    df["distance_to_level"] = df["signed_distance"].abs()

    for threshold in config.near_thresholds:
        col = f"near_{threshold}"
        df[col] = np.where(df["distance_to_level"].notna(), (df["distance_to_level"] <= threshold).astype("Int64"), pd.NA)

    df["side"] = pd.NA
    df.loc[df["signed_distance"] < 0, "side"] = "below"
    df.loc[df["signed_distance"] > 0, "side"] = "above"
    df.loc[df["signed_distance"] == 0, "side"] = "at_level"
    df["side_near_5"] = np.where(df["near_5"].eq(1), df["side"], "not_near")


def add_return_variables(df: pd.DataFrame) -> None:
    prev_close = df["close_price"].shift(1)
    valid = (df["close_price"] > 0) & (prev_close > 0)
    df["prev_close_price"] = prev_close
    df["log_return"] = np.nan
    df.loc[valid, "log_return"] = np.log(df.loc[valid, "close_price"] / prev_close.loc[valid])
    df["abs_log_return"] = df["log_return"].abs()


def add_parkinson_variables(df: pd.DataFrame) -> None:
    valid = (df["high_price"] > 0) & (df["low_price"] > 0) & (df["high_price"] >= df["low_price"])
    ratio_log = np.log(df.loc[valid, "high_price"] / df.loc[valid, "low_price"])
    df["parkinson_valid_input"] = valid.astype("Int64")
    df["parkinson_variance"] = np.nan
    df.loc[valid, "parkinson_variance"] = (ratio_log**2) / (4 * np.log(2))
    df["parkinson_volatility"] = np.sqrt(df["parkinson_variance"])


def add_volume_variables(df: pd.DataFrame, config: Config) -> None:
    volume = df["volume_kg_raw"]
    valid = volume.notna() & (volume >= 0)
    df["volume_valid_for_log"] = valid.astype("Int64")
    df["log_volume"] = np.nan
    df.loc[valid, "log_volume"] = np.log1p(volume.loc[valid])
    df["normal_log_volume_20d_lagged"] = (
        df["log_volume"].shift(1).rolling(config.normal_volume_window, min_periods=config.normal_volume_window).mean()
    )
    df["abnormal_volume_20d"] = df["log_volume"] - df["normal_log_volume_20d_lagged"]
    df["volume_history_count_20d_lagged"] = (
        df["log_volume"].shift(1).rolling(config.normal_volume_window, min_periods=1).count()
    )


def add_lagged_state_variables(df: pd.DataFrame) -> None:
    for col in STATE_COLUMNS:
        df[f"{col}_lag1"] = df[col].shift(1)


def add_quality_flags(df: pd.DataFrame, config: Config) -> None:
    flags: list[list[str]] = [[] for _ in range(len(df))]

    def mark(mask: pd.Series, flag: str) -> None:
        for i in df.index[mask.fillna(False)]:
            flags[i].append(flag)

    mark(df["trade_date"].isna(), "date_parse_failed")
    mark(df["trade_date"].duplicated(keep=False) & df["trade_date"].notna(), "duplicate_date")
    mark(df["close_price"].isna(), "close_missing_or_non_numeric")
    for col in ["open_price", "high_price", "low_price", "close_price"]:
        mark(df[col].notna() & (df[col] <= 0), f"{col}_non_positive")
    mark(df["high_price"].notna() & df["low_price"].notna() & (df["high_price"] < df["low_price"]), "high_below_low")
    mark(~df["parkinson_valid_input"].eq(1), "parkinson_input_invalid")
    mark(df["volume_kg_raw"].isna(), "volume_missing_or_non_numeric")
    mark(df["volume_kg_raw"].notna() & (df["volume_kg_raw"] < 0), "volume_negative")
    mark(df["volume_kg_raw"].eq(0), "volume_zero")
    mark(
        df["normal_log_volume_20d_lagged"].isna() & df["log_volume"].notna(),
        "insufficient_20d_lagged_volume_history",
    )
    mark(df["log_return"].abs() > config.potential_break_return_threshold, "potential_price_break_or_data_gap")

    df["variable_construction_flags"] = [";".join(item) for item in flags]


def run_validations(df: pd.DataFrame, original_row_count: int, config: Config) -> dict:
    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    valid_distance = df["distance_to_level"].dropna()
    add_check(
        "distance_to_level_between_0_and_25",
        valid_distance.between(0, config.level_step / 2 + NUMERIC_TOLERANCE).all(),
        f"valid={len(valid_distance)}, min={valid_distance.min()}, max={valid_distance.max()}",
    )

    monotonic_near = (
        (df["near_2"].fillna(0) <= df["near_3"].fillna(0))
        & (df["near_3"].fillna(0) <= df["near_5"].fillna(0))
        & (df["near_5"].fillna(0) <= df["near_10"].fillna(0))
    )
    add_check("near_threshold_nesting", monotonic_near.all(), f"violations={int((~monotonic_near).sum())}")

    distance_identity = (df["distance_to_level"] - df["signed_distance"].abs()).abs()
    add_check(
        "distance_equals_abs_signed_distance",
        distance_identity.dropna().le(NUMERIC_TOLERANCE).all(),
        f"max_abs_diff={distance_identity.max()}",
    )

    levels = df["nearest_level"].dropna()
    add_check(
        "nearest_level_multiple_of_50",
        ((levels % config.level_step).abs() <= NUMERIC_TOLERANCE).all(),
        f"valid={len(levels)}",
    )

    parkinson_var = df["parkinson_variance"].dropna()
    parkinson_vol = df["parkinson_volatility"].dropna()
    add_check(
        "parkinson_non_negative",
        (parkinson_var.ge(0).all() and parkinson_vol.ge(0).all()),
        f"variance_valid={len(parkinson_var)}, volatility_valid={len(parkinson_vol)}",
    )
    square_diff = (df["parkinson_volatility"] ** 2 - df["parkinson_variance"]).abs()
    add_check(
        "parkinson_vol_squared_equals_variance",
        square_diff.dropna().le(1e-12).all(),
        f"max_abs_diff={square_diff.max()}",
    )

    volume_window_ok = validate_lagged_volume_window(df, config.normal_volume_window)
    add_check(
        "volume_20d_baseline_lagged_no_current_or_future",
        volume_window_ok["passed"],
        volume_window_ok["detail"],
    )

    lag_ok = validate_lagged_state_variables(df)
    add_check("lagged_state_variables_from_previous_row", lag_ok["passed"], lag_ok["detail"])

    add_check(
        "row_count_preserved",
        len(df) == original_row_count,
        f"input_rows={original_row_count}, output_rows={len(df)}",
    )

    return {
        "checks": checks,
        "counts": build_allowed_summary_counts(df),
        "manual_judgment_items": build_manual_judgment_items(df, config),
    }


def validate_lagged_volume_window(df: pd.DataFrame, window: int) -> dict:
    baseline = df["normal_log_volume_20d_lagged"]
    valid_indices = df.index[baseline.notna()]
    for idx in valid_indices:
        prior_values = df.loc[idx - window : idx - 1, "log_volume"] if idx >= window else pd.Series(dtype=float)
        expected = prior_values.mean() if len(prior_values) == window and prior_values.notna().all() else np.nan
        actual = baseline.loc[idx]
        if pd.isna(expected) or abs(actual - expected) > 1e-12:
            return {"passed": False, "detail": f"first violation at output row index {idx}"}
    return {"passed": True, "detail": f"checked {len(valid_indices)} non-missing baselines using only t-1 to t-20"}


def validate_lagged_state_variables(df: pd.DataFrame) -> dict:
    for col in STATE_COLUMNS:
        lag_col = f"{col}_lag1"
        expected = df[col].shift(1)
        actual = df[lag_col]
        mismatch = ~(actual.fillna("__NA__").astype(str).eq(expected.fillna("__NA__").astype(str)))
        if mismatch.any():
            return {"passed": False, "detail": f"{lag_col} mismatch count={int(mismatch.sum())}"}
    return {"passed": True, "detail": f"checked {len(STATE_COLUMNS)} lagged state variables"}


def build_allowed_summary_counts(df: pd.DataFrame) -> dict:
    flag_series = df["variable_construction_flags"].str.split(";").explode()
    flag_counts = flag_series[flag_series.ne("")].value_counts().to_dict()
    near_counts = {}
    for threshold in [2, 3, 5, 10]:
        col = f"near_{threshold}"
        count = int(df[col].eq(1).sum())
        near_counts[col] = {"count": count, "share": count / len(df) if len(df) else np.nan}
    side_near_5_counts = df.loc[df["near_5"].eq(1), "side"].value_counts(dropna=False).to_dict()
    variable_counts = {
        col: {"valid": int(df[col].notna().sum()), "missing": int(df[col].isna().sum())}
        for col in [
            "nearest_level",
            "signed_distance",
            "distance_to_level",
            "near_2",
            "near_3",
            "near_5",
            "near_10",
            "side",
            "side_near_5",
            "log_return",
            "abs_log_return",
            "parkinson_variance",
            "parkinson_volatility",
            "log_volume",
            "normal_log_volume_20d_lagged",
            "abnormal_volume_20d",
        ]
    }
    return {
        "input_rows": int(len(df)),
        "output_rows": int(len(df)),
        "variable_valid_missing_counts": variable_counts,
        "near_counts": near_counts,
        "side_counts_within_near_5": side_near_5_counts,
        "quality_flag_counts": flag_counts,
        "volume_missing_count": int(df["volume_kg_raw"].isna().sum()),
        "volume_zero_count": int(df["volume_kg_raw"].eq(0).sum()),
        "volume_negative_count": int((df["volume_kg_raw"] < 0).sum()),
        "insufficient_20d_volume_history_count": int(
            df["variable_construction_flags"].str.contains("insufficient_20d_lagged_volume_history", regex=False).sum()
        ),
    }


def build_manual_judgment_items(df: pd.DataFrame, config: Config) -> list[dict]:
    flagged_breaks = df[df["variable_construction_flags"].str.contains("potential_price_break_or_data_gap", regex=False)]
    if flagged_breaks.empty:
        return []
    return [
        {
            "issue": "potential_price_break_or_data_gap",
            "count": int(len(flagged_breaks)),
            "rule": f"abs(log_return) > {config.potential_break_return_threshold}",
            "note": "Cannot determine from this file alone whether these are real market moves, contract/market breaks, or data issues; manual review is needed before cleaning.",
        }
    ]


def build_manual_check_sample(df: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []

    criteria = [
        ("near_5_below", df["near_5"].eq(1) & df["side"].eq("below")),
        ("near_5_above", df["near_5"].eq(1) & df["side"].eq("above")),
        ("at_level", df["side"].eq("at_level")),
        ("not_near_5", df["near_5"].eq(0)),
        ("near_midpoint", df["distance_to_level"].between(24.5, 25.0, inclusive="both")),
        ("positive_abnormal_volume", df["abnormal_volume_20d"] > 0),
        ("negative_abnormal_volume", df["abnormal_volume_20d"] < 0),
        ("high_parkinson_volatility", df["parkinson_volatility"].rank(method="first", ascending=False) <= 2),
        ("low_parkinson_volatility", df["parkinson_volatility"].rank(method="first", ascending=True) <= 2),
    ]

    for label, mask in criteria:
        rows = df.loc[mask].head(2).copy()
        if not rows.empty:
            rows["sample_reason"] = label
            selected.append(rows)

    if selected:
        sample = pd.concat(selected, ignore_index=True)
        sample = sample.drop_duplicates(subset=["source_row_number", "sample_reason"])
    else:
        sample = df.head(10).copy()
        sample["sample_reason"] = "fallback_first_rows"

    columns = [
        "sample_reason",
        "trade_date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume_kg_raw",
        "nearest_level",
        "signed_distance",
        "distance_to_level",
        "near_5",
        "side",
        "log_return",
        "abs_log_return",
        "parkinson_variance",
        "parkinson_volatility",
        "log_volume",
        "normal_log_volume_20d_lagged",
        "abnormal_volume_20d",
        "near_5_lag1",
        "side_lag1",
    ]
    return sample[columns].head(18)


def build_variable_dictionary(config: Config) -> pd.DataFrame:
    rows = [
        var("trade_date", "parse(date)", "date", "当日", "日期无法解析", "交易日期。"),
        var("open_price", "numeric(open)", "open", "当日", "缺失或非数值", "开盘价，人民币元/克。"),
        var("high_price", "numeric(high)", "high", "当日", "缺失或非数值", "最高价，人民币元/克。"),
        var("low_price", "numeric(low)", "low", "当日", "缺失或非数值", "最低价，人民币元/克。"),
        var("close_price", "numeric(close)", "close", "当日", "缺失或非数值", "收盘价，人民币元/克。"),
        var("volume_kg_raw", "numeric(volume_kg)", "volume_kg", "当日", "缺失或非数值", "成交量，千克。"),
        var("nearest_level", "50 * floor(close_price / 50 + 0.5)", "close_price", "当日", "close_price 缺失", "最近的50元整数关口；中点选择较高关口。"),
        var("signed_distance", "close_price - nearest_level", "close_price, nearest_level", "当日", "任一输入缺失", "收盘价相对最近关口的有符号距离。"),
        var("distance_to_level", "abs(signed_distance)", "signed_distance", "当日", "signed_distance 缺失", "收盘价离最近关口的绝对距离。"),
        var("near_2", "1(distance_to_level <= 2)", "distance_to_level", "当日", "distance_to_level 缺失", "是否位于最近关口2元以内。"),
        var("near_3", "1(distance_to_level <= 3)", "distance_to_level", "当日", "distance_to_level 缺失", "是否位于最近关口3元以内。"),
        var("near_5", "1(distance_to_level <= 5)", "distance_to_level", "当日", "distance_to_level 缺失", "主定义：是否位于最近关口5元以内。"),
        var("near_10", "1(distance_to_level <= 10)", "distance_to_level", "当日", "distance_to_level 缺失", "是否位于最近关口10元以内。"),
        var("side", "below if signed_distance<0; above if >0; at_level if =0", "signed_distance", "当日", "signed_distance 缺失", "收盘价位于最近关口下方、上方或正好等于关口。"),
        var("side_near_5", "side if near_5=1 else not_near", "side, near_5", "当日", "near_5 缺失", "主定义范围内的方向；非near_5编码为not_near。"),
        var("log_return", "ln(close_price / close_price_lag1)", "close_price", "使用上一交易日，不用未来", "首行或当前/上一收盘价无效", "收盘到收盘对数收益率。"),
        var("abs_log_return", "abs(log_return)", "log_return", "当日", "log_return 缺失", "收益波动代理变量。"),
        var("parkinson_variance", "(ln(high_price / low_price)^2) / (4 ln 2)", "high_price, low_price", "当日", "high<=0, low<=0, high<low 或缺失", "Parkinson日内方差。"),
        var("parkinson_volatility", "sqrt(parkinson_variance)", "parkinson_variance", "当日", "parkinson_variance 缺失", "Parkinson日内波动率。"),
        var("log_volume", "ln(1 + volume_kg_raw)", "volume_kg_raw", "当日", "volume缺失、非数值或负数", "对数成交量。"),
        var("normal_log_volume_20d_lagged", "mean(log_volume[t-20]...log_volume[t-1])", "log_volume", "仅使用过去20个交易日", "不足20个有效历史成交量", "滞后20日正常成交量基准。"),
        var("abnormal_volume_20d", "log_volume - normal_log_volume_20d_lagged", "log_volume, normal_log_volume_20d_lagged", "当日减历史基准", "任一输入缺失", "相对过去20个交易日基准的异常成交量。"),
    ]
    for col in STATE_COLUMNS:
        rows.append(
            var(
                f"{col}_lag1",
                f"{col}[t-1]",
                col,
                "上一条交易日记录，不是自然日前一天",
                "首行或上一交易日变量缺失",
                f"{col} 的一期滞后状态变量。",
            )
        )
    rows.extend(
        [
            var("prev_close_price", "close_price[t-1]", "close_price", "上一条交易日记录", "首行", "收益率人工核验中间变量。"),
            var("volume_history_count_20d_lagged", "count(valid log_volume[t-20]...t-1)", "log_volume", "仅使用过去", "无", "成交量基准窗口有效历史数量。"),
            var("parkinson_valid_input", "1(high>0 and low>0 and high>=low)", "high_price, low_price", "当日", "无", "Parkinson输入是否有效。"),
            var("volume_valid_for_log", "1(volume>=0 and numeric)", "volume_kg_raw", "当日", "无", "成交量是否可取ln(1+volume)。"),
            var("variable_construction_flags", "semicolon-separated rule flags", "all inputs", "当日或历史窗口", "无", "变量构造质量标记。"),
        ]
    )
    return pd.DataFrame(rows)


def var(name: str, formula: str, inputs: str, direction: str, missing: str, meaning: str) -> dict:
    return {
        "variable_name": name,
        "formula": formula,
        "input_fields": inputs,
        "time_direction": direction,
        "missing_value_conditions": missing,
        "economic_meaning": meaning,
    }


def dictionary_to_markdown(dictionary_df: pd.DataFrame) -> str:
    return "# Variable Dictionary\n\n" + df_to_markdown(dictionary_df)


def build_validation_report(df: pd.DataFrame, sample_df: pd.DataFrame, validation: dict, config: Config) -> str:
    counts = validation["counts"]
    checks_df = pd.DataFrame(validation["checks"])
    variable_counts_df = pd.DataFrame(
        [
            {"variable": k, "valid": v["valid"], "missing": v["missing"]}
            for k, v in counts["variable_valid_missing_counts"].items()
        ]
    )
    near_counts_df = pd.DataFrame(
        [
            {"variable": k, "count": v["count"], "share": f"{v['share']:.4%}"}
            for k, v in counts["near_counts"].items()
        ]
    )
    side_counts_df = pd.DataFrame(
        [{"side": k, "count": v} for k, v in counts["side_counts_within_near_5"].items()]
    )
    flag_counts_df = pd.DataFrame(
        [{"flag": k, "count": v} for k, v in counts["quality_flag_counts"].items()]
    )

    manual_1 = pick_manual_formula_row(sample_df, "near_5_above")
    manual_2 = pick_manual_formula_row(sample_df, "positive_abnormal_volume")

    return f"""# Stage 2 Variable Construction and Validation Report

## Scope

This stage constructs, validates, and saves research variables only. It does not compare Near and Not Near observations, create effect charts, run statistical tests, estimate regressions, search thresholds, or make research conclusions.

## Input and Field Mapping

- Input file: `{config.source}`
- Date field: `{DATE_COL}`
- Open field: `{OPEN_COL}`
- High field: `{HIGH_COL}`
- Low field: `{LOW_COL}`
- Close field: `{CLOSE_COL}`
- Volume field: `{VOLUME_COL}`
- Original rows: {counts["input_rows"]}
- Output rows: {counts["output_rows"]}
- Row count difference: {counts["output_rows"] - counts["input_rows"]}

Rows are sorted by parsed trading date before constructing time-series variables. No rows are dropped to remove missing constructed variables.

## Rounding Rule

Nearest 50-RMB level is computed as `50 * floor(close / 50 + 0.5)`. This is an explicit half-up rule: if the close is exactly halfway between two 50-RMB levels, the higher level is selected. This avoids opaque banker's rounding.

## Validation Checks

{df_to_markdown(checks_df)}

## Variable Valid and Missing Counts

{df_to_markdown(variable_counts_df)}

## Near-Level Counts

{df_to_markdown(near_counts_df)}

## `near_5` Side Counts

{df_to_markdown(side_counts_df) if not side_counts_df.empty else "_No near_5 observations._"}

## Data Quality Flag Counts

{df_to_markdown(flag_counts_df) if not flag_counts_df.empty else "_No flags._"}

## Volume Input Counts

- Missing volume records: {counts["volume_missing_count"]}
- Zero volume records: {counts["volume_zero_count"]}
- Negative volume records: {counts["volume_negative_count"]}
- Records without enough 20-trading-day lagged volume history: {counts["insufficient_20d_volume_history_count"]}

## Potential Manual Judgment Items

{df_to_markdown(pd.DataFrame(validation["manual_judgment_items"])) if validation["manual_judgment_items"] else "_None from configured checks._"}

## Manual Verification Sample

{df_to_markdown(sample_df)}

## Formula Walkthrough 1

{manual_formula_walkthrough(manual_1)}

## Formula Walkthrough 2

{manual_formula_walkthrough(manual_2)}
"""


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].dt.strftime("%Y-%m-%d")
    display = display.replace({np.nan: ""})
    columns = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("|", "\\|").replace("\n", "<br>") for col in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pick_manual_formula_row(sample_df: pd.DataFrame, reason: str) -> pd.Series:
    row = sample_df[sample_df["sample_reason"].eq(reason)]
    if row.empty:
        return sample_df.iloc[0]
    return row.iloc[0]


def manual_formula_walkthrough(row: pd.Series) -> str:
    high = row["high_price"]
    low = row["low_price"]
    close = row["close_price"]
    level = row["nearest_level"]
    volume = row["volume_kg_raw"]
    normal = row["normal_log_volume_20d_lagged"]
    signed = row["signed_distance"]
    distance = row["distance_to_level"]
    return f"""Record `{row["trade_date"]}`:

- Nearest level: `50 * floor({close} / 50 + 0.5) = {level}`.
- Signed distance: `{close} - {level} = {signed}`; absolute distance: `abs({signed}) = {distance}`.
- `near_5`: `{distance} <= 5`, so the stored value is `{row["near_5"]}`.
- Parkinson variance: `(ln({high} / {low})^2) / (4 ln 2) = {row["parkinson_variance"]}`.
- Parkinson volatility: `sqrt({row["parkinson_variance"]}) = {row["parkinson_volatility"]}`.
- Log volume: `ln(1 + {volume}) = {row["log_volume"]}`.
- Abnormal volume: `log_volume - normal_log_volume_20d_lagged = {row["log_volume"]} - {normal} = {row["abnormal_volume_20d"]}`.
"""


if __name__ == "__main__":
    main()
