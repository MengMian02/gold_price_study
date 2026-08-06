"""Stage 2.6: build the canonical working analysis dataset.

This script derives the working dataset that all downstream analysis (Stage 3
onward) should read. It performs exactly two reductions on the Stage 2 variable
file and nothing else:

1. Exclude trading dates whose disposition is ``exclude`` in ``data/decisions.csv``
   (the single, git-tracked record of which findings were reviewed and what was
   decided). The four seed exclusions and their reasons live there, not in code.
2. Drop every volume and volume-derived column, because reliable volume data is
   not available for this series. Volume tokens are also stripped from the
   ``variable_construction_flags`` text so no volume references remain.

It does not clean prices, winsorize, interpolate, compare groups, run tests, or
overwrite the raw CSV or the Stage 2 file. Inputs are read only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = (
    WORKSPACE_ROOT / "outputs" / "stage2_variable_construction" / "au9999_research_variables_stage2.csv"
)
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "stage2_variable_construction"
DEFAULT_OUTPUT_NAME = "au9999_analysis_dataset.csv"
DEFAULT_DECISIONS_FILE = PROJECT_ROOT / "data" / "decisions.csv"

FLAGS_COLUMN = "variable_construction_flags"
DECISIONS_COLUMNS = ["date", "rule", "disposition", "reason", "reviewer", "decided_on"]
VALID_DISPOSITIONS = {"exclude", "keep", "set_nan", "not_applicable"}


@dataclass
class Config:
    source: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_name: str = DEFAULT_OUTPUT_NAME
    decisions_file: Path = DEFAULT_DECISIONS_FILE


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path without exposing local folders."""
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the volume-free working analysis dataset")
    parser.add_argument("--source", default=str(DEFAULT_INPUT), help="Stage 2 variable CSV. Read only.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--decisions-file", default=str(DEFAULT_DECISIONS_FILE), help="Decisions CSV. Read only.")
    return parser.parse_args()


def is_volume_name(name: str) -> bool:
    return "volume" in str(name).lower()


def strip_volume_flags(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    tokens = [t for t in str(value).split(";") if t and not is_volume_name(t)]
    return ";".join(tokens)


def load_exclusions(decisions_file: Path, dataset_dates: set[str]) -> list[str]:
    """Return the dates to exclude, validating the decisions file first.

    Every date in the decisions file must exist in the dataset; a date that does
    not is a typo or a stale entry left over after a data refresh, and we fail
    loudly rather than silently drop nothing.
    """
    if not decisions_file.exists():
        raise FileNotFoundError(f"Decisions file not found: {decisions_file}")
    decisions = pd.read_csv(decisions_file, dtype=str, keep_default_na=False)

    missing_columns = [c for c in DECISIONS_COLUMNS if c not in decisions.columns]
    if missing_columns:
        raise ValueError(f"decisions.csv is missing required columns: {missing_columns}")

    unknown_dispositions = sorted(set(decisions["disposition"]) - VALID_DISPOSITIONS)
    if unknown_dispositions:
        raise ValueError(
            f"decisions.csv has unknown disposition values {unknown_dispositions}; "
            f"allowed: {sorted(VALID_DISPOSITIONS)}."
        )

    unknown_dates = [d for d in decisions["date"].tolist() if d not in dataset_dates]
    if unknown_dates:
        raise ValueError(
            "decisions.csv references dates that are absent from the dataset "
            f"(typos or stale entries after a data refresh): {unknown_dates}"
        )

    return decisions.loc[decisions["disposition"] == "exclude", "date"].tolist()


def build(config: Config) -> dict:
    if not config.source.exists():
        raise FileNotFoundError(f"Stage 2 input not found: {config.source}")

    df = pd.read_csv(config.source, dtype=str, keep_default_na=False)
    input_rows = len(df)
    input_columns = list(df.columns)

    if "trade_date" not in df.columns:
        raise ValueError("Input is missing the required 'trade_date' column.")

    parsed_date = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dataset_dates = set(parsed_date.dropna().tolist())

    excluded_dates = load_exclusions(config.decisions_file, dataset_dates)
    excluded_set = set(excluded_dates)
    keep_mask = ~parsed_date.isin(excluded_set)
    removed_by_date = {d: int((parsed_date == d).sum()) for d in excluded_dates}
    df = df.loc[keep_mask].reset_index(drop=True)

    volume_columns = [c for c in df.columns if is_volume_name(c)]
    df = df.drop(columns=volume_columns)

    if FLAGS_COLUMN in df.columns:
        df[FLAGS_COLUMN] = df[FLAGS_COLUMN].map(strip_volume_flags)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / config.output_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    manifest = {
        "source_file": repository_relative(config.source),
        "output_file": repository_relative(output_path),
        "input_rows": int(input_rows),
        "output_rows": int(len(df)),
        "excluded_dates": excluded_dates,
        "rows_removed_per_excluded_date": removed_by_date,
        "rows_removed_total": int(input_rows - len(df)),
        "input_columns": len(input_columns),
        "output_columns": len(df.columns),
        "dropped_volume_columns": volume_columns,
        "remaining_columns": list(df.columns),
        "no_volume_column_remains": not any(is_volume_name(c) for c in df.columns),
    }

    manifest_path = config.output_dir / "analysis_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_file"] = repository_relative(manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    config = Config(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
        decisions_file=Path(args.decisions_file),
    )
    result = build(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
