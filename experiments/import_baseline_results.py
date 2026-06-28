#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

DATASET_MAP = {"spam": "spambase"}
METHOD_MAP = {
    "fcalc": "fcalc_deterministic",
    "fcalc_rand": "fcalc_randomized",
    "ips_knn": "ips_knn",
}
DEFAULT_CLASSIFIERS = ["fcalc", "fcalc_rand", "ips_knn"]


def ci95_half_width(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n <= 1:
        return 0.0
    return float(T_CRITICAL_95.get(n - 1, 1.96) * clean.std(ddof=1) / math.sqrt(n))


def load_configured_datasets(config_path: Path) -> set[str]:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    datasets = config.get("datasets", {})
    return {name for name, spec in datasets.items() if not isinstance(spec, dict) or spec.get("enabled", True)}


def parse_best_params(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except Exception:
        return text


def imported_rows(imported_dir: Path, classifiers: list[str], configured_datasets: set[str]) -> pd.DataFrame:
    macro_path = imported_dir / "repeated_macro_f1" / "raw_repeat_results.csv"
    size_path = imported_dir / "repeated_sizes" / "raw_repeat_sizes.csv"
    macro = pd.read_csv(macro_path)
    sizes = pd.read_csv(size_path)

    if classifiers != ["all"]:
        macro = macro[macro["classifier"].isin(classifiers)].copy()
        sizes = sizes[sizes["classifier"].isin(classifiers)].copy()

    for frame in [macro, sizes]:
        frame["source_dataset"] = frame["dataset"]
        frame["dataset"] = frame["dataset"].replace(DATASET_MAP)

    macro = macro[macro["dataset"].isin(configured_datasets)].copy()
    sizes = sizes[sizes["dataset"].isin(configured_datasets)].copy()

    merged = macro.merge(
        sizes[
            [
                "dataset",
                "classifier",
                "repeat",
                "split_seed",
                "n_train",
                "n_test",
                "n_features",
                "n_classes",
                "primary_metric",
                "primary_value",
                "secondary_metric",
                "secondary_value",
                "tertiary_metric",
                "tertiary_value",
                "avg_rrc_size",
                "max_rrc_size",
                "rrc_rc_ratio",
            ]
        ],
        on=["dataset", "classifier", "repeat", "split_seed"],
        how="left",
    )
    merged["method"] = merged["classifier"].map(METHOD_MAP).fillna(merged["classifier"])
    merged["metric"] = "imported_baseline"
    merged["k"] = np.nan
    merged["seed"] = merged["split_seed"].astype(int)
    merged["primary_f1"] = merged["macro_f1"]
    merged["best_params"] = merged["best_params"].map(parse_best_params)
    merged["source_classifier"] = merged["classifier"]

    keep = [
        "dataset",
        "seed",
        "repeat",
        "method",
        "metric",
        "k",
        "primary_f1",
        "macro_f1",
        "macro_f1_percent",
        "cv_macro_f1",
        "best_params",
        "elapsed_seconds",
        "status",
        "error",
        "source_classifier",
        "source_dataset",
        "n_train",
        "n_test",
        "n_features",
        "n_classes",
        "primary_metric",
        "primary_value",
        "secondary_metric",
        "secondary_value",
        "tertiary_metric",
        "tertiary_value",
        "avg_rrc_size",
        "max_rrc_size",
        "rrc_rc_ratio",
    ]
    return merged[[col for col in keep if col in merged.columns]].sort_values(["dataset", "method", "seed"])


def summarize_imported(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (dataset, method), df in rows.groupby(["dataset", "method"]):
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": "imported_baseline",
                "k": np.nan,
                "macro_f1_mean": float(df["macro_f1"].mean()),
                "macro_f1_std": float(df["macro_f1"].std(ddof=1)) if len(df) > 1 else 0.0,
                "macro_f1_ci95": ci95_half_width(df["macro_f1"]),
                "primary_f1_mean": float(df["primary_f1"].mean()),
                "primary_f1_std": float(df["primary_f1"].std(ddof=1)) if len(df) > 1 else 0.0,
                "primary_f1_ci95": ci95_half_width(df["primary_f1"]),
                "elapsed_seconds_mean": float(df["elapsed_seconds"].mean()),
                "runs": int(df["seed"].nunique()),
                "repeat_rows": int(len(df)),
                "source": "imported_baseline",
            }
        )
    return pd.DataFrame(summaries).sort_values(["dataset", "method"])


def build_paper_comparison(run_dir: Path, imported_summary: pd.DataFrame) -> pd.DataFrame:
    summary_path = run_dir / "summary_by_dataset_metric.csv"
    frames = []
    if summary_path.exists():
        ranking = pd.read_csv(summary_path)
        ranking = ranking[ranking["k"].notna()].copy()
        if not ranking.empty:
            best_idx = ranking.groupby(["dataset", "method", "metric"], dropna=False)["primary_f1_mean"].idxmax()
            best = ranking.loc[best_idx].copy()
            best["source"] = "ranked_lazyfca"
            best["macro_f1_ci95"] = best.get("primary_f1_ci95", np.nan)
            frames.append(best)
    frames.append(imported_summary.copy())
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["dataset", "source", "method", "metric"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import preserved macro-F1 baseline results for paper comparison.")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--imported-dir", default="experiments/imported_baselines")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=DEFAULT_CLASSIFIERS,
        help="Imported classifiers to keep, or 'all'. Default: fcalc fcalc_rand ips_knn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    imported_dir = ROOT / args.imported_dir
    run_dir = ROOT / args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configured_datasets = load_configured_datasets(config_path)
    rows = imported_rows(imported_dir, args.classifiers, configured_datasets)
    summary = summarize_imported(rows)
    comparison = build_paper_comparison(run_dir, summary)

    rows.to_csv(run_dir / "imported_baseline_results.csv", index=False)
    summary.to_csv(run_dir / "imported_baseline_summary.csv", index=False)
    comparison.to_csv(run_dir / "paper_comparison_macro_f1.csv", index=False)

    print(f"Imported baseline rows: {len(rows)}")
    print(f"Imported baseline summary rows: {len(summary)}")
    print(f"Paper comparison rows: {len(comparison)}")
    print(f"Wrote: {run_dir}")


if __name__ == "__main__":
    main()
