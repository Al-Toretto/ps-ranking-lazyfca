#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.import_baseline_results import build_paper_comparison
from experiments.import_baseline_results import imported_rows
from experiments.import_baseline_results import load_configured_datasets
from experiments.import_baseline_results import summarize_imported
from experiments.run_experiments import dataset_specs
from experiments.run_experiments import load_config
from experiments.run_experiments import prepare_dataset
from experiments.run_experiments import ranked_classifiers


DATASET_ORDER = [
    "breast_cancer",
    "ionosphere",
    "parkinsons",
    "rice",
    "sonar",
    "spambase",
    "waveform",
    "vehicle",
    "page_blocks",
    "glass",
    "image_segmentation",
]

SELECTED_METRICS = {
    "query_similarity": "locality representative",
    "query_weighted_precision": "primary reported metric",
    "precision": "purity-only reference",
    "description_volume": "geometric-size reference",
    "query_weighted_log_odds_ratio": "log-odds variant",
    "delta_stability": "stability-family diagnostic",
    "log_odds_ratio": "support-sensitive reference",
}


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in df.columns:
        return df
    data = df[df["dataset"].isin(DATASET_ORDER)].copy()
    data["dataset"] = pd.Categorical(data["dataset"], categories=DATASET_ORDER, ordered=True)
    sort_cols = [col for col in ["dataset", "method", "metric", "k", "seed", "repeat"] if col in data.columns]
    return data.sort_values(sort_cols).reset_index(drop=True)


def best_within_k(summary: pd.DataFrame, method: str, metric: str, max_k: int = 10) -> pd.DataFrame:
    subset = summary[
        (summary["method"] == method)
        & (summary["metric"] == metric)
        & (summary["k"].between(1, max_k))
    ].copy()
    idx = subset.groupby("dataset", observed=False)["primary_f1_mean"].idxmax()
    return subset.loc[idx].reset_index(drop=True)


def export_filtered_results(run_dir: Path, output_dir: Path, imported_dir: Path, config_path: Path) -> None:
    configured_datasets = load_configured_datasets(config_path).intersection(DATASET_ORDER)
    baseline_rows = imported_rows(imported_dir, ["all"], configured_datasets)
    baseline_summary = summarize_imported(baseline_rows)
    comparison = build_paper_comparison(run_dir, baseline_summary, configured_datasets)

    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows.to_csv(run_dir / "imported_baseline_results.csv", index=False)
    baseline_summary.to_csv(run_dir / "imported_baseline_summary.csv", index=False)
    comparison.to_csv(run_dir / "paper_comparison_macro_f1.csv", index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "summary_by_dataset_metric.csv",
        "compactness_summary.csv",
        "dataset_diagnostics.csv",
        "topk_plot_data.csv",
        "imported_baseline_results.csv",
        "imported_baseline_summary.csv",
        "paper_comparison_macro_f1.csv",
    ]:
        source = run_dir / filename
        if source.exists():
            ordered(pd.read_csv(source)).to_csv(output_dir / filename, index=False)


def export_metric_screening(summary: pd.DataFrame, output_dir: Path) -> None:
    compact = summary[(summary["method"] == "global_topk") & (summary["k"].between(1, 10))].copy()
    idx = compact.groupby(["dataset", "metric"], observed=False)["primary_f1_mean"].idxmax()
    best = compact.loc[idx].copy()
    best["rank"] = best.groupby("dataset", observed=False)["primary_f1_mean"].rank(
        method="min",
        ascending=False,
    )
    rows = []
    for metric, group in best.groupby("metric", observed=False):
        rows.append(
            {
                "metric": metric,
                "tied_wins": int((group["rank"] == 1).sum()),
                "top_3": int((group["rank"] <= 3).sum()),
                "mean_rank": float(group["rank"].mean()),
                "mean_best_macro_f1": float(group["primary_f1_mean"].mean()),
                "median_best_k": float(group["k"].median()),
                "role": SELECTED_METRICS.get(metric, "diagnostic"),
            }
        )
    screening = pd.DataFrame(rows).sort_values(["mean_rank", "metric"]).reset_index(drop=True)
    screening.to_csv(output_dir / "diagnostic_metric_screening_all_enabled_k1_10.csv", index=False)
    screening[screening["metric"].isin(SELECTED_METRICS)].to_csv(
        output_dir / "diagnostic_metric_screening_selected_metrics.csv",
        index=False,
    )


def export_fixed_k_table(summary: pd.DataFrame, output_dir: Path) -> None:
    qwp = summary[
        (summary["method"] == "global_topk")
        & (summary["metric"] == "query_weighted_precision")
        & (summary["k"].isin([1, 3, 5, 10]))
    ].copy()
    fixed = qwp.pivot(index="dataset", columns="k", values="primary_f1_mean").reset_index()
    fixed.columns = ["dataset", *[f"macro_f1_k_{int(col)}" for col in fixed.columns[1:]]]
    best = best_within_k(summary, "global_topk", "query_weighted_precision", max_k=10)
    best = best[["dataset", "k", "primary_f1_mean", "primary_f1_ci95"]].rename(
        columns={
            "k": "best_k_le_10",
            "primary_f1_mean": "best_macro_f1_le_10",
            "primary_f1_ci95": "best_macro_f1_le_10_ci95",
        }
    )
    table = ordered(fixed.merge(best, on="dataset", how="left"))
    means = {"dataset": "mean"}
    for col in table.columns:
        if col != "dataset":
            means[col] = table[col].mean()
    table = pd.concat([table, pd.DataFrame([means])], ignore_index=True)
    table.to_csv(output_dir / "table_qwp_fixed_k.csv", index=False)


def baseline_value(baselines: pd.DataFrame, dataset: str, method: str) -> dict[str, float]:
    row = baselines[(baselines["dataset"] == dataset) & (baselines["method"] == method)]
    if row.empty:
        return {"mean": np.nan, "ci95": np.nan}
    first = row.iloc[0]
    return {"mean": float(first["primary_f1_mean"]), "ci95": float(first["primary_f1_ci95"])}


def export_comparison_tables(summary: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> None:
    qwp = best_within_k(summary, "global_topk", "query_weighted_precision", max_k=10)
    random = best_within_k(summary, "random_topk", "random", max_k=10)
    rows = []
    for dataset in DATASET_ORDER:
        qwp_row = qwp[qwp["dataset"] == dataset].iloc[0]
        random_row = random[random["dataset"] == dataset].iloc[0]
        row = {
            "dataset": dataset,
            "best_k_le_10": int(qwp_row["k"]),
            "qwp_topk_mean": float(qwp_row["primary_f1_mean"]),
            "qwp_topk_ci95": float(qwp_row["primary_f1_ci95"]),
            "random_topk_mean": float(random_row["primary_f1_mean"]),
            "random_topk_ci95": float(random_row["primary_f1_ci95"]),
        }
        for method in ["fcalc_deterministic", "fcalc_randomized", "ips_knn"]:
            value = baseline_value(baselines, dataset, method)
            row[f"{method}_mean"] = value["mean"]
            row[f"{method}_ci95"] = value["ci95"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table_compact_budget_comparison.csv", index=False)

    rows = []
    for dataset in DATASET_ORDER:
        qwp_row = qwp[qwp["dataset"] == dataset].iloc[0]
        row = {
            "dataset": dataset,
            "qwp_topk_mean": float(qwp_row["primary_f1_mean"]),
            "qwp_topk_ci95": float(qwp_row["primary_f1_ci95"]),
        }
        for method in ["knn", "svm", "random_forest", "xgboost"]:
            value = baseline_value(baselines, dataset, method)
            row[f"{method}_mean"] = value["mean"]
            row[f"{method}_ci95"] = value["ci95"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table_classical_context.csv", index=False)


def export_compactness_table(summary: pd.DataFrame, output_dir: Path) -> None:
    qwp = summary[(summary["method"] == "global_topk") & (summary["metric"] == "query_weighted_precision")].copy()
    compact = qwp[qwp["k"].between(1, 10)].copy()
    best = best_within_k(summary, "global_topk", "query_weighted_precision", max_k=10)
    rows = []
    for dataset in DATASET_ORDER:
        d_best = best[best["dataset"] == dataset].iloc[0]
        d_compact = compact[compact["dataset"] == dataset].sort_values("k")
        row = {
            "dataset": dataset,
            "macro_f1_k_1": float(d_compact[d_compact["k"] == 1]["primary_f1_mean"].iloc[0]),
            "best_k_le_10": int(d_best["k"]),
            "best_macro_f1_le_10": float(d_best["primary_f1_mean"]),
            "best_macro_f1_le_10_ci95": float(d_best["primary_f1_ci95"]),
        }
        for pct in [1, 3, 5]:
            threshold = row["best_macro_f1_le_10"] * (1.0 - pct / 100.0)
            selected = d_compact[d_compact["primary_f1_mean"] >= threshold].sort_values("k").iloc[0]
            row[f"k_within_{pct}pct"] = int(selected["k"])
            row[f"macro_f1_within_{pct}pct"] = float(selected["primary_f1_mean"])
            row[f"compression_ratio_within_{pct}pct"] = float(selected["compression_ratio_mean"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table_qwp_compactness.csv", index=False)


def clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def export_rice_example(config_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    spec = dataset_specs(config, ["rice"])[0]
    payload = prepare_dataset(config, spec, 1998)
    query_index = 122
    sample = payload["X_test"].iloc[query_index]
    explanation = payload["model"].explain_sample(sample)
    ranked = ranked_classifiers(explanation, "query_weighted_precision")[:5]

    candidate_rows = []
    for rank, (class_index, source_index, classifier, score) in enumerate(ranked, start=1):
        candidate_rows.append(
            {
                "rank": rank,
                "candidate_class": class_index,
                "source_index_within_class": source_index,
                "tp": int(classifier.metrics.get_metric("tp")),
                "fp": int(classifier.metrics.get_metric("fp")),
                "query_similarity": float(classifier.metrics.get_metric("query_similarity")),
                "query_weighted_precision": float(score),
            }
        )
    pd.DataFrame(candidate_rows).to_csv(
        output_dir / "retained_candidates_rice_seed1998_query122_top5.csv",
        index=False,
    )

    _class_index, _source_index, classifier, _score = ranked[4]
    interval_rows = []
    for idx, feature in enumerate(payload["X_test"].columns):
        interval_rows.append(
            {
                "feature": clean_feature_name(feature),
                "query_value": float(sample.iloc[idx]),
                "source_value": float(classifier.source.numeric[idx]),
                "interval_min": float(classifier.numeric_minimum[idx]),
                "interval_max": float(classifier.numeric_maximum[idx]),
            }
        )
    pd.DataFrame(interval_rows).to_csv(
        output_dir / "retained_candidate_rice_seed1998_query122_rank5_interval.csv",
        index=False,
    )

    metadata = {
        "dataset": "rice",
        "seed": 1998,
        "query_index": query_index,
        "true_class": int(payload["y_test"][query_index]),
        "metric": "query_weighted_precision",
        "k": 5,
        "predicted_class": int(np.bincount([row["candidate_class"] for row in candidate_rows], minlength=2).argmax()),
    }
    (output_dir / "retained_candidate_example_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, run_dir: Path) -> None:
    metadata = {
        "source_run_dir": str(run_dir),
        "datasets": DATASET_ORDER,
        "seeds": list(range(1998, 2008)),
        "primary_metric": "macro-F1",
        "ranking_metric_reported_in_main_tables": "query_weighted_precision",
        "compact_budget": "k <= 10",
    }
    (output_dir / "artifact_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact public result artifacts for the manuscript.")
    parser.add_argument("--config", type=Path, default=Path("experiments/config.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/results/ranking_macro_f1_10splits"))
    parser.add_argument("--imported-dir", type=Path, default=Path("experiments/imported_baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--skip-example", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_filtered_results(args.run_dir, args.output_dir, args.imported_dir, args.config)
    summary = ordered(pd.read_csv(args.output_dir / "summary_by_dataset_metric.csv"))
    baselines = ordered(pd.read_csv(args.output_dir / "imported_baseline_summary.csv"))

    export_metric_screening(summary, args.output_dir)
    export_fixed_k_table(summary, args.output_dir)
    export_comparison_tables(summary, baselines, args.output_dir)
    export_compactness_table(summary, args.output_dir)
    if not args.skip_example:
        export_rice_example(args.config, args.output_dir)
    write_manifest(args.output_dir, args.run_dir)
    print(f"Wrote paper artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
