#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ranked_lazyfca_matplotlib")

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "Missing dependency: PyYAML. Install project requirements or run: "
        "python3 -m pip install PyYAML"
    ) from exc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

try:
    from lazyfca import LazyFCA
    from lazyfca.metrics import METADATA_DICT
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    if exc.name == "numba":
        raise SystemExit(
            "Missing dependency: numba. Install project requirements before running LazyFCA experiments."
        ) from exc
    raise

from baselines.interval_lazy_methods import FCalcBaseline
from baselines.interval_lazy_methods import IPSKNNBaseline


PRIMARY_METRIC = "primary_f1"
PRIMARY_METRIC_LABEL = "macro-F1"
VANILLA_METRIC = "all"
RANDOM_METRIC = "random"
BASELINE_METRIC = "baseline"
CURRENT_VANILLA_METHOD = "current_vanilla_lazyfca"
LEGACY_VANILLA_METHOD = "vanilla_lazyfca"
BASELINE_METHODS = {"fcalc_deterministic", "fcalc_randomized", "ips_knn"}
LAZY_METHODS = {CURRENT_VANILLA_METHOD, LEGACY_VANILLA_METHOD, "global_topk", "random_topk"}

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


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    target: str
    drop_columns: tuple[str, ...] = ()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    return config


def enabled_names(section: dict) -> list[str]:
    names = []
    for name, value in section.items():
        enabled = True
        if isinstance(value, dict):
            enabled = bool(value.get("enabled", True))
        elif isinstance(value, bool):
            enabled = value
        if enabled:
            names.append(name)
    return names


def configured_names(section: dict) -> list[str]:
    return list(section)


def selected_names(config_names: list[str], cli_names: typing.Optional[list[str]]) -> list[str]:
    if not cli_names:
        return config_names
    requested = set(cli_names)
    missing = sorted(requested.difference(config_names))
    if missing:
        raise ValueError(f"Requested unknown or disabled names: {missing}")
    return [name for name in config_names if name in requested]


def dataset_specs(config: dict, names: list[str]) -> list[DatasetSpec]:
    specs = []
    for name in names:
        raw = config["datasets"][name]
        specs.append(
            DatasetSpec(
                name=name,
                path=ROOT / raw["path"],
                target=raw["target"],
                drop_columns=tuple(raw.get("drop_columns", []) or []),
            )
        )
    return specs


def one_hot_encoder():
    try:
        return sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", dtype=bool, sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", dtype=bool, sparse=False)


def preprocess_dataset(spec: DatasetSpec, test_size: float, seed: int):
    df = pd.read_csv(spec.path)
    missing = [col for col in [spec.target, *spec.drop_columns] if col not in df.columns]
    if missing:
        raise ValueError(f"{spec.name}: missing columns {missing}")

    y_raw = df[spec.target]
    X = df.drop(columns=[spec.target, *spec.drop_columns])
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train_raw, X_test_raw, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    numeric_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X_train_raw.columns if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", one_hot_encoder(), categorical_cols))

    if transformers:
        preprocessor = sklearn.compose.ColumnTransformer(transformers=transformers)
        X_train_arr = preprocessor.fit_transform(X_train_raw)
        X_test_arr = preprocessor.transform(X_test_raw)
        columns = preprocessor.get_feature_names_out()
        X_train = pd.DataFrame(X_train_arr, columns=columns, index=X_train_raw.index)
        X_test = pd.DataFrame(X_test_arr, columns=columns, index=X_test_raw.index)
    else:
        X_train = pd.DataFrame(index=X_train_raw.index)
        X_test = pd.DataFrame(index=X_test_raw.index)

    categorical_features = [col for col in X_train.columns if col.startswith("categorical__")]
    if categorical_features:
        X_train[categorical_features] = X_train[categorical_features].astype(bool)
        X_test[categorical_features] = X_test[categorical_features].astype(bool)

    for col in X_train.columns:
        if col not in categorical_features:
            X_train[col] = pd.to_numeric(X_train[col], errors="raise").astype(float)
            X_test[col] = pd.to_numeric(X_test[col], errors="raise").astype(float)

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": pd.Series(y_train).reset_index(drop=True),
        "y_test": np.asarray(y_test, dtype=int),
        "label_names": [str(label) for label in label_encoder.classes_],
        "n_classes": int(len(label_encoder.classes_)),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "encoded_feature_count": int(X_train.shape[1]),
    }


def stable_json(data: typing.Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(data: typing.Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()[:12]


def dataset_fingerprint(spec: DatasetSpec, test_size: float, seed: int) -> str:
    stat = spec.path.stat()
    return short_hash(
        {
            "name": spec.name,
            "path": str(spec.path.relative_to(ROOT)),
            "path_size": stat.st_size,
            "path_mtime_ns": stat.st_mtime_ns,
            "target": spec.target,
            "drop_columns": spec.drop_columns,
            "test_size": test_size,
            "seed": seed,
        }
    )


def chunk_path(run_dir: Path, dataset: str, seed: int, method: str, metric: str) -> Path:
    return run_dir / "chunks" / f"{dataset}__seed{seed}__{method}__{metric}.csv"


def diagnostic_path(run_dir: Path, dataset: str, seed: int) -> Path:
    return run_dir / "diagnostics" / f"{dataset}__seed{seed}.csv"


def append_manifest(run_dir: Path, event: dict) -> None:
    path = run_dir / "manifest.jsonl"
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def prepare_dataset(config: dict, spec: DatasetSpec, seed: int) -> dict:
    data = preprocess_dataset(spec, float(config["test_size"]), seed)
    model = LazyFCA(n_classes=data["n_classes"])
    model.fit(data["X_train"], data["y_train"])

    return {
        "dataset": spec.name,
        "seed": seed,
        "test_size": float(config["test_size"]),
        "model": model,
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "X_train_shape": tuple(data["X_train"].shape),
        "X_test_shape": tuple(data["X_test"].shape),
        "y_train": data["y_train"].to_numpy(dtype=int),
        "y_test": data["y_test"],
        "label_names": data["label_names"],
        "n_classes": data["n_classes"],
        "numeric_feature_count": data["numeric_feature_count"],
        "categorical_feature_count": data["categorical_feature_count"],
        "encoded_feature_count": data["encoded_feature_count"],
        "class_priors": np.bincount(data["y_train"], minlength=data["n_classes"]).astype(float)
        / float(len(data["y_train"])),
    }


def flatten_classifiers(explanation) -> list[tuple[int, int, typing.Any]]:
    flat = []
    for class_index, classifiers in enumerate(explanation.class_classifiers):
        for source_index, classifier in enumerate(classifiers):
            flat.append((class_index, source_index, classifier))
    return flat


def safe_score(classifier, metric: str) -> float:
    value = classifier.metrics.score_for_ranking(metric)
    if value is None or np.isnan(value):
        return float("-inf")
    return float(value)


def ranked_classifiers(explanation, metric: str) -> list[tuple[int, int, typing.Any, float]]:
    rows = []
    for class_index, source_index, classifier in flatten_classifiers(explanation):
        score = safe_score(classifier, metric)
        query_similarity = classifier.metrics.get_metric("query_similarity")
        tp = classifier.metrics.get_metric("tp")
        fp = classifier.metrics.get_metric("fp")
        rows.append((class_index, source_index, classifier, score, query_similarity, tp, fp))

    rows.sort(
        key=lambda row: (
            -row[3],
            -(float(row[4]) if row[4] is not None and np.isfinite(row[4]) else float("-inf")),
            -(int(row[5]) if row[5] is not None else -1),
            int(row[6]) if row[6] is not None else 10**12,
            row[0],
            row[1],
        )
    )
    return [(class_index, source_index, classifier, score) for class_index, source_index, classifier, score, *_ in rows]


def k_values_from_config(k_grid: dict, full_k: int, smoke: bool) -> list[int]:
    if smoke:
        return [k for k in [1, 2, 5] if k <= full_k]

    values = [int(k) for k in k_grid.get("low", []) if int(k) >= 1]
    geometric = k_grid.get("geometric", {}) or {}
    start = int(geometric.get("start", 12))
    stop_raw = geometric.get("stop", "full")
    stop = full_k if stop_raw == "full" else int(stop_raw)
    num = int(geometric.get("num", 24))
    if full_k >= start and num > 0:
        values.extend(np.geomspace(start, min(stop, full_k), num=num).round().astype(int).tolist())
    values.append(full_k)
    return sorted(set(k for k in values if 1 <= k <= full_k))


def choose_class(counts: np.ndarray, score_sums: np.ndarray, priors: np.ndarray) -> int:
    tied = np.flatnonzero(counts == counts.max())
    if len(tied) == 1:
        return int(tied[0])
    tied_scores = score_sums[tied]
    tied = tied[np.flatnonzero(tied_scores == tied_scores.max())]
    if len(tied) == 1:
        return int(tied[0])
    tied_priors = priors[tied]
    tied = tied[np.flatnonzero(tied_priors == tied_priors.max())]
    return int(tied.min())


def predict_from_retained(
    retained: list[tuple[int, int, typing.Any, float]],
    n_classes: int,
    priors: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_classes, dtype=float)
    score_sums = np.zeros(n_classes, dtype=float)
    for class_index, _source_index, _classifier, score in retained:
        counts[class_index] += 1.0
        if np.isposinf(score):
            score_sums[class_index] = np.inf
        elif np.isfinite(score):
            score_sums[class_index] += score
    pred = choose_class(counts, score_sums, priors)
    proba = counts / counts.sum() if counts.sum() > 0 else priors.copy()
    return pred, proba, counts, score_sums


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def one_vs_rest_scores(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> tuple[float, float, float, int, int, int]:
    tp = int(((y_true == label) & (y_pred == label)).sum())
    fp = int(((y_true != label) & (y_pred == label)).sum())
    fn = int(((y_true == label) & (y_pred != label)).sum())
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1, tp, fp, fn


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    per_class = [one_vs_rest_scores(y_true, y_pred, label) for label in range(n_classes)]
    supports = np.bincount(y_true, minlength=n_classes).astype(float)
    f1_values = np.asarray([row[2] for row in per_class], dtype=float)
    precision_values = np.asarray([row[0] for row in per_class], dtype=float)
    recall_values = np.asarray([row[1] for row in per_class], dtype=float)
    macro_f1 = float(f1_values.mean()) if len(f1_values) else 0.0
    weighted_f1 = safe_divide(float((f1_values * supports).sum()), float(supports.sum()))
    precision = float(precision_values.mean()) if len(precision_values) else 0.0
    recall = float(recall_values.mean()) if len(recall_values) else 0.0
    primary_f1 = macro_f1
    return {
        "precision": precision,
        "recall": recall,
        "primary_f1": primary_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def prediction_score_matrix(y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    scores = np.zeros((len(y_pred), n_classes), dtype=float)
    for row_index, label in enumerate(y_pred):
        if 0 <= int(label) < n_classes:
            scores[row_index, int(label)] = 1.0
    return scores


def metric_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    metric: str,
    k: typing.Optional[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    total_available_mean: float,
    retained_mean: float,
    repeat: typing.Optional[int] = None,
    extra: typing.Optional[dict] = None,
) -> dict:
    n_classes = y_score.shape[1]
    labels = list(range(n_classes))
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = classification_scores(y_true, y_pred, n_classes)
    try:
        auc = (
            sklearn.metrics.roc_auc_score(y_true, y_score[:, 1])
            if n_classes == 2
            else sklearn.metrics.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        auc = float("nan")

    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
    row = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "metric": metric,
        "k": None if k is None or pd.isna(k) else int(k),
        "repeat": repeat,
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "primary_f1": scores["primary_f1"],
        "macro_f1": scores["macro_f1"],
        "weighted_f1": scores["weighted_f1"],
        "auc_roc": auc,
        "total_available_mean": total_available_mean,
        "retained_mean": retained_mean,
        "compression_ratio": retained_mean / total_available_mean if total_available_mean else float("nan"),
        "confusion_matrix": json.dumps(confusion.tolist()),
        "invalid_prediction_count": int((~np.isin(y_pred, labels)).sum()),
    }
    if n_classes == 2:
        _precision, _recall, _f1, tp, fp, fn = one_vs_rest_scores(y_true, y_pred, 1)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        row.update({"true_positive": int(tp), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn)})
    if extra:
        row.update(extra)
    return row


def enabled_work(run_dir: Path, dataset: str, seed: int, methods: list[str], metrics: list[str], force: bool) -> dict:
    work = {"vanilla": False, "vanilla_method": None, "global_metrics": [], "random": False, "baselines": []}
    vanilla_method = None
    if CURRENT_VANILLA_METHOD in methods:
        vanilla_method = CURRENT_VANILLA_METHOD
    elif LEGACY_VANILLA_METHOD in methods:
        vanilla_method = LEGACY_VANILLA_METHOD
    if vanilla_method is not None:
        path = chunk_path(run_dir, dataset, seed, vanilla_method, VANILLA_METRIC)
        if force or not path.exists():
            work["vanilla"] = True
            work["vanilla_method"] = vanilla_method
        else:
            append_manifest(
                run_dir,
                {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": vanilla_method, "metric": VANILLA_METRIC},
            )

    if "global_topk" in methods:
        for metric in metrics:
            path = chunk_path(run_dir, dataset, seed, "global_topk", metric)
            if force or not path.exists():
                work["global_metrics"].append(metric)
            else:
                append_manifest(
                    run_dir,
                    {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": "global_topk", "metric": metric},
                )

    if "random_topk" in methods:
        path = chunk_path(run_dir, dataset, seed, "random_topk", RANDOM_METRIC)
        if force or not path.exists():
            work["random"] = True
        else:
            append_manifest(
                run_dir,
                {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": "random_topk", "metric": RANDOM_METRIC},
            )

    for method in sorted(BASELINE_METHODS.intersection(methods)):
        path = chunk_path(run_dir, dataset, seed, method, BASELINE_METRIC)
        if force or not path.exists():
            work["baselines"].append(method)
        else:
            append_manifest(
                run_dir,
                {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": method, "metric": BASELINE_METRIC},
            )
    return work


def no_work(work: dict) -> bool:
    return not has_lazy_work(work) and not work["baselines"]


def has_lazy_work(work: dict) -> bool:
    return bool(work["vanilla"] or work["global_metrics"] or work["random"])


def make_prediction_store(
    work: dict,
    k_values: list[int],
    random_repeats: int,
    full_k: int,
) -> dict[tuple[str, str, int, typing.Optional[int]], dict[str, list]]:
    store = {}
    if work["vanilla"]:
        store[(work["vanilla_method"], VANILLA_METRIC, full_k, None)] = {
            "pred": [],
            "score": [],
            "retained": [],
            "available": [],
        }
    for metric in work["global_metrics"]:
        for k in k_values:
            store[("global_topk", metric, k, None)] = {"pred": [], "score": [], "retained": [], "available": []}
    if work["random"]:
        for repeat in range(random_repeats):
            for k in k_values:
                store[("random_topk", RANDOM_METRIC, k, repeat)] = {"pred": [], "score": [], "retained": [], "available": []}
    return store


def add_prediction(
    bucket: dict[str, list],
    pred: int,
    proba: np.ndarray,
    retained: int,
    available: int,
) -> None:
    bucket["pred"].append(pred)
    bucket["score"].append(proba)
    bucket["retained"].append(retained)
    bucket["available"].append(available)


def update_diagnostic_counts(explanation, diagnostic: dict) -> None:
    counts = [len(classifiers) for classifiers in explanation.class_classifiers]
    total_available = int(sum(counts))
    diagnostic["total_by_query"].append(total_available)
    for classifiers in explanation.class_classifiers:
        for classifier in classifiers:
            diagnostic["total_classifiers"] += 1
            if classifier.metrics.tp == 1:
                diagnostic["singleton"] += 1
            if classifier.metrics.fp == 0:
                diagnostic["fp_zero"] += 1


def build_streaming_diagnostics(payload: dict, diagnostic: dict) -> dict:
    train_counts = np.bincount(payload["y_train"], minlength=payload["n_classes"])
    test_counts = np.bincount(payload["y_test"], minlength=payload["n_classes"])
    total_by_query = diagnostic["total_by_query"] or [0]
    total = diagnostic["total_classifiers"]
    return {
        "dataset": payload["dataset"],
        "seed": payload["seed"],
        "n_classes": payload["n_classes"],
        "train_size": int(len(payload["y_train"])),
        "test_size": int(len(payload["y_test"])),
        "train_class_counts": json.dumps(train_counts.tolist()),
        "test_class_counts": json.dumps(test_counts.tolist()),
        "numeric_feature_count": payload["numeric_feature_count"],
        "categorical_feature_count": payload["categorical_feature_count"],
        "encoded_feature_count": payload["encoded_feature_count"],
        "total_classifiers_mean": float(np.mean(total_by_query)),
        "total_classifiers_min": int(np.min(total_by_query)),
        "total_classifiers_max": int(np.max(total_by_query)),
        "singleton_tp1_rate": diagnostic["singleton"] / total if total else float("nan"),
        "fp_zero_rate": diagnostic["fp_zero"] / total if total else float("nan"),
    }


def stream_evaluate_chunks(
    run_dir: Path,
    payload: dict,
    work: dict,
    k_values: list[int],
    random_repeats: int,
) -> dict:
    dataset = payload["dataset"]
    seed = int(payload["seed"])
    n_classes = payload["n_classes"]
    priors = payload["class_priors"]
    full_k = len(payload["y_train"])
    store = make_prediction_store(work, k_values, random_repeats, full_k)
    diagnostic = {"total_by_query": [], "total_classifiers": 0, "singleton": 0, "fp_zero": 0}

    for query_idx, (_row_idx, sample) in enumerate(payload["X_test"].iterrows()):
        explanation = payload["model"].explain_sample(sample)
        update_diagnostic_counts(explanation, diagnostic)
        available = sum(len(classifiers) for classifiers in explanation.class_classifiers)

        if work["vanilla"]:
            counts = np.asarray([len(classifiers) for classifiers in explanation.class_classifiers], dtype=float)
            pred = choose_class(counts, counts.copy(), priors)
            proba = counts / counts.sum() if counts.sum() > 0 else priors.copy()
            add_prediction(
                store[(work["vanilla_method"], VANILLA_METRIC, full_k, None)],
                pred,
                proba,
                int(counts.sum()),
                available,
            )

        for metric in work["global_metrics"]:
            ranked = ranked_classifiers(explanation, metric)
            for k in k_values:
                kept = ranked[: min(k, len(ranked))]
                pred, proba, _counts, _score_sums = predict_from_retained(kept, n_classes, priors)
                add_prediction(store[("global_topk", metric, k, None)], pred, proba, len(kept), available)

        if work["random"]:
            flat = flatten_classifiers(explanation)
            for repeat in range(random_repeats):
                rng = np.random.default_rng(seed * 1_000_003 + query_idx * 1_009 + repeat)
                order = rng.permutation(len(flat))
                ranked = [(flat[i][0], flat[i][1], flat[i][2], 1.0) for i in order]
                for k in k_values:
                    kept = ranked[: min(k, len(ranked))]
                    pred, proba, _counts, _score_sums = predict_from_retained(kept, n_classes, priors)
                    add_prediction(store[("random_topk", RANDOM_METRIC, k, repeat)], pred, proba, len(kept), available)

    by_chunk: dict[tuple[str, str], list[dict]] = {}
    for (method, metric, k, repeat), bucket in store.items():
        row = metric_row(
            dataset=dataset,
            seed=seed,
            method=method,
            metric=metric,
            k=k,
            repeat=repeat,
            y_true=payload["y_test"],
            y_pred=np.asarray(bucket["pred"], dtype=int),
            y_score=np.vstack(bucket["score"]),
            total_available_mean=float(np.mean(bucket["available"])),
            retained_mean=float(np.mean(bucket["retained"])),
        )
        by_chunk.setdefault((method, metric), []).append(row)

    for (method, metric), rows in by_chunk.items():
        path = chunk_path(run_dir, dataset, seed, method, metric)
        df = pd.DataFrame(rows).sort_values(["repeat", "k"], na_position="first")
        write_chunk(path, df)
        append_manifest(
            run_dir,
            {
                "type": "chunk",
                "status": "completed",
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "metric": metric,
                "rows": int(len(df)),
                "path": str(path.relative_to(run_dir)),
            },
        )

    return build_streaming_diagnostics(payload, diagnostic)


def method_options(config: dict, method: str) -> dict:
    raw = config.get("methods", {}).get(method, {}) or {}
    return raw if isinstance(raw, dict) else {}


def build_baseline(method: str, options: dict, seed: int, smoke: bool):
    if method == "fcalc_deterministic":
        return FCalcBaseline(
            randomized=False,
            scaler=options.get("scaler", "minmax"),
            tune_aggregation=bool(options.get("tune_aggregation", True)),
            families=list(options.get("families", ["standard", "proximity"])),
            family=options.get("family", "standard"),
            aggregation=options.get("aggregation", "standard-support"),
            seed=seed,
            cv_seed=int(options.get("cv_seed", 1998)),
        )
    if method == "fcalc_randomized":
        num_iters_grid = list(options.get("num_iters_grid", [10, 20, 30, 40, 50]))
        subsample_size_grid = list(options.get("subsample_size_grid", list(range(1, 11))))
        if smoke:
            num_iters_grid = num_iters_grid[:1]
            subsample_size_grid = subsample_size_grid[:2]
        return FCalcBaseline(
            randomized=True,
            scaler=options.get("scaler", "minmax"),
            tune_aggregation=bool(options.get("tune_aggregation", True)),
            families=list(options.get("families", ["standard", "proximity"])),
            family=options.get("family", "standard"),
            aggregation=options.get("aggregation", "standard-support"),
            num_iters=int(options.get("num_iters", 20)),
            subsample_size=int(options.get("subsample_size", 5)),
            num_iters_grid=num_iters_grid,
            subsample_size_grid=subsample_size_grid,
            seed=seed,
            cv_seed=int(options.get("cv_seed", 1998)),
        )
    if method == "ips_knn":
        return IPSKNNBaseline(
            scaler=options.get("scaler", "standard"),
            tune_k=bool(options.get("tune_k", True)),
            k_values=list(options.get("k_values", [1, 3, 5, 7, 9, 15, 25, 51])),
            k=int(options.get("k", 3)),
            p=int(options.get("p", 2)),
            weights=options.get("weights", "distance"),
            seed=seed,
            cv_seed=int(options.get("cv_seed", 1998)),
        )
    raise ValueError(f"Unknown baseline method: {method}")


def run_baseline_chunks(
    run_dir: Path,
    config: dict,
    payload: dict,
    work: dict,
    *,
    smoke: bool,
) -> None:
    dataset = payload["dataset"]
    seed = int(payload["seed"])
    y_true = np.asarray(payload["y_test"], dtype=int)
    n_classes = int(payload["n_classes"])

    for method in work["baselines"]:
        options = method_options(config, method)
        if bool(options.get("numeric_only", False)) and int(payload["categorical_feature_count"]) > 0:
            append_manifest(
                run_dir,
                {
                    "type": "chunk",
                    "status": "skipped_numeric_only",
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "metric": BASELINE_METRIC,
                    "categorical_feature_count": int(payload["categorical_feature_count"]),
                },
            )
            print(f"[skip]  {dataset} seed={seed}: {method} is numeric-only", flush=True)
            continue

        started = time.time()
        baseline = build_baseline(method, options, seed=seed, smoke=smoke)
        baseline.fit(payload["X_train"], payload["y_train"])
        y_pred = np.asarray(baseline.predict(payload["X_test"]), dtype=int)
        y_score = prediction_score_matrix(y_pred, n_classes)
        timings = baseline.get_timings()
        compactness = baseline.get_compactness()
        invalid_count = int((~np.isin(y_pred, list(range(n_classes)))).sum())
        extra = {
            "selected_params": json.dumps(baseline.get_params_used(), sort_keys=True),
            "runtime_seconds": round(time.time() - started, 6),
            "fit_seconds": timings.get("fit_seconds", 0.0),
            "predict_seconds": timings.get("predict_seconds", 0.0),
            "unclassified_count": invalid_count,
            "unclassified_rate": invalid_count / len(y_pred) if len(y_pred) else float("nan"),
            **compactness,
        }
        row = metric_row(
            dataset=dataset,
            seed=seed,
            method=method,
            metric=BASELINE_METRIC,
            k=None,
            repeat=None,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            total_available_mean=float("nan"),
            retained_mean=float("nan"),
            extra=extra,
        )
        path = chunk_path(run_dir, dataset, seed, method, BASELINE_METRIC)
        write_chunk(path, pd.DataFrame([row]))
        append_manifest(
            run_dir,
            {
                "type": "chunk",
                "status": "completed",
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "metric": BASELINE_METRIC,
                "rows": 1,
                "path": str(path.relative_to(run_dir)),
            },
        )


def write_chunk(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_diagnostic(run_dir: Path, diagnostic: dict) -> None:
    path = diagnostic_path(run_dir, str(diagnostic["dataset"]), int(diagnostic["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame([diagnostic]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def combine_chunks(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((run_dir / "chunks").glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def combine_diagnostics(run_dir: Path) -> pd.DataFrame:
    frames = []
    diagnostics_path = run_dir / "dataset_diagnostics.csv"
    if diagnostics_path.exists():
        frames.append(pd.read_csv(diagnostics_path))
    for path in sorted((run_dir / "diagnostics").glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["dataset", "seed"], keep="last")
        .sort_values(["dataset", "seed"])
    )


def ci95_half_width(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n <= 1:
        return 0.0
    df = n - 1
    t_value = T_CRITICAL_95.get(df, 1.96)
    return float(t_value * clean.std(ddof=1) / math.sqrt(n))


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    vanilla = results[results["method"].isin([CURRENT_VANILLA_METHOD, LEGACY_VANILLA_METHOD])].copy()
    topk = results[~results["method"].isin([CURRENT_VANILLA_METHOD, LEGACY_VANILLA_METHOD])].copy()
    group_cols = ["dataset", "method", "metric", "k"]
    if topk.empty:
        summary = pd.DataFrame()
    else:
        seed_level = (
            topk.groupby([*group_cols, "seed"], dropna=False)
            .agg(
                primary_f1=(PRIMARY_METRIC, "mean"),
                accuracy=("accuracy", "mean"),
                macro_f1=("macro_f1", "mean"),
                weighted_f1=("weighted_f1", "mean"),
                retained_mean=("retained_mean", "mean"),
                compression_ratio=("compression_ratio", "mean"),
                repeat_rows=("primary_f1", "count"),
            )
            .reset_index()
        )
        rows = []
        for keys, df in seed_level.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            for source, prefix in [
                ("primary_f1", "primary_f1"),
                ("accuracy", "accuracy"),
                ("macro_f1", "macro_f1"),
                ("weighted_f1", "weighted_f1"),
                ("retained_mean", "retained"),
                ("compression_ratio", "compression_ratio"),
            ]:
                row[f"{prefix}_mean"] = float(df[source].mean())
                row[f"{prefix}_std"] = float(df[source].std(ddof=1)) if len(df) > 1 else 0.0
                row[f"{prefix}_ci95"] = ci95_half_width(df[source])
            row["runs"] = int(df["seed"].nunique())
            row["repeat_rows"] = int(df["repeat_rows"].sum())
            rows.append(row)
        summary = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    if summary.empty:
        return vanilla, summary, pd.DataFrame()

    compact_rows = []
    for (dataset, method, metric), df in summary.groupby(["dataset", "method", "metric"], dropna=False):
        df = df.sort_values("k")
        best_idx = df["primary_f1_mean"].idxmax()
        best = df.loc[best_idx]
        best_k = None if pd.isna(best["k"]) else int(best["k"])
        row = {
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "best_k": best_k,
            "best_primary_f1_mean": float(best["primary_f1_mean"]),
            "best_primary_f1_ci95": float(best.get("primary_f1_ci95", 0.0)),
            "best_retained_mean": float(best["retained_mean"]),
            "best_retained_ci95": float(best.get("retained_ci95", 0.0)),
            "best_compression_ratio_mean": float(best["compression_ratio_mean"]),
            "best_compression_ratio_ci95": float(best.get("compression_ratio_ci95", 0.0)),
        }
        if best_k is None:
            for pct in [1, 3, 5]:
                row[f"smallest_k_within_{pct}pct"] = np.nan
                row[f"primary_f1_within_{pct}pct"] = float(best["primary_f1_mean"])
            compact_rows.append(row)
            continue
        for pct in [1, 3, 5]:
            threshold = row["best_primary_f1_mean"] * (1.0 - pct / 100.0)
            eligible = df[df["primary_f1_mean"] >= threshold].sort_values("k")
            if eligible.empty:
                row[f"smallest_k_within_{pct}pct"] = np.nan
                row[f"primary_f1_within_{pct}pct"] = np.nan
            else:
                selected = eligible.iloc[0]
                row[f"smallest_k_within_{pct}pct"] = int(selected["k"])
                row[f"primary_f1_within_{pct}pct"] = float(selected["primary_f1_mean"])
        compact_rows.append(row)
    compact = pd.DataFrame(compact_rows).sort_values(["dataset", "best_primary_f1_mean"], ascending=[True, False])
    return vanilla, summary, compact


def write_plots(run_dir: Path, summary: pd.DataFrame, vanilla: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot_dir = run_dir / "plots"
    metric_plot_dir = plot_dir / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset, df in summary.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for (method, metric), sub in df.groupby(["method", "metric"], dropna=False):
            sub = sub.sort_values("k")
            label = f"{method}:{metric}"
            if sub["k"].notna().any():
                ax.plot(sub["k"], sub["primary_f1_mean"], marker="o", linewidth=1.5, markersize=3, label=label)
            else:
                ax.axhline(sub["primary_f1_mean"].mean(), linestyle=":", linewidth=1.2, label=label)
        vanilla_sub = vanilla[vanilla["dataset"] == dataset]
        if not vanilla_sub.empty:
            ax.axhline(
                vanilla_sub[PRIMARY_METRIC].mean(),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="vanilla LazyFCA",
            )
        ax.set_xscale("log")
        ax.set_xlabel("k retained classifiers")
        ax.set_ylabel(PRIMARY_METRIC_LABEL)
        ax.set_title(f"{dataset}: compactness-first top-k ranking")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_topk_primary_f1.png", dpi=180)
        plt.close(fig)

        vanilla_sub = vanilla[vanilla["dataset"] == dataset]
        vanilla_f1 = None if vanilla_sub.empty else float(vanilla_sub[PRIMARY_METRIC].mean())
        for (method, metric), sub in df.groupby(["method", "metric"], dropna=False):
            sub = sub.sort_values("k")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if sub["k"].notna().any():
                ax.plot(
                    sub["k"],
                    sub["primary_f1_mean"],
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                    label=f"{method}:{metric}",
                )
            else:
                ax.axhline(sub["primary_f1_mean"].mean(), linestyle=":", linewidth=1.8, label=f"{method}:{metric}")
            if "primary_f1_ci95" in sub and sub["primary_f1_ci95"].notna().any():
                lower = sub["primary_f1_mean"] - sub["primary_f1_ci95"].fillna(0.0)
                upper = sub["primary_f1_mean"] + sub["primary_f1_ci95"].fillna(0.0)
                if sub["k"].notna().any():
                    ax.fill_between(sub["k"], lower, upper, alpha=0.15)
            if vanilla_f1 is not None:
                ax.axhline(
                    vanilla_f1,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="vanilla LazyFCA",
                )
            if sub["k"].notna().any():
                ax.set_xscale("log")
            ax.set_xlabel("k retained classifiers")
            ax.set_ylabel(PRIMARY_METRIC_LABEL)
            ax.set_title(f"{dataset}: {method}:{metric}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(metric_plot_dir / f"{dataset}__{method}__{metric}_topk_primary_f1.png", dpi=180)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incremental experiments for ranked LazyFCA.")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    config = load_config(config_path)
    if config.get("cache_explanations", False):
        raise ValueError(
            "cache_explanations=true is disabled because full LazyFCA explanations can require excessive RAM. "
            "Use the default streaming evaluator instead."
        )
    if args.smoke:
        config = {**config, "seeds": [0], "run_name": "smoke"}

    run_name = args.run_name or config.get("run_name", "default")
    output_dir = ROOT / config.get("output_dir", "experiments/results")
    run_dir = output_dir / run_name
    for subdir in ["chunks", "diagnostics", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset_names = selected_names(enabled_names(config["datasets"]), args.datasets)
    metric_names = selected_names(enabled_names(config["metrics"]), args.metrics)
    unknown_metrics = sorted(set(metric_names).difference(METADATA_DICT))
    if unknown_metrics:
        raise ValueError(f"Unknown LazyFCA metric names in config/CLI: {unknown_metrics}")
    method_config_names = configured_names(config["methods"])
    method_names = (
        selected_names(method_config_names, args.methods)
        if args.methods
        else enabled_names(config["methods"])
    )
    seeds = args.seeds if args.seeds is not None else list(config.get("seeds", [0, 1, 2, 3, 4]))
    force = bool(args.force or config.get("force", False))
    random_repeats = int(config.get("random_topk_repeats", 5))

    if args.smoke:
        dataset_names = dataset_names[:1]
        metric_names = metric_names[: min(2, len(metric_names))]
        random_repeats = 1

    config_fingerprint = short_hash(
        {
            "config_path": str(config_path.relative_to(ROOT)),
            "datasets": dataset_names,
            "metrics": metric_names,
            "methods": method_names,
            "seeds": seeds,
            "test_size": config.get("test_size"),
            "k_grid": config.get("k_grid"),
            "random_topk_repeats": random_repeats,
            "method_options": {name: method_options(config, name) for name in method_names},
        }
    )
    append_manifest(
        run_dir,
        {
            "type": "run",
            "status": "started",
            "config_fingerprint": config_fingerprint,
            "datasets": dataset_names,
            "metrics": metric_names,
            "methods": method_names,
            "seeds": seeds,
            "force": force,
            "smoke": bool(args.smoke),
        },
    )

    diagnostics = []
    for spec in dataset_specs(config, dataset_names):
        for seed in seeds:
            work = enabled_work(run_dir, spec.name, seed, method_names, metric_names, force=force)
            payload = prepare_dataset(config, spec, seed)
            full_k = len(payload["y_train"])
            k_values = k_values_from_config(config.get("k_grid", {}), full_k=full_k, smoke=bool(args.smoke))
            lazy_methods_selected = bool(set(method_names).intersection(LAZY_METHODS))
            diagnostics_missing = force or not diagnostic_path(run_dir, spec.name, seed).exists()
            diagnostic_required = diagnostics_missing and lazy_methods_selected
            if no_work(work) and not diagnostic_required:
                print(f"[skip]  {spec.name} seed={seed}: all chunks already exist", flush=True)
                continue

            started = time.time()
            if has_lazy_work(work) or diagnostic_required:
                mode = "diagnostic" if not has_lazy_work(work) else "stream"
                print(
                    f"[{mode}] {spec.name} seed={seed} "
                    f"metrics={len(work['global_metrics'])} k={len(k_values)}",
                    flush=True,
                )
                diagnostic = stream_evaluate_chunks(
                    run_dir,
                    payload,
                    work=work,
                    k_values=k_values,
                    random_repeats=random_repeats,
                )
                write_diagnostic(run_dir, diagnostic)
                diagnostics.append(diagnostic)

            if work["baselines"]:
                print(
                    f"[baseline] {spec.name} seed={seed} methods={','.join(work['baselines'])}",
                    flush=True,
                )
                run_baseline_chunks(run_dir, config, payload, work, smoke=bool(args.smoke))

            append_manifest(
                run_dir,
                {
                    "type": "dataset_seed",
                    "status": "completed",
                    "dataset": spec.name,
                    "seed": seed,
                    "seconds": round(time.time() - started, 3),
                    "streaming": bool(has_lazy_work(work) or diagnostic_required),
                    "diagnostics_only": bool(diagnostic_required and not has_lazy_work(work)),
                    "baseline_methods": work["baselines"],
                },
            )

    diagnostics_df = combine_diagnostics(run_dir)
    if not diagnostics_df.empty:
        diagnostics_df.to_csv(run_dir / "dataset_diagnostics.csv", index=False)

    results = combine_chunks(run_dir)
    if not results.empty:
        results.to_csv(run_dir / "topk_results.csv", index=False)
        vanilla, summary, compact = summarize_results(results)
        vanilla.to_csv(run_dir / "vanilla_lazyfca.csv", index=False)
        results[results["method"].isin(BASELINE_METHODS)].to_csv(run_dir / "baseline_results.csv", index=False)
        summary.to_csv(run_dir / "summary_by_dataset_metric.csv", index=False)
        plot_columns = [
            "dataset",
            "method",
            "metric",
            "k",
            "primary_f1_mean",
            "primary_f1_std",
            "primary_f1_ci95",
            "accuracy_mean",
            "accuracy_ci95",
            "macro_f1_mean",
            "macro_f1_ci95",
            "weighted_f1_mean",
            "weighted_f1_ci95",
            "retained_mean",
            "retained_ci95",
            "compression_ratio_mean",
            "compression_ratio_ci95",
            "runs",
            "repeat_rows",
        ]
        summary[[col for col in plot_columns if col in summary.columns]].to_csv(
            run_dir / "topk_plot_data.csv",
            index=False,
        )
        compact.to_csv(run_dir / "compactness_summary.csv", index=False)
        write_plots(run_dir, summary, vanilla)

    append_manifest(run_dir, {"type": "run", "status": "completed", "config_fingerprint": config_fingerprint})
    print(f"Done. Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
