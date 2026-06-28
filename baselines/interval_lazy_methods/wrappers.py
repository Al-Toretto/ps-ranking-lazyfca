from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from . import fcalc
from .ips_knn import IPSKNNClassifier


STANDARD_METHODS = ["standard", "standard-support", "ratio-support"]
PROXIMITY_METHODS = ["proximity", "proximity-non-falsified", "proximity-support"]


def _as_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def _as_series(y: pd.Series | np.ndarray | list) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(y).reset_index(drop=True)


def _macro_f1_safe(y_true, y_pred) -> float:
    return f1_score(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), average="macro", zero_division=0)


def _cv_splitter(y, seed: int) -> StratifiedKFold:
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for baseline cross-validation")
    return StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)


def _can_sample(y, subsample_size: int) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return bool(counts.min() >= subsample_size)


def _can_sample_cv(y, subsample_size: int, seed: int) -> bool:
    splitter = _cv_splitter(y, seed)
    for train_index, _valid_index in splitter.split(np.zeros_like(y), y):
        if not _can_sample(y[train_index], subsample_size):
            return False
    return True


def _methods_for_families(families: list[str]) -> list[tuple[str, str]]:
    configs: list[tuple[str, str]] = []
    if "standard" in families:
        configs.extend(("standard", method) for method in STANDARD_METHODS)
    if "proximity" in families:
        configs.extend(("proximity", method) for method in PROXIMITY_METHODS)
    unknown = sorted(set(families).difference({"standard", "proximity"}))
    if unknown:
        raise ValueError(f"Unknown FCALC families: {unknown}")
    return configs


def predict_standard(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    *,
    randomize: bool = False,
    num_iters: int = 10,
    subsample_size: int = 1,
    seed: int = 42,
) -> np.ndarray:
    classifier = fcalc.classifier.PatternClassifier(
        x_train,
        y_train,
        method=method,
        randomize=randomize,
        seed=seed,
        num_iters=num_iters,
        subsample_size=subsample_size,
    )
    classifier.predict(x_test)
    return np.asarray(classifier.predictions)


def _support_proximity(
    context: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
    *,
    num_iters: int | None = None,
    subsample_size: int | None = None,
    seed: int = 42,
):
    classes = np.unique(labels)
    class_lengths = np.array([len(context[labels == class_label]) for class_label in classes])
    support = []
    proximity = []
    rng = np.random.default_rng(seed=seed)

    for class_label in classes:
        train_pos = context[labels == class_label]
        train_neg = context[labels != class_label]

        if num_iters is None:
            sampled_groups = train_pos[:, np.newaxis, :]
        else:
            if subsample_size is None:
                raise ValueError("subsample_size must be provided when num_iters is provided")
            sampled_groups = np.zeros((num_iters, subsample_size, context.shape[1]))
            for group_idx in range(num_iters):
                sampled_groups[group_idx] = rng.choice(train_pos, size=subsample_size, replace=False, shuffle=True)

        positive_support = np.zeros((len(test), len(sampled_groups)))
        positive_counter = np.zeros((len(test), len(sampled_groups)))
        positive_proximity = np.zeros((len(test), len(sampled_groups)))

        for test_idx in range(len(test)):
            for group_idx, group in enumerate(sampled_groups):
                low = np.minimum(test[test_idx], np.min(group, axis=0))
                high = np.maximum(test[test_idx], np.max(group, axis=0))
                pos_mask = (~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0
                cnt_mask = (~((low <= train_neg) & (train_neg <= high))).sum(axis=1) == 0
                positive_proximity[test_idx][group_idx] = (
                    1
                    - np.linalg.norm(train_pos[pos_mask] - test[test_idx], axis=1).mean()
                    / np.sqrt(context.shape[1])
                )
                positive_support[test_idx][group_idx] = pos_mask.sum()
                positive_counter[test_idx][group_idx] = cnt_mask.sum()

        support.append(np.array((positive_support, positive_counter)))
        proximity.append(positive_proximity)

    return support, proximity, classes, class_lengths


def _proximity_based(proximity, support, classes) -> np.ndarray:
    predictions = np.full(proximity[0].shape[0], -1.0)
    criteria = np.zeros((len(classes), proximity[0].shape[0]))
    for class_idx in range(len(classes)):
        criteria[class_idx] = proximity[class_idx].mean(axis=1)
    criteria = criteria.T
    pred_mask = (np.max(criteria, axis=1)[:, None] == criteria).sum(axis=-1) < 2
    predictions[pred_mask] = classes[np.argmax(criteria[pred_mask], axis=-1)]
    return predictions


def _proximity_non_falsified(proximity, support, classes) -> np.ndarray:
    predictions = np.full(proximity[0].shape[0], -1.0)
    criteria = np.zeros((len(classes), proximity[0].shape[0]))
    for class_idx in range(len(classes)):
        criteria[class_idx] = (proximity[class_idx] * (support[class_idx][1] == 0)).mean(axis=1)
    criteria = criteria.T
    pred_mask = (np.max(criteria, axis=1)[:, None] == criteria).sum(axis=-1) < 2
    predictions[pred_mask] = classes[np.argmax(criteria[pred_mask], axis=-1)]
    return predictions


def _proximity_support(proximity, support, classes, class_lengths) -> np.ndarray:
    predictions = np.full(proximity[0].shape[0], -1.0)
    criteria = np.zeros((len(classes), proximity[0].shape[0]))
    for class_idx in range(len(classes)):
        criteria[class_idx] = (
            support[class_idx][0] * proximity[class_idx] * (support[class_idx][1] == 0)
        ).sum(axis=1)
    criteria = criteria.T / class_lengths
    pred_mask = (np.max(criteria, axis=1)[:, None] == criteria).sum(axis=-1) < 2
    predictions[pred_mask] = classes[np.argmax(criteria[pred_mask], axis=-1)]
    return predictions


def predict_proximity(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
    *,
    num_iters: int | None = None,
    subsample_size: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    support, proximity, classes, class_lengths = _support_proximity(
        x_train,
        y_train,
        x_test,
        num_iters=num_iters,
        subsample_size=subsample_size,
        seed=seed,
    )
    if method == "proximity":
        return _proximity_based(proximity, support, classes)
    if method == "proximity-non-falsified":
        return _proximity_non_falsified(proximity, support, classes)
    if method == "proximity-support":
        return _proximity_support(proximity, support, classes, class_lengths)
    raise ValueError(f"Unknown proximity method: {method}")


def _predict_fcalc_config(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    family = config["family"]
    method = config["aggregation"]
    if family == "standard":
        return predict_standard(
            x_train,
            y_train,
            x_test,
            method,
            randomize=bool(config.get("randomized", False)),
            num_iters=int(config.get("num_iters", 10)),
            subsample_size=int(config.get("subsample_size", 1)),
            seed=seed,
        )
    if family == "proximity":
        return predict_proximity(
            x_train,
            y_train,
            x_test,
            method,
            num_iters=int(config["num_iters"]) if config.get("randomized", False) else None,
            subsample_size=int(config["subsample_size"]) if config.get("randomized", False) else None,
            seed=seed,
        )
    raise ValueError(f"Unknown FCALC family: {family}")


def _score_fcalc_config(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    *,
    cv_seed: int,
    seed: int,
) -> float:
    scores = []
    splitter = _cv_splitter(y_train, cv_seed)
    for train_index, valid_index in splitter.split(x_train, y_train):
        predictions = _predict_fcalc_config(
            x_train[train_index],
            y_train[train_index],
            x_train[valid_index],
            config,
            seed=seed,
        )
        scores.append(_macro_f1_safe(y_train[valid_index], predictions))
    return float(np.mean(scores))


def _tune_deterministic_fcalc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    families: list[str],
    cv_seed: int,
    seed: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for family, method in _methods_for_families(families):
        config = {"family": family, "aggregation": method, "randomized": False}
        cv_f1 = _score_fcalc_config(x_train, y_train, config, cv_seed=cv_seed, seed=seed)
        candidate = {**config, "cv_macro_f1": cv_f1}
        if best is None or candidate["cv_macro_f1"] > best["cv_macro_f1"]:
            best = candidate
    if best is None:
        raise ValueError("No deterministic FCALC configuration was available")
    return best


def _tune_randomized_fcalc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    families: list[str],
    num_iters_grid: list[int],
    subsample_size_grid: list[int],
    cv_seed: int,
    seed: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for family, method in _methods_for_families(families):
        for num_iters in num_iters_grid:
            for subsample_size in subsample_size_grid:
                if not _can_sample_cv(y_train, int(subsample_size), cv_seed):
                    continue
                config = {
                    "family": family,
                    "aggregation": method,
                    "randomized": True,
                    "num_iters": int(num_iters),
                    "subsample_size": int(subsample_size),
                }
                cv_f1 = _score_fcalc_config(x_train, y_train, config, cv_seed=cv_seed, seed=seed)
                candidate = {**config, "cv_macro_f1": cv_f1}
                if best is None or candidate["cv_macro_f1"] > best["cv_macro_f1"]:
                    best = candidate
    if best is None:
        raise ValueError("No feasible randomized FCALC configuration was found")
    return best


class FCalcBaseline:
    def __init__(
        self,
        *,
        randomized: bool,
        scaler: str = "minmax",
        tune_aggregation: bool = True,
        families: list[str] | None = None,
        family: str = "standard",
        aggregation: str = "standard-support",
        num_iters: int = 20,
        subsample_size: int = 5,
        num_iters_grid: list[int] | None = None,
        subsample_size_grid: list[int] | None = None,
        seed: int = 42,
        cv_seed: int = 1998,
    ):
        self.randomized = bool(randomized)
        self.scaler_name = scaler
        self.tune_aggregation = bool(tune_aggregation)
        self.families = families or ["standard", "proximity"]
        self.fixed_family = family
        self.fixed_aggregation = aggregation
        self.fixed_num_iters = int(num_iters)
        self.fixed_subsample_size = int(subsample_size)
        self.num_iters_grid = [int(v) for v in (num_iters_grid or [10, 20, 30, 40, 50])]
        self.subsample_size_grid = [int(v) for v in (subsample_size_grid or list(range(1, 11)))]
        self.seed = int(seed)
        self.cv_seed = int(cv_seed)
        self.scaler = None
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.params_used: dict[str, Any] = {}
        self.fit_seconds: float | None = None
        self.predict_seconds: float | None = None

    def _make_scaler(self):
        if self.scaler_name == "minmax":
            return MinMaxScaler()
        if self.scaler_name in {"none", None}:
            return None
        raise ValueError(f"Unsupported FCALC scaler: {self.scaler_name}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray):
        started = time.time()
        X_train = _as_frame(X_train)
        y_train_arr = np.asarray(y_train, dtype=int)
        self.scaler = self._make_scaler()
        self.x_train = self.scaler.fit_transform(X_train) if self.scaler is not None else X_train.to_numpy(dtype=float)
        self.y_train = y_train_arr

        if self.tune_aggregation:
            if self.randomized:
                self.params_used = _tune_randomized_fcalc(
                    self.x_train,
                    self.y_train,
                    families=self.families,
                    num_iters_grid=self.num_iters_grid,
                    subsample_size_grid=self.subsample_size_grid,
                    cv_seed=self.cv_seed,
                    seed=self.seed,
                )
            else:
                self.params_used = _tune_deterministic_fcalc(
                    self.x_train,
                    self.y_train,
                    families=self.families,
                    cv_seed=self.cv_seed,
                    seed=self.seed,
                )
        else:
            self.params_used = {
                "family": self.fixed_family,
                "aggregation": self.fixed_aggregation,
                "randomized": self.randomized,
            }
            if self.randomized:
                self.params_used.update(
                    {"num_iters": self.fixed_num_iters, "subsample_size": self.fixed_subsample_size}
                )

        self.params_used.update({"scaler": self.scaler_name})
        self.fit_seconds = time.time() - started
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise ValueError("FCalcBaseline must be fitted before prediction")
        started = time.time()
        X_test = _as_frame(X_test)
        x_test = self.scaler.transform(X_test) if self.scaler is not None else X_test.to_numpy(dtype=float)
        predictions = _predict_fcalc_config(self.x_train, self.y_train, x_test, self.params_used, seed=self.seed)
        self.predict_seconds = time.time() - started
        return np.asarray(predictions, dtype=int)

    def get_params_used(self) -> dict[str, Any]:
        return dict(self.params_used)

    def get_compactness(self) -> dict[str, float]:
        return {}

    def get_timings(self) -> dict[str, float]:
        return {
            "fit_seconds": float(self.fit_seconds or 0.0),
            "predict_seconds": float(self.predict_seconds or 0.0),
        }


class IPSKNNBaseline:
    def __init__(
        self,
        *,
        scaler: str = "standard",
        tune_k: bool = True,
        k_values: list[int] | None = None,
        k: int = 3,
        p: int = 2,
        weights: str = "distance",
        seed: int = 42,
        cv_seed: int = 1998,
    ):
        self.scaler_name = scaler
        self.tune_k = bool(tune_k)
        self.k_values = [int(v) for v in (k_values or [1, 3, 5, 7, 9, 15, 25, 51])]
        self.k = int(k)
        self.p = int(p)
        self.weights = weights
        self.seed = int(seed)
        self.cv_seed = int(cv_seed)
        self.scaler = None
        self.model: IPSKNNClassifier | None = None
        self.X_train_scaled: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.params_used: dict[str, Any] = {}
        self.compactness: dict[str, float] = {}
        self.fit_seconds: float | None = None
        self.predict_seconds: float | None = None

    def _make_scaler(self):
        if self.scaler_name == "standard":
            return StandardScaler()
        if self.scaler_name in {"none", None}:
            return None
        raise ValueError(f"Unsupported IPS-KNN scaler: {self.scaler_name}")

    def _scaled_frame(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = _as_frame(X)
        if self.scaler is None:
            return X.astype(float)
        values = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    def _score_k(self, X: pd.DataFrame, y: pd.Series, k: int) -> float:
        scores = []
        splitter = _cv_splitter(y.to_numpy(dtype=int), self.cv_seed)
        for train_index, valid_index in splitter.split(X, y):
            X_train_fold = X.iloc[train_index]
            X_valid_fold = X.iloc[valid_index]
            y_train_fold = y.iloc[train_index]
            y_valid_fold = y.iloc[valid_index]
            fold_scaler = self._make_scaler()
            if fold_scaler is not None:
                X_train_fit = pd.DataFrame(
                    fold_scaler.fit_transform(X_train_fold),
                    columns=X.columns,
                    index=X_train_fold.index,
                )
                X_valid_eval = pd.DataFrame(
                    fold_scaler.transform(X_valid_fold),
                    columns=X.columns,
                    index=X_valid_fold.index,
                )
            else:
                X_train_fit = X_train_fold.astype(float)
                X_valid_eval = X_valid_fold.astype(float)
            model = IPSKNNClassifier(k=k, p=self.p, weights=self.weights)
            model.fit(X_train_fit, y_train_fold)
            predictions = model.predict(X_valid_eval)
            scores.append(_macro_f1_safe(y_valid_fold, predictions))
        return float(np.mean(scores))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray):
        started = time.time()
        X_train = _as_frame(X_train)
        y_train_series = _as_series(y_train).astype(int)

        if self.tune_k:
            valid_k_values = [k for k in self.k_values if 1 <= k <= len(X_train)]
            if not valid_k_values:
                valid_k_values = [1]
            scored = [{"k": k, "cv_macro_f1": self._score_k(X_train, y_train_series, k)} for k in valid_k_values]
            best = max(scored, key=lambda item: (item["cv_macro_f1"], -item["k"]))
            self.k = int(best["k"])
            self.params_used = {**best, "p": self.p, "weights": self.weights, "scaler": self.scaler_name}
        else:
            self.params_used = {"k": self.k, "p": self.p, "weights": self.weights, "scaler": self.scaler_name}

        self.scaler = self._make_scaler()
        self.X_train_scaled = self._scaled_frame(X_train, fit=True)
        self.y_train = y_train_series
        self.model = IPSKNNClassifier(k=self.k, p=self.p, weights=self.weights).fit(
            self.X_train_scaled, self.y_train
        )
        self.fit_seconds = time.time() - started
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("IPSKNNBaseline must be fitted before prediction")
        started = time.time()
        X_test_scaled = self._scaled_frame(X_test, fit=False)
        y_pred, _reason_df, size_dict, _reduced_df, _importance_df = self.model._predict_with_explanation(
            X_test_scaled,
            include_reduced_reason_for_classification=True,
            include_feature_importance=False,
        )
        sizes = np.asarray(list(size_dict.values()), dtype=float)
        self.compactness = {
            "avg_rrc_size": float(np.mean(sizes)) if len(sizes) else math.nan,
            "max_rrc_size": float(np.max(sizes)) if len(sizes) else math.nan,
            "rrc_rc_ratio": float(np.mean(sizes) / X_test_scaled.shape[1]) if len(sizes) else math.nan,
        }
        self.predict_seconds = time.time() - started
        return np.asarray(y_pred, dtype=int)

    def get_params_used(self) -> dict[str, Any]:
        return dict(self.params_used)

    def get_compactness(self) -> dict[str, float]:
        return dict(self.compactness)

    def get_timings(self) -> dict[str, float]:
        return {
            "fit_seconds": float(self.fit_seconds or 0.0),
            "predict_seconds": float(self.predict_seconds or 0.0),
        }
