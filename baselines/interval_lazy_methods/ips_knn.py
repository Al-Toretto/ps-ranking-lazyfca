from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pandas as pd

from .hyperrectangle import Hyperrectangle
from .information_gain import InformationGainAnalyzer


def expand_hyperrectangle_by_information_gain(
    rect: dict[Any, tuple[float, float]], X: pd.DataFrame, y: pd.Series
) -> tuple[dict[Any, tuple[float, float]], int]:
    hyperrectangle = Hyperrectangle([(rect[col][0], rect[col][1]) for col in X.columns])
    condition_list = hyperrectangle.find_condition_list_inside_hyperrectangle(X)
    mask = functools.reduce(lambda left, right: left & right, condition_list)
    count_inside_rect = mask.sum()

    cols = list(X.columns)
    used_cols = []
    X_new = X.copy()
    y_new = y.copy()
    while len(X_new) > count_inside_rect:
        max_gain = -1.0
        max_gain_col = None
        for col in cols:
            if col in used_cols:
                continue
            gain = InformationGainAnalyzer.find_information_gain_for_splitting_by_interval(
                X_new, y_new, col, rect[col]
            )
            if gain > max_gain:
                max_gain = gain
                max_gain_col = col
        if max_gain_col is None:
            break
        used_cols.append(max_gain_col)
        X_new = X_new[
            (X_new.loc[:, max_gain_col] >= rect[max_gain_col][0])
            & (X_new.loc[:, max_gain_col] <= rect[max_gain_col][1])
        ]
        y_new = y_new[X_new.index]

    new_rect = rect.copy()
    used_cols_set = set(used_cols)
    for col in new_rect:
        if col not in used_cols_set:
            new_rect[col] = (X[col].min(), X[col].max())
    return new_rect, len(used_cols_set)


def find_feature_importance_scores_for_hyperrectangle(
    rect: dict[Any, tuple[float, float]], X: pd.DataFrame, y: pd.Series
) -> dict[Any, float]:
    scores = {
        col: InformationGainAnalyzer.find_information_gain_for_splitting_by_interval(X, y, col, interval)
        for col, interval in rect.items()
    }
    total = sum(scores.values())
    if total != 0:
        scores = {col: score / total for col, score in scores.items()}
    return scores


class IPSKNNClassifier:
    def __init__(self, k: int = 3, p: int = 2, weights: str = "distance"):
        self.k = int(k)
        self.p = int(p)
        self.weights = weights
        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"k": self.k, "p": self.p, "weights": self.weights}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def _find_sorted_distances_and_samples_mask_supporting_one_sample(
        self, sample: pd.Series, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        distances = np.power(
            np.sum(np.power(np.subtract(df, sample), self.p), axis="columns"),
            1 / self.p,
        ).sort_values(kind="mergesort")
        knn_indices = np.array(distances[: self.k].index)
        knn = df.loc[knn_indices]

        knn_hyperrectangle = Hyperrectangle([(knn[col].min(), knn[col].max()) for col in df.columns])
        conditions = knn_hyperrectangle.find_condition_list_inside_hyperrectangle(df)
        inside_hyperrectangle_mask = functools.reduce(lambda left, right: left & right, conditions)
        return distances, inside_hyperrectangle_mask

    def _find_vote_score_for_distance(self, distance: float) -> float:
        if self.weights == "uniform":
            return 1.0
        if self.weights == "distance":
            return 1.0 / distance if distance != 0 else 10000.0
        raise ValueError(f"Unknown weights value: {self.weights}")

    def _find_votes_for_one_sample(self, sample: pd.Series) -> tuple[dict[Any, float], pd.Series, pd.Series]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("IPSKNNClassifier must be fitted before prediction")
        distances, mask = self._find_sorted_distances_and_samples_mask_supporting_one_sample(sample, self.X_train)
        supporting_labels = self.y_train[mask]
        votes: dict[Any, float] = {}
        if self.weights == "uniform":
            votes = {label: float(count) for label, count in supporting_labels.value_counts().to_dict().items()}
        elif self.weights == "distance":
            for index, label in supporting_labels.items():
                votes[label] = votes.get(label, 0.0) + self._find_vote_score_for_distance(distances[index])
        else:
            raise ValueError(f"Unknown weights value: {self.weights}")
        return votes, distances, mask

    def _predict_one_sample(self, sample: pd.Series) -> Any:
        votes, _, _ = self._find_votes_for_one_sample(sample)
        return max(votes.items(), key=lambda item: item[1])[0]

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        y_pred = pd.Series(data=np.nan, index=X_test.index, dtype=object)
        for index, sample in X_test.iterrows():
            y_pred[index] = self._predict_one_sample(sample)
        return y_pred

    def _find_reduced_reason_for_classification(
        self, reason_for_classification: dict[Any, tuple[float, float]]
    ) -> tuple[dict[Any, tuple[float, float]], int]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("IPSKNNClassifier must be fitted before explanations")
        return expand_hyperrectangle_by_information_gain(reason_for_classification, self.X_train, self.y_train)

    def _find_feature_importance_scores(self, rect: dict[Any, tuple[float, float]]) -> dict[Any, float]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("IPSKNNClassifier must be fitted before explanations")
        return find_feature_importance_scores_for_hyperrectangle(rect, self.X_train, self.y_train)

    def _predict_with_explanation_one_sample_with_metadata(
        self, sample: pd.Series
    ) -> tuple[Any, dict[Any, tuple[float, float]], dict[str, Any]]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("IPSKNNClassifier must be fitted before explanations")
        votes, distances, mask = self._find_votes_for_one_sample(sample)
        sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        largest_label = sorted_votes[0][0]
        largest_score = sorted_votes[0][1]
        second_largest_label = sorted_votes[1][0] if len(sorted_votes) > 1 else None
        second_largest_score = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0

        mask_largest = mask & (self.y_train == largest_label)
        mask_other = mask & (self.y_train != largest_label)
        distances_largest = distances[mask_largest]

        explanation_score = 0.0
        explanation_indices = []
        for dist_index, dist_value in distances_largest.items():
            explanation_score += self._find_vote_score_for_distance(dist_value)
            explanation_indices.append(dist_index)
            if explanation_score >= second_largest_score:
                break

        explanation_samples = self.X_train.loc[explanation_indices]
        reason_for_classification = {
            col: (
                min(explanation_samples[col].min(), sample.loc[col]),
                max(explanation_samples[col].max(), sample.loc[col]),
            )
            for col in self.X_train.columns
        }
        return (
            largest_label,
            reason_for_classification,
            {
                "k": self.k,
                "predicted_label": largest_label,
                "opposer_label": second_largest_label,
                "supporter_count": int(mask_largest.sum()),
                "opposer_count": int(mask_other.sum()),
                "original_supporter_score": largest_score,
                "original_opposer_score": second_largest_score,
                "taken_supporter_count": len(explanation_indices),
                "taken_supporter_score": explanation_score,
                "taken_supporter_indices": explanation_indices,
            },
        )

    def _predict_with_explanation(
        self,
        X_test: pd.DataFrame,
        include_reduced_reason_for_classification: bool = False,
        include_feature_importance: bool = False,
    ) -> tuple[pd.Series, pd.DataFrame, dict[Any, int], pd.DataFrame | None, pd.DataFrame | None]:
        y_pred = pd.Series(data=np.nan, index=X_test.index, dtype=object)
        reason_by_index = {}
        reduced_reason_by_index = {}
        feature_importance_by_index = {}
        classifier_size_by_index = {}

        for index, sample in X_test.iterrows():
            predicted_label, reason, _metadata = self._predict_with_explanation_one_sample_with_metadata(sample)
            y_pred[index] = predicted_label
            reason_by_index[index] = reason
            classifier_size = len(reason)
            if include_reduced_reason_for_classification:
                reduced_reason_by_index[index], classifier_size = self._find_reduced_reason_for_classification(reason)
            classifier_size_by_index[index] = classifier_size
            if include_feature_importance:
                source_reason = reduced_reason_by_index[index] if include_reduced_reason_for_classification else reason
                feature_importance_by_index[index] = self._find_feature_importance_scores(source_reason)

        reason_df = pd.DataFrame.from_dict(reason_by_index, orient="index")
        reduced_reason_df = (
            pd.DataFrame.from_dict(reduced_reason_by_index, orient="index")
            if include_reduced_reason_for_classification
            else None
        )
        feature_importance_df = (
            pd.DataFrame.from_dict(feature_importance_by_index, orient="index") if include_feature_importance else None
        )
        return y_pred, reason_df, classifier_size_by_index, reduced_reason_df, feature_importance_df
