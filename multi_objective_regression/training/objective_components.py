"""
Objective components for binary classification with a unified API.

All score(...) methods in components follow the same signature:
    score(y_true, y_probs, threshold=0.5)

- y_true: array-like of shape (n_samples,) with ground truth binary labels (0/1)
- y_probs: array-like of shape (n_samples,)
    - For thresholded metrics (accuracy, precision, recall, specificity, ...),
      y_probs is treated as probability/score for the positive class and is
      converted to binary using 'threshold' (>= threshold -> class 1).
    - For ROC AUC, y_probs is used directly as probability/score (threshold is ignored).
- threshold: float, default 0.5
    - Used by thresholded metrics to derive binary predictions.
    - Ignored by metrics that work on continuous scores (ROC AUC).

Each component supports an optional 'weight' that is applied via weighted_score(...).

This module only adds new classes and does not modify existing project code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from imblearn.metrics import specificity_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
)


def _to_binary_predictions(y_probs: np.ndarray, threshold: float) -> np.ndarray:
    return (y_probs >= float(threshold)).astype(int)


class ObjectiveComponent(ABC):
    """
    Base class for objective components.

    Parameters
    ----------
    weight : Optional[float]
        Optional non-negative weight applied when calling weighted_score.
        If None, weighted_score returns the raw score.
    """

    def __init__(self, weight: Optional[float] = None) -> None:
        if weight is not None and weight < 0:
            raise ValueError("weight must be non-negative if provided")
        self.weight: Optional[float] = weight

    @abstractmethod
    def score(
        self,
        y_true: Sequence[int],
        y_probs: Sequence[float],
        threshold: float = 0.5,
    ) -> float:
        """
        Compute the metric score using a unified signature.

        For thresholded metrics, 'y_probs' is converted to binary with 'threshold'.
        For score-based metrics (e.g., ROC AUC), 'y_probs' is used directly.

        Returns
        -------
        float
            The computed metric value.
        """

    def weighted_score(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute the weighted score, using 'weight' if set.

        Returns
        -------
        float
            score if weight is None, otherwise weight * score
        """
        s = self.score(y_true, y_probs, threshold)
        return float(s if self.weight is None else self.weight * s)


class AccuracyComponent(ObjectiveComponent):
    """
    Accuracy metric component.

    Uses threshold to binarize y_probs before computing accuracy.
    """

    def score(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        y_pred = _to_binary_predictions(y_probs, threshold)
        return float(accuracy_score(y_true, y_pred))


class PrecisionComponent(ObjectiveComponent):
    """
    Precision metric component.

    Uses threshold to binarize y_probs before computing precision.

    Parameters
    ----------
    weight : Optional[float]
        Non-negative optional weight.
    zero_division : int | float, default=0
        Value to return when there is a zero division (no positive predictions).
    """

    def __init__(
        self,
        weight: Optional[float] = None,
        zero_division: int | float = 0,
    ) -> None:
        super().__init__(weight)
        self.zero_division = zero_division

    def score(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        y_pred = _to_binary_predictions(y_probs, threshold)
        return float(precision_score(y_true, y_pred, zero_division=self.zero_division))


class RecallComponent(ObjectiveComponent):
    """
    Recall (Sensitivity/TPR) metric component.

    Uses threshold to binarize y_probs before computing recall.
    """

    def score(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        y_pred = _to_binary_predictions(y_probs, threshold)
        return float(recall_score(y_true, y_pred))


class SpecificityComponent(ObjectiveComponent):
    """
    Specificity (True Negative Rate) metric component.

    Uses threshold to binarize y_probs, then computes:
        Specificity = TN / (TN + FP)

    Edge cases:
    - If only negatives present in y_true -> specificity = 1.0
    - If denominator TN+FP == 0 -> returns 0.0
    """

    def score(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        y_pred = _to_binary_predictions(y_probs, threshold)
        return float(specificity_score(y_true, y_pred))


class F1ScoreComponent(ObjectiveComponent):
    """
    F1 score metric component.

    Uses threshold to binarize y_probs before computing F1 score.
    """

    def __init__(
        self,
        weight: Optional[float] = None,
        zero_division: int | float = 0,
    ) -> None:
        super().__init__(weight)
        self.zero_division = zero_division

    def score(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        y_pred = _to_binary_predictions(y_probs, threshold)
        return float(f1_score(y_true, y_pred, zero_division=self.zero_division))


class RocAucComponent(ObjectiveComponent):
    """
    ROC AUC metric component.

    Uses y_probs directly as probability/score for the positive class.
    The 'threshold' parameter is accepted for API consistency but ignored.
    """

    def score(
        self,
        y_true: Sequence[int],
        y_probs: Sequence[float],
        threshold: float = 0.5,  # kept for unified API; not used
    ) -> float:
        return float(roc_auc_score(y_true, y_probs))


class PrAucComponent(ObjectiveComponent):
    """
    Precision-Recall AUC metric component.

    Uses y_probs directly as probabilities/scores for the positive class.
    The 'threshold' parameter is accepted for API consistency but ignored.
    """

    def score(
        self,
        y_true: Sequence[int],
        y_probs: Sequence[float],
        threshold: float = 0.5,  # kept for unified API; not used
    ) -> float:
        return float(average_precision_score(y_true, y_probs))


class GiniComponent(ObjectiveComponent):
    """
    Gini coefficient metric component.

    Computed from ROC AUC as: Gini = 2 * AUC - 1.
    Uses y_probs directly as probabilities/scores for the positive class.
    The 'threshold' parameter is accepted for API consistency but ignored.
    """

    def score(
        self,
        y_true: Sequence[int],
        y_probs: Sequence[float],
        threshold: float = 0.5,  # kept for unified API; not used
    ) -> float:
        auc = float(roc_auc_score(y_true, y_probs))
        return 2.0 * auc - 1.0


__all__ = [
    "ObjectiveComponent",
    "AccuracyComponent",
    "PrecisionComponent",
    "RecallComponent",
    "SpecificityComponent",
    "F1ScoreComponent",
    "RocAucComponent",
    "PrAucComponent",
    "GiniComponent",
]
