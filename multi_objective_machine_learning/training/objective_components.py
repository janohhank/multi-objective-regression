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

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Any, Iterable, Mapping, Union, Tuple

import numpy as np
from imblearn.metrics import specificity_score
from pandas import Series
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

    NAME: str = None

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

    NAME: str = "accuracy"

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

    NAME: str = "precision"

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

    NAME: str = "recall"

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

    NAME: str = "specificity"

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

    NAME: str = "f1_score"

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

    NAME: str = "roc_auc"

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

    NAME: str = "pr_auc"

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

    NAME: str = "gini"

    def score(
        self,
        y_true: Sequence[int],
        y_probs: Sequence[float],
        threshold: float = 0.5,  # kept for unified API; not used
    ) -> float:
        auc = float(roc_auc_score(y_true, y_probs))
        return 2.0 * auc - 1.0


class CoefficientSignDiffComponent(ObjectiveComponent):
    """
    Coefficient sign difference component.

    Computes a score in [0, 1] that penalizes features whose learned coefficient
    sign disagrees with the sign of their correlation to the target.

    The score is defined as:
        1 - (# of penalized features) / (# of evaluated features)

    A feature is penalized if:
      - its correlation to target is missing/NaN, or
      - the product correlation * coefficient is negative, or approximately 0
        within abs tolerance 'zero_tol'.

    This component ignores y_true/y_probs; they're accepted for API consistency.
    """

    NAME: str = "coefficient_sign_diff"

    __coefficients: dict[str, float]
    __pearson_correlation_to_target_feature: Series

    def update_context(
        self,
        coefficients: dict[str, float],
        pearson_correlation_to_target_feature: Series,
    ) -> None:
        """
        Update or provide context needed for scoring at runtime.
        """
        self.__coefficients = coefficients
        self.__pearson_correlation_to_target_feature = (
            pearson_correlation_to_target_feature
        )

    def score(
        self,
        y_true: Sequence[int],
        y_probs: Sequence[float],
        threshold: float = 0.5,
    ) -> float:
        coefficient_sign_diff_checks: dict[str, bool] = {}
        for feature, coefficient in self.__coefficients.items():
            if math.isnan(self.__pearson_correlation_to_target_feature[feature]):
                coefficient_sign_diff_checks[feature] = True
            else:
                check: float = (
                    self.__pearson_correlation_to_target_feature[feature] * coefficient
                )
                coefficient_sign_diff_checks[feature] = (
                    math.isclose(check, 0.0) or check < 0.0
                )
        coefficient_sign_diff_score: float = 1.0 - sum(
            coefficient_sign_diff_checks.values()
        ) / len(coefficient_sign_diff_checks)

        return coefficient_sign_diff_score


# Map name aliases to component classes
_COMPONENT_ALIASES: dict[str, type[ObjectiveComponent]] = {
    # Accuracy
    AccuracyComponent.NAME: AccuracyComponent,
    # Precision
    PrecisionComponent.NAME: PrecisionComponent,
    # Recall (Sensitivity / TPR)
    RecallComponent.NAME: RecallComponent,
    # Specificity (TNR)
    SpecificityComponent.NAME: SpecificityComponent,
    # F1
    F1ScoreComponent.NAME: F1ScoreComponent,
    # ROC AUC
    RocAucComponent.NAME: RocAucComponent,
    # PR AUC (Average Precision)
    PrAucComponent.NAME: PrAucComponent,
    # Gini
    GiniComponent.NAME: GiniComponent,
    # Coefficient sign difference
    CoefficientSignDiffComponent.NAME: CoefficientSignDiffComponent,
}


# Extra __init__ kwargs supported per component class
_ALLOWED_INIT_KWARGS: dict[type[ObjectiveComponent], set[str]] = {
    PrecisionComponent: {"zero_division"},
    F1ScoreComponent: {"zero_division"},
}


def create_objective_component(
    name: str,
    weight: Optional[float] = None,
    **kwargs: Any,
) -> ObjectiveComponent:
    """
    Factory: create a single objective component by name.

    Parameters
    ----------
    name : str
        Case-insensitive name or alias, e.g.:
        "accuracy", "precision", "recall", "specificity",
        "f1_score", "roc_auc", "pr_auc", "gini"
    weight : Optional[float]
        Optional non-negative weight applied in weighted_score(...).
    **kwargs : Any
        Extra constructor kwargs supported by some components:
          - PrecisionComponent: zero_division
          - F1ScoreComponent: zero_division

    Returns
    -------
    ObjectiveComponent

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    cls = _COMPONENT_ALIASES.get(name)
    if cls is None:
        available = ", ".join(sorted(set(_COMPONENT_ALIASES.keys())))
        raise ValueError(
            f"Unknown objective component '{name}'. Available names: {available}"
        )

    allowed = _ALLOWED_INIT_KWARGS.get(cls, set())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return cls(weight=weight, **filtered_kwargs)  # type: ignore[arg-type]


def create_objective_components(
    specs: Union[Mapping[str, float], Iterable[Tuple[str, float]]],
    default_kwargs: Optional[dict[str, Any]] = None,
) -> list[ObjectiveComponent]:
    """
    Build multiple components from a name->weight mapping or (name, weight) pairs.

    Examples
    --------
    create_objective_components({"precision": 0.4, "recall": 0.6})
    create_objective_components([("accuracy", 1.0), ("roc_auc", 0.5)],
                                default_kwargs={"zero_division": 0})
    """
    components: list[ObjectiveComponent] = []
    if isinstance(specs, Mapping):
        items = specs.items()
    else:
        items = list(specs)

    for name, weight in items:  # type: ignore[misc]
        kwargs = dict(default_kwargs) if default_kwargs else {}
        comp = create_objective_component(name, weight=weight, **kwargs)
        components.append(comp)

    return components


def create_objective_components_dictionary(
    specs: Union[Mapping[str, float], Iterable[Tuple[str, float]]],
    default_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, ObjectiveComponent]:
    """
    Build multiple components from a name->weight mapping or (name, weight) pairs and return a dictionary.

    Examples
    --------
    create_objective_components_dictionary({"precision": 0.4, "recall": 0.6})
    create_objective_components_dictionary([("accuracy", 1.0), ("roc_auc", 0.5)],
                                           default_kwargs={"zero_division": 0})
    """
    components: dict[str, ObjectiveComponent] = {}
    if isinstance(specs, Mapping):
        items = specs.items()
    else:
        items = list(specs)

    for name, weight in items:  # type: ignore[misc]
        kwargs = dict(default_kwargs) if default_kwargs else {}
        comp = create_objective_component(name, weight=weight, **kwargs)
        components[name] = comp

    return components


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
    "CoefficientSignDiffComponent",
    "create_objective_component",
    "create_objective_components",
]
