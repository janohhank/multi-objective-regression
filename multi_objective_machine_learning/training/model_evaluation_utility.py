from abc import ABC
from typing import Any

import numpy as np
from pandas import DataFrame
from training.objective_components import (
    ObjectiveComponent,
)


class ModelEvaluationUtility(ABC):

    @staticmethod
    def evaluate_log_regression(
        objective_components: list[ObjectiveComponent],
        model: Any,
        x: DataFrame,
        y_true: DataFrame,
    ):
        y_pred: np.ndarray = model.predict(x)
        y_probs: np.ndarray = model.predict_proba(x)[:, 1]

        results: dict[str, float] = {}
        if len(np.unique(y_pred)) != len(np.unique(y_true)):
            for objective_component in objective_components:
                results[objective_component.NAME] = 0.0
            return results

        multi_objective_score: float = 0.0
        for objective_component in objective_components:
            score: float = objective_component.score(y_true, y_probs)
            weighted_score: float = objective_component.weighted_score(y_true, y_probs)
            results[objective_component.NAME] = score
            multi_objective_score += weighted_score
        results["multi_objective_score"] = multi_objective_score

        return results
