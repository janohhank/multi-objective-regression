import math
from abc import ABC

import numpy as np
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from training.objective_components import (
    ObjectiveComponent,
)


class ModelEvaluationUtility(ABC):

    @staticmethod
    def evaluate_log_regression(
        objective_components: list[ObjectiveComponent],
        selected_features: list[str],
        correlation_to_target_feature: Series,
        log_regression: LogisticRegression,
        x: DataFrame,
        y_true: DataFrame,
    ):
        y_pred: np.ndarray = log_regression.predict(x)
        y_probs: np.ndarray = log_regression.predict_proba(x)[:, 1]

        results: dict[str, float] = {}
        if len(np.unique(y_pred)) != len(np.unique(y_true)):
            for objective_component in objective_components:
                results[objective_component.NAME] = 0.0
            return results

        # Coefficients sign diff penalty calculation
        coefficients: dict[str, float] = dict(
            zip(selected_features, log_regression.coef_[0])
        )

        coefficient_sign_diff_checks: dict[str, bool] = {}
        for feature, coefficient in coefficients.items():
            if math.isnan(correlation_to_target_feature[feature]):
                coefficient_sign_diff_checks[feature] = True
            else:
                check: float = correlation_to_target_feature[feature] * coefficient
                coefficient_sign_diff_checks[feature] = (
                    math.isclose(check, 0.0) or check < 0.0
                )
        coefficient_sign_diff_score: float = 1.0 - sum(
            coefficient_sign_diff_checks.values()
        ) / len(coefficient_sign_diff_checks)

        multi_objective_score: float = 0.0
        for objective_component in objective_components:
            score: float = objective_component.score(y_true, y_probs)
            weighted_score: float = objective_component.weighted_score(y_true, y_probs)
            results[objective_component.NAME] = score
            multi_objective_score += weighted_score
        results["multi_objective_score"] = multi_objective_score

        return results
