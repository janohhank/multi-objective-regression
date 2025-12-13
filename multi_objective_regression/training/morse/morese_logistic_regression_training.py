import math
import time
import typing
from copy import deepcopy

import numpy as np
from dto.morse_training_results import MorseTrainingResults
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from training.objective_components import ObjectiveComponent
from training.objective_components import (
    create_objective_components,
)
from utils.training_utility import TrainingUtility


class MorseLogisticRegressionTraining:
    __training_parameters: TrainingParameters = None
    __objective_components: list[ObjectiveComponent] = None

    def __init__(self, parameters: TrainingParameters):
        self.__training_parameters = parameters

        self.__objective_components = create_objective_components(
            self.__training_parameters.multi_objective_functions
        )

    def train(
        self,
        index: int,
        training_setup: TrainingSetup,
        pearson_correlation_to_target_feature: DataFrame,
        x_train: DataFrame,
        y_train: DataFrame,
        x_validation: DataFrame,
        y_validation: DataFrame,
        x_test: DataFrame,
        y_test: DataFrame,
    ) -> TrainingResult:
        start: float = time.perf_counter()

        # Reduced train datasets.
        x_train_reduced = x_train[training_setup.features].copy()
        x_validation_reduced = x_validation[training_setup.features].copy()
        x_test_reduced = x_test[training_setup.features].copy()

        # Create z-score standardization scaler from train dataset.
        scaler = TrainingUtility.fit_standard_scaler(x_train_reduced)

        # Standardize train dataset.
        scaled_x_train = scaler.transform(x_train_reduced)

        # Standardize validation and test dataset.
        scaled_x_validation = scaler.transform(x_validation_reduced)
        scaled_x_test = scaler.transform(x_test_reduced)

        log_regression = LogisticRegression(
            n_jobs=4,
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1024,
            random_state=42,
        )
        log_regression.fit(scaled_x_train, y_train)

        coefficients: dict[str, float] = dict(
            zip(training_setup.features, log_regression.coef_[0])
        )

        elapsed_sec: float = time.perf_counter() - start

        return MorseTrainingResults(
            training_setup=training_setup,
            validation_results=self.evaluate(
                training_setup,
                pearson_correlation_to_target_feature,
                log_regression,
                scaled_x_validation,
                y_validation,
            ),
            test_results=self.evaluate(
                training_setup,
                pearson_correlation_to_target_feature,
                log_regression,
                scaled_x_test,
                y_test,
            ),
            training_time_seconds=elapsed_sec,
            model=deepcopy(log_regression),
            scaler=deepcopy(scaler),
            # MORSE results fields
            index=index,
            coefficients=coefficients,
            interception=float(log_regression.intercept_[0]),
            iteration=int(log_regression.n_iter_[0]),
        )

    def evaluate(
        self,
        training_setup: TrainingSetup,
        correlation_to_target_feature: DataFrame,
        log_regression: LogisticRegression,
        x_test: DataFrame,
        y_test: DataFrame,
    ):
        y_pred: typing.Any = log_regression.predict(x_test)
        y_probs: typing.Any = log_regression.predict_proba(x_test)[:, 1]

        if len(np.unique(y_pred)) != len(np.unique(y_test)):
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0,
                "pr_auc": 0.0,
                "gini_score": 0.0,
                "coefficient_sign_diff_score": 0.0,
                "multi_objective_score": 0.0,
            }

        # Coefficients sign diff penalty calculation
        coefficients: dict[str, float] = dict(
            zip(training_setup.features, log_regression.coef_[0])
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

        results: dict[str, float] = {}
        multi_objective_score: float = 0.0
        for objective_component in self.__objective_components:
            score: float = objective_component.weighted_score(y_test, y_probs)
            results[objective_component.NAME] = score
            multi_objective_score += score
        results["multi_objective_score"] = multi_objective_score

        return results

    def suggest_training_setup_candidates(
        self, training_result: dict[int, TrainingResult]
    ) -> list[TrainingSetup]:
        results: list[TrainingResult] = []

        for index, training_result in training_result.items():
            is_contains: bool = any(
                math.isclose(coefficient, 0.0)
                for coefficient in training_result.coefficients.values()
            )
            if not is_contains:
                continue

            new_training_setup: TrainingSetup = deepcopy(training_result.training_setup)
            new_training_setup.index = -1
            new_training_setup.features = []
            for feature, coefficient in training_result.coefficients.items():
                if not math.isclose(coefficient, 0.0):
                    new_training_setup.features.append(feature)

            if len(new_training_setup.features) != 0:
                results.append(new_training_setup)

        return results
