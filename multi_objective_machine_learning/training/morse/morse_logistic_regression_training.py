import math
import time
from copy import deepcopy

from dto.morse_training_results import MorseTrainingResults
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression
from training.model_evaluation_utility import (
    ModelEvaluationUtility,
)
from training.objective_components import ObjectiveComponent
from training.objective_components import (
    create_objective_components,
)
from training.objective_components import create_objective_components_dictionary
from utils.training_utility import TrainingUtility


class MorseLogisticRegressionTraining:
    __training_parameters: TrainingParameters = None
    __objective_components: dict[str, ObjectiveComponent] = None
    __coefficient_sign_diff_component: ObjectiveComponent = None

    def __init__(self, parameters: TrainingParameters):
        self.__training_parameters = parameters

        self.__objective_components: dict[str, ObjectiveComponent] = (
            create_objective_components_dictionary(
                self.__training_parameters.multi_objective_functions
            )
        )
        self.__coefficient_sign_diff_component: ObjectiveComponent = (
            self.__objective_components["coefficient_sign_diff"]
        )

    def train(
        self,
        index: int,
        training_setup: TrainingSetup,
        pearson_correlation_to_target_feature: Series,
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

        if self.__coefficient_sign_diff_component is not None:
            self.__coefficient_sign_diff_component.update_context(
                coefficients, pearson_correlation_to_target_feature
            )

        elapsed_sec: float = time.perf_counter() - start

        return MorseTrainingResults(
            training_setup=training_setup,
            validation_results=ModelEvaluationUtility.evaluate_log_regression(
                self.__objective_components.values(),
                log_regression,
                scaled_x_validation,
                y_validation,
            ),
            test_results=ModelEvaluationUtility.evaluate_log_regression(
                self.__objective_components.values(),
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
