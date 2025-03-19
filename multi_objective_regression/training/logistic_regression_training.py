import time
import typing

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class LogisticRegressionTraining:
    __training_parameters: TrainingParameters = None

    def __init__(self, parameters: TrainingParameters):
        self.__training_parameters = parameters

    def train(
        self,
        index: int,
        training_setup: TrainingSetup,
        covariance_to_target_feature,
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) -> TrainingResult:
        start: float = time.perf_counter()

        # Reduced train datasets.
        x_train_reduced = x_train[training_setup.features].copy()
        x_validation_reduced = x_validation[training_setup.features].copy()
        x_test_reduced = x_test[training_setup.features].copy()

        # Create z-score standardization scaler from train dataset.
        scaler = LogisticRegressionTraining.__get_standardization_scaler(
            x_train_reduced
        )

        # Standardize train dataset.
        scaled_x_train = scaler.transform(x_train_reduced)

        # Standardize validation and test dataset.
        scaled_x_validation = scaler.transform(x_validation_reduced)
        scaled_x_test = scaler.transform(x_test_reduced)

        log_regression = LogisticRegression(
            n_jobs=4, penalty="l2", C=1.0, solver="lbfgs", max_iter=1024
        )
        log_regression.fit(scaled_x_train, y_train)

        coefficients: dict[str, float] = dict(
            zip(training_setup.features, log_regression.coef_[0])
        )

        elapsed: float = time.perf_counter() - start

        return TrainingResult(
            index,
            training_setup,
            coefficients,
            float(log_regression.intercept_[0]),
            int(log_regression.n_iter_[0]),
            LogisticRegressionTraining.evaluate_model(
                self.__training_parameters,
                training_setup,
                covariance_to_target_feature,
                log_regression,
                scaled_x_validation,
                y_validation,
            ),
            LogisticRegressionTraining.evaluate_model(
                self.__training_parameters,
                training_setup,
                covariance_to_target_feature,
                log_regression,
                scaled_x_test,
                y_test,
            ),
            elapsed,
        )

    @staticmethod
    def evaluate_model(
        training_parameters: TrainingParameters,
        training_setup: TrainingSetup,
        covariance_to_target_feature,
        log_regression,
        x_test,
        y_test,
    ):
        coefficients: dict[str, float] = dict(
            zip(training_setup.features, log_regression.coef_[0])
        )

        coefficient_sign_diff_checks: dict[str, bool] = {}
        for feature, coefficient in coefficients.items():
            coefficient_sign_diff_checks[feature] = (
                covariance_to_target_feature[feature] * coefficient < 0
            )
        coefficient_sign_diff_penalty: float = 1.0 - sum(
            coefficient_sign_diff_checks.values()
        ) / len(coefficient_sign_diff_checks)

        y_validation_pred: typing.Any = log_regression.predict(x_test)
        accuracy: float = accuracy_score(y_test, y_validation_pred)
        precision: float = precision_score(y_test, y_validation_pred)

        y_probs: typing.Any = log_regression.predict_proba(x_test)[:, 1]
        roc_auc: float = roc_auc_score(y_test, y_probs)

        gini_score: float = 2 * roc_auc - 1

        multi_objective_score: float = (
            training_parameters.multi_objective_function_weights["accuracy_weight"]
            * accuracy
            + training_parameters.multi_objective_function_weights["precision_weight"]
            * precision
            + training_parameters.multi_objective_function_weights["roc_auc_weight"]
            * roc_auc
            + training_parameters.multi_objective_function_weights["gini_score_weight"]
            * gini_score
            + training_parameters.multi_objective_function_weights[
                "coefficient_sign_diff_penalty_weight"
            ]
            * coefficient_sign_diff_penalty
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "roc_auc": roc_auc,
            "gini_score": gini_score,
            "coefficient_sign_diff_penalty": coefficient_sign_diff_penalty,
            "coefficient_sign_diff_checks": coefficient_sign_diff_checks,
            "multi_objective_score": multi_objective_score,
        }

    @staticmethod
    def __get_standardization_scaler(dataset: DataFrame) -> typing.Any:
        standard_scaler = StandardScaler()
        return standard_scaler.fit(dataset)
