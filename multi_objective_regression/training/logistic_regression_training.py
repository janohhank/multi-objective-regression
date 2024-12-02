import time
import typing

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score


class LogisticRegressionTraining:
    __training_parameters: TrainingParameters = None

    __test_accuracy_weight: float = 1.0
    __test_precision_weight: float = 1.0
    __roc_auc_weight: float = 1.0
    __gini_score_weight: float = 1.0
    __coefficient_sign_diff_penalty_weight: float = 1.0

    def __init__(self, parameters: TrainingParameters):
        self.__training_parameters = parameters

        self.__test_accuracy_weight = (
            self.__training_parameters.multi_objective_function_weights[
                "test_accuracy_weight"
            ]
        )
        self.__test_precision_weight = (
            self.__training_parameters.multi_objective_function_weights[
                "test_precision_weight"
            ]
        )
        self.__roc_auc_weight = (
            self.__training_parameters.multi_objective_function_weights[
                "roc_auc_weight"
            ]
        )
        self.__gini_score_weight = (
            self.__training_parameters.multi_objective_function_weights[
                "gini_score_weight"
            ]
        )
        self.__coefficient_sign_diff_penalty_weight = (
            self.__training_parameters.multi_objective_function_weights[
                "coefficient_sign_diff_penalty_weight"
            ]
        )

    def train(
        self,
        index: int,
        training_setup: TrainingSetup,
        covariance_to_target_feature,
        x_train,
        y_train,
        x_test,
        y_test,
    ) -> TrainingResult:
        start: float = time.perf_counter()

        log_regression = LogisticRegression()
        log_regression.fit(x_train[training_setup.features], y_train)

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

        y_test_pred: typing.Any = log_regression.predict(
            x_test[training_setup.features]
        )
        test_accuracy: float = accuracy_score(y_test, y_test_pred)
        test_precision: float = precision_score(y_test, y_test_pred)

        y_probs: typing.Any = log_regression.predict_proba(
            x_test[training_setup.features]
        )[:, 1]
        roc_auc: float = roc_auc_score(y_test, y_probs)

        gini_score: float = 2 * roc_auc - 1

        multi_objective_score: float = (
            self.__test_accuracy_weight * test_accuracy
            + self.__test_precision_weight * test_precision
            + self.__roc_auc_weight * roc_auc
            + self.__gini_score_weight * gini_score
            + self.__coefficient_sign_diff_penalty_weight
            * coefficient_sign_diff_penalty
        )

        elapsed: float = time.perf_counter() - start

        return TrainingResult(
            index,
            training_setup,
            coefficients,
            coefficient_sign_diff_checks,
            float(log_regression.intercept_[0]),
            int(log_regression.n_iter_[0]),
            test_accuracy,
            test_precision,
            roc_auc,
            gini_score,
            coefficient_sign_diff_penalty,
            multi_objective_score,
            elapsed,
        )
