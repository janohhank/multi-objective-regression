from dataclasses import dataclass

from dto.training_setup import TrainingSetup


@dataclass
class TrainingResult:
    index: int
    training_setup: TrainingSetup
    coefficients: dict[str, float]
    coefficient_sign_diff_checks: dict[str, bool]
    interception: float
    iteration: int
    test_accuracy: float
    test_precision: float
    roc_auc: float
    gini_score: float
    coefficient_sign_diff_penalty: float
    multi_objective_score: float
    training_time_seconds: float
