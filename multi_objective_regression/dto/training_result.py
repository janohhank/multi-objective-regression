from dataclasses import dataclass

from dto.training_setup import TrainingSetup


@dataclass
class TrainingResult:
    index: int
    training_setup: TrainingSetup
    coefficients: dict[str, float]
    interception: float
    iteration: int
    validation_results: dict[str, float]
    test_results: dict[str, float]
    training_time_seconds: float
