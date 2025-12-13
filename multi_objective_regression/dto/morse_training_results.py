from dataclasses import dataclass

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup


@dataclass(kw_only=True)
class MorseTrainingResults(TrainingResult):
    index: int
    # Calculated regression coefficients
    coefficients: dict[str, float]
    # Calculated interception point
    interception: float
    # Number of iterations of the regression train
    iteration: int
