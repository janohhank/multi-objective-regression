from dataclasses import dataclass

from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup


@dataclass(kw_only=True)
class DeapTrainingResults(TrainingResult):
    pass
