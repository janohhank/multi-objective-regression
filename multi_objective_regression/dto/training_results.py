from dataclasses import dataclass

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult


@dataclass
class TrainingResults:
    # Used training parameters during the traning
    training_parameters: TrainingParameters
    # Results of the different algorithms
    results: dict[str, TrainingResult]
