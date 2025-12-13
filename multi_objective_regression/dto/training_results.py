from dataclasses import dataclass, asdict

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult


@dataclass
class TrainingResults:
    # Used training parameters during the training
    training_parameters: TrainingParameters
    # Results of the different algorithms
    results: dict[str, TrainingResult]

    def to_dict(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if not self.__dataclass_fields__[k].metadata.get("skip", False)
        }
