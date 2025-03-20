from dataclasses import dataclass, field, asdict

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
    model: object = field(default=None, metadata={"skip": True})
    scaler: object = field(default=None, metadata={"skip": True})

    def to_dict(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if not self.__dataclass_fields__[k].metadata.get("skip", False)
        }
