from dataclasses import dataclass, field, asdict

from dto.training_setup import TrainingSetup


@dataclass
class TrainingResult:
    index: int
    # Contains the selected features, and training details
    training_setup: TrainingSetup
    # Metric results on the validation dataset
    validation_results: dict[str, float]
    # Metric results on the test dataset
    test_results: dict[str, float]
    # Current regression training time in seconds
    training_time_seconds: float
    # Trained model and standard scaler binary objects
    model: object = field(default=None, metadata={"skip": True})
    scaler: object = field(default=None, metadata={"skip": True})

    def to_dict(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if not self.__dataclass_fields__[k].metadata.get("skip", False)
        }
