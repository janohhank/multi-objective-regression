from dataclasses import dataclass, field, asdict

from dto.training_setup import TrainingSetup
import orjson


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

    @staticmethod
    def _parse_base_fields(data: dict) -> dict:
        return {
            "index": data.get("index"),
            "training_setup": (
                TrainingSetup.from_dict(**data["training_setup"])
                if "training_setup" in data and data["training_setup"] is not None
                else None
            ),
            "validation_results": data.get("validation_results", {}) or {},
            "test_results": data.get("test_results", {}) or {},
            "training_time_seconds": data.get("training_time_seconds"),
        }

    @classmethod
    def from_dict(cls, **data) -> "TrainingResult":
        base_kwargs = cls._parse_base_fields(data)
        return cls(**base_kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "TrainingResult":
        data = orjson.loads(json_str)
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object for TrainingResult")
        return cls.from_dict(**data)
