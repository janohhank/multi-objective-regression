from dataclasses import dataclass, field, asdict

from dto.training_setup import TrainingSetup
from orjson import orjson


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

    @staticmethod
    def from_json(json_str: str) -> "TrainingResult":
        data = orjson.loads(json_str)

        return TrainingResult(
            index=data["index"] if "index" in data else None,
            training_setup=data["training_setup"] if "training_setup" in data else {},
            coefficients=data["coefficients"] if "coefficients" in data else {},
            interception=(data["interception"] if "interception" in data else []),
            iteration=(data["iteration"] if "iteration" in data else None),
            validation_results=(
                data["validation_results"] if "validation_results" in data else {}
            ),
            test_results=(data["test_results"] if "test_results" in data else {}),
            training_time_seconds=(
                data["training_time_seconds"]
                if "training_time_seconds" in data
                else None
            ),
            model=None,
            scaler=None,
        )
