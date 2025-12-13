from dataclasses import dataclass

from dto.training_result import TrainingResult
import orjson


@dataclass(kw_only=True)
class MorseTrainingResults(TrainingResult):
    # Calculated regression coefficients
    coefficients: dict[str, float]
    # Calculated interception point
    interception: float
    # Number of iterations of the regression train
    iteration: int

    @classmethod
    def from_dict(cls, **data: dict) -> "MorseTrainingResults":
        base_kwargs = cls._parse_base_fields(data)
        return cls(
            **base_kwargs,
            coefficients=data.get("coefficients", {}) or {},
            interception=data.get("interception"),
            iteration=data.get("iteration"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MorseTrainingResults":
        data = orjson.loads(json_str)
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object for MorseTrainingResults")
        return cls.from_dict(**data)
