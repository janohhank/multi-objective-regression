from dataclasses import dataclass

import orjson
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup


@dataclass(kw_only=True)
class DeapTrainingResults(TrainingResult):
    pass

    @classmethod
    def from_dict(cls, **data: dict) -> "DeapTrainingResults":
        base_kwargs = cls._parse_base_fields(data)
        return cls(
            **base_kwargs,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DeapTrainingResults":
        data = orjson.loads(json_str)
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object for DeapTrainingResults")
        return cls.from_dict(**data)
