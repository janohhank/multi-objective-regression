import typing
from dataclasses import dataclass

from dto.deap_parameters import DeapParameters
from dto.morse_parameters import MorseParameters
from orjson import orjson


@dataclass
class TrainingParameters:
    # Dataset properties
    train_dataset: typing.Optional[str] = None
    # Optional, calculated from train_dataset
    validation_dataset: typing.Optional[str] = None
    # Optional, calculated from train_dataset
    test_dataset: typing.Optional[str] = None

    # Meta-optimization algorithms and parameters
    algorithms: typing.Optional[list[str]] = None
    morse: MorseParameters = None
    deap: DeapParameters = None

    # Pre-defined objective function weights
    multi_objective_function_weights: typing.Optional[dict[str, float]] = None

    target_feature: typing.Optional[str] = None
    features: typing.Optional[list[str]] = None

    @staticmethod
    def from_json(json_str: str) -> "TrainingParameters":
        data = orjson.loads(json_str)

        return TrainingParameters(
            train_dataset=data["train_dataset"] if "train_dataset" in data else None,
            validation_dataset=(
                data["validation_dataset"] if "validation_dataset" in data else None
            ),
            test_dataset=data["test_dataset"] if "test_dataset" in data else None,
            algorithms=data["algorithms"] if "algorithms" in data else [],
            morse=(
                MorseParameters.from_dict(**data["morse"]) if "morse" in data else {}
            ),
            deap=(DeapParameters.from_dict(**data["deap"]) if "deap" in data else {}),
            target_feature=data["target_feature"] if "target_feature" in data else None,
            features=data["features"] if "features" in data else [],
            multi_objective_function_weights=(
                data["multi_objective_function_weights"]
                if "multi_objective_function_weights" in data
                else {}
            ),
        )
