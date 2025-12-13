import typing
from dataclasses import dataclass

from dto.deap_parameters import DeapParameters
from dto.morse_parameters import MorseParameters
from orjson import orjson


@dataclass
class TrainingParameters:
    # Dataset properties
    train_dataset: typing.Optional[str]
    # Optional, calculated from train_dataset
    validation_dataset: typing.Optional[str]
    # Optional, calculated from train_dataset
    test_dataset: typing.Optional[str]

    # Selected meta-optimization algorithm(s) and those parameters
    algorithms: typing.Optional[list[str]]
    morse: MorseParameters
    deap: DeapParameters

    # Pre-defined objective function weights
    multi_objective_functions: typing.Optional[dict[str, float]]

    # Feature names
    target_feature: typing.Optional[str]
    features: typing.Optional[list[str]]

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
            multi_objective_functions=(
                data["multi_objective_functions"]
                if "multi_objective_functions" in data
                else {}
            ),
        )
