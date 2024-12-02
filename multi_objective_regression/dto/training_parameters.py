import typing
from dataclasses import dataclass

from orjson import orjson


@dataclass
class TrainingParameters:
    train_dataset: typing.Optional[str] = None
    target_feature: typing.Optional[str] = None
    features: typing.Optional[list[str]] = None
    excluded_features: typing.Optional[list[list[str]]] = None
    allowed_features_count: typing.Optional[int] = None
    select_top_n_training: typing.Optional[int] = None
    mutation_and_crossover_iteration: typing.Optional[int] = None
    multi_objective_function_weights: typing.Optional[dict[str, float]] = None

    @staticmethod
    def from_json(json_str: str) -> "TrainingParameters":
        data = orjson.loads(json_str)

        return TrainingParameters(
            train_dataset=data["train_dataset"] if "train_dataset" in data else None,
            target_feature=data["target_feature"] if "target_feature" in data else None,
            features=data["features"] if "features" in data else [],
            excluded_features=(
                data["excluded_features"] if "excluded_features" in data else []
            ),
            allowed_features_count=(
                data["allowed_features_count"]
                if "allowed_features_count" in data
                else None
            ),
            select_top_n_training=(
                data["select_top_n_training"]
                if "select_top_n_training" in data
                else None
            ),
            mutation_and_crossover_iteration=(
                data["mutation_and_crossover_iteration"]
                if "mutation_and_crossover_iteration" in data
                else None
            ),
            multi_objective_function_weights=(
                data["multi_objective_function_weights"]
                if "multi_objective_function_weights" in data
                else {}
            ),
        )
