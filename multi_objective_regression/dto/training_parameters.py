import typing
from dataclasses import dataclass

from orjson import orjson


@dataclass
class TrainingParameters:
    train_dataset: typing.Optional[str] = None
    target_feature: typing.Optional[str] = None
    features: typing.Optional[list[str]] = None
    excluded_feature_sets: typing.Optional[list[list[str]]] = None
    initial_training_feature_sets_count: typing.Optional[int] = None
    initial_training_top_n_selection_count: typing.Optional[int] = None
    mutation_and_crossover_iteration: typing.Optional[int] = None
    mutation_feature_change_probability: typing.Optional[float] = None
    crossover_feature_selection_probability: typing.Optional[float] = None
    multi_objective_function_weights: typing.Optional[dict[str, float]] = None

    @staticmethod
    def from_json(json_str: str) -> "TrainingParameters":
        data = orjson.loads(json_str)

        return TrainingParameters(
            train_dataset=data["train_dataset"] if "train_dataset" in data else None,
            target_feature=data["target_feature"] if "target_feature" in data else None,
            features=data["features"] if "features" in data else [],
            excluded_feature_sets=(
                data["excluded_feature_sets"] if "excluded_feature_sets" in data else []
            ),
            initial_training_feature_sets_count=(
                data["initial_training_feature_sets_count"]
                if "initial_training_feature_sets_count" in data
                else None
            ),
            initial_training_top_n_selection_count=(
                data["initial_training_top_n_selection_count"]
                if "initial_training_top_n_selection_count" in data
                else None
            ),
            mutation_and_crossover_iteration=(
                data["mutation_and_crossover_iteration"]
                if "mutation_and_crossover_iteration" in data
                else None
            ),
            mutation_feature_change_probability=(
                data["mutation_feature_change_probability"]
                if "mutation_feature_change_probability" in data
                else None
            ),
            crossover_feature_selection_probability=(
                data["crossover_feature_selection_probability"]
                if "crossover_feature_selection_probability" in data
                else None
            ),
            multi_objective_function_weights=(
                data["multi_objective_function_weights"]
                if "multi_objective_function_weights" in data
                else {}
            ),
        )
