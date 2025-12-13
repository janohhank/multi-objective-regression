from dataclasses import dataclass


@dataclass
class MorseParameters:
    excluded_feature_sets: list[list[str]]
    initial_training_setup_generator_type: str
    initial_training_setup_count: int
    initial_training_top_n_selection_count: int
    mutation_and_crossover_iteration: int
    mutation_and_crossover_balance: float
    mutation_feature_change_probability: float
    crossover_feature_selection_probability: float

    @staticmethod
    def from_dict(**data: dict) -> "MorseParameters":
        return MorseParameters(
            excluded_feature_sets=(
                data["excluded_feature_sets"] if "excluded_feature_sets" in data else []
            ),
            initial_training_setup_generator_type=(
                data["initial_training_setup_generator_type"]
                if "initial_training_setup_generator_type" in data
                else None
            ),
            initial_training_setup_count=(
                data["initial_training_setup_count"]
                if "initial_training_setup_count" in data
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
            mutation_and_crossover_balance=(
                data["mutation_and_crossover_balance"]
                if "mutation_and_crossover_balance" in data
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
        )
