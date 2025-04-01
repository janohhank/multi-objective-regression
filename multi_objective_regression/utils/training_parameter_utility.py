import math
from abc import ABC

from dto.training_parameters import TrainingParameters


class TrainingParameterUtility(ABC):

    @staticmethod
    def read_training_parameters(
        training_parameters_file_path: str,
    ) -> TrainingParameters:
        with open(training_parameters_file_path) as file:
            training_parameters_file_content = file.read()

        training_parameters: TrainingParameters = TrainingParameters.from_json(
            training_parameters_file_content
        )

        sum_of_weights: float = 0.0
        for weight in training_parameters.multi_objective_function_weights.values():
            sum_of_weights += weight
        if not math.isclose(sum_of_weights, 1.0):
            raise ValueError(
                f"The sum of multi-objective weights is not 1.0, instead of {sum_of_weights}."
            )

        return training_parameters
