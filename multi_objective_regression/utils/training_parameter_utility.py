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

        return training_parameters
