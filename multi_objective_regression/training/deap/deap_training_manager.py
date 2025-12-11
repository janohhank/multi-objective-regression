from dto.training_parameters import TrainingParameters
from training.training_manager import TrainingManager


class DeapTrainingManager(TrainingManager):

    def __init__(
        self,
        training_parameters: TrainingParameters,
    ):
        super().__init__(training_parameters)

        self.TYPE = "DEAP"
