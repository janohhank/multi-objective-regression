import random
from copy import deepcopy

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup


class MutationCrossoverManager:
    __training_parameters: TrainingParameters = None

    def __init__(self, training_parameters: TrainingParameters):
        self.__training_parameters = training_parameters

    def features_mutation(self, training_result: TrainingResult) -> TrainingSetup:
        training_setup: TrainingSetup = training_result.training_setup
        new_training_setup: TrainingSetup = deepcopy(training_setup)
        new_training_setup.index = -1

        for feature in self.__training_parameters.features:
            if (
                random.uniform(0, 1)
                <= self.__training_parameters.mutation_feature_change_probability
            ):
                if feature in new_training_setup.features:
                    new_training_setup.features.remove(feature)
                else:
                    new_training_setup.features.append(feature)
        return new_training_setup

    def features_crossover(
        self,
        training_result_a: TrainingResult,
        training_result_b: TrainingResult,
    ) -> TrainingSetup:
        training_setup_a: TrainingSetup = training_result_a.training_setup
        training_setup_b: TrainingSetup = training_result_b.training_setup

        new_training_setup: TrainingSetup = TrainingSetup(-1, [])
        for feature in self.__training_parameters.features:
            if (
                random.uniform(0, 1)
                < self.__training_parameters.crossover_feature_selection_probability
            ):
                if feature in training_setup_a.features:
                    new_training_setup.features.append(feature)
            else:
                if feature in training_setup_b.features:
                    new_training_setup.features.append(feature)
        return new_training_setup
