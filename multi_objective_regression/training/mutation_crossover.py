import random
from copy import deepcopy

from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup


class MutationCrossover:

    def features_mutation(self, training_result: TrainingResult) -> TrainingSetup:
        training_setup: TrainingSetup = training_result.training_setup
        new_training_setup: TrainingSetup = deepcopy(training_setup)
        new_training_setup.features = []

        for feature in training_setup.features:
            if random.uniform(0, 1) < 0.5:
                new_training_setup.features.append(feature)

        if len(new_training_setup.features) == 0:
            return None

        return new_training_setup

    def features_crossover(
        self, training_result_a: TrainingResult, training_result_b: TrainingResult
    ) -> TrainingSetup:
        training_setup_a: TrainingSetup = training_result_a.training_setup
        training_setup_b: TrainingSetup = training_result_b.training_setup

        new_training_setup: TrainingSetup = deepcopy(training_setup_a)
        new_training_setup.features = []

        for feature_index in range(
            min(len(training_setup_a.features), len(training_setup_b.features))
        ):
            if random.uniform(0, 1) < 0.5:
                new_training_setup.features.append(
                    training_setup_a.features[feature_index]
                )
            else:
                new_training_setup.features.append(
                    training_setup_b.features[feature_index]
                )

        return new_training_setup
