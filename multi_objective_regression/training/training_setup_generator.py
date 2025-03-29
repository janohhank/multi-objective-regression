import itertools
import random
from abc import ABC
from collections import Counter

from dto.training_parameters import TrainingParameters
from dto.training_setup import TrainingSetup


class TrainingSetupGenerator(ABC):
    __MIN_NUMBER_OF_FEATURES: int = 4

    @staticmethod
    def generate_training_setups(
        training_parameters: TrainingParameters,
    ) -> dict[int, TrainingSetup]:
        training_setups = {}
        match training_parameters.initial_training_setup_generator_type:
            case "ALL_COMBINATIONS":
                training_setups = (
                    TrainingSetupGenerator.__generate_all_feature_combinations(
                        training_parameters
                    )
                )
            case "RANDOM_COMBINATIONS":
                training_setups = (
                    TrainingSetupGenerator.__generate_n_random_feature_combinations(
                        training_parameters
                    )
                )

        feature_size_counts = Counter(
            len(training_setup.features) for training_setup in training_setups.values()
        )
        for size, count in feature_size_counts.items():
            print(
                f"Generated initial feature setup(s): {count} piece(s) of {size} feature size."
            )

        return training_setups

    # deprecated
    @staticmethod
    def __generate_all_feature_combinations(
        training_parameters: TrainingParameters,
    ) -> dict[int, TrainingSetup]:
        training_setups = {}

        all_feature_combinations: list[tuple] = list(
            itertools.combinations(
                training_parameters.features,
                TrainingSetupGenerator.__MIN_NUMBER_OF_FEATURES,
            )
        )

        index: int = 1
        for combination in all_feature_combinations:
            if len(training_parameters.excluded_feature_sets) > 0:
                skip: bool = False
                for exclusion_tuple in training_parameters.excluded_feature_sets:
                    if set(exclusion_tuple).issubset(set(combination)):
                        skip = True

                if skip:
                    continue

            training_setups[index] = TrainingSetup(index, list(combination))
            index = index + 1

        return training_setups

    @staticmethod
    def __generate_n_random_feature_combinations(
        training_parameters: TrainingParameters,
    ) -> dict[int, TrainingSetup]:
        training_setups = {}

        index: int = 1
        used_combinations = set()
        while len(training_setups) < training_parameters.initial_training_setup_count:
            size = random.choice(
                range(
                    TrainingSetupGenerator.__MIN_NUMBER_OF_FEATURES,
                    len(training_parameters.features) + 1,
                )
            )
            combination = tuple(random.sample(training_parameters.features, size))

            if combination not in used_combinations:
                used_combinations.add(combination)

                skip: bool = False
                for exclusion_tuple in training_parameters.excluded_feature_sets:
                    if set(exclusion_tuple).issubset(set(combination)):
                        skip = True

                if skip:
                    continue

                training_setups[index] = TrainingSetup(index, list(combination))
                index = index + 1

        return training_setups
