import itertools
from abc import ABC

from dto.training_parameters import TrainingParameters
from dto.training_setup import TrainingSetup


class TrainingSetupGenerator(ABC):

    @staticmethod
    def generate_training_setups(
        training_parameters: TrainingParameters,
    ) -> dict[int, TrainingSetup]:
        training_setups = {}

        TrainingSetupGenerator.__generate_feature_combinations(
            training_parameters, training_setups
        )

        return training_setups

    # milyen valószínűséggel kerüljön be egy feautre az alap training set-be ehhez egy prob parameter
    # itt is figyelni, hogy ne legyen duplikátum, kizárási feltételekre
    # new variable max_initial_training_count = 1000
    @staticmethod
    def __generate_feature_combinations(
        training_parameters: TrainingParameters, training_setups
    ) -> None:
        all_feature_combinations: list[tuple] = list(
            itertools.combinations(
                training_parameters.features,
                training_parameters.initial_training_feature_sets_count,
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
