import itertools

from dto.training_parameters import TrainingParameters
from dto.training_setup import TrainingSetup


class TrainingSetupGenerator:
    __training_parameters: TrainingParameters = None

    def __init__(self, training_parameters: TrainingParameters) -> None:
        self.__training_parameters = training_parameters

    def generate_training_setups(self) -> dict[int, TrainingSetup]:
        training_setups = {}

        self.__generate_feature_combinations(training_setups)

        return training_setups

    def __generate_feature_combinations(self, training_setups):
        all_feature_combinations: list[tuple] = list(
            itertools.combinations(
                self.__training_parameters.features,
                self.__training_parameters.allowed_features_count,
            )
        )

        index: int = 1
        for combination in all_feature_combinations:
            if len(self.__training_parameters.excluded_features) > 0:
                skip: bool = False
                for exclusion_tuple in self.__training_parameters.excluded_features:
                    if set(exclusion_tuple).issubset(set(combination)):
                        skip = True

                if skip:
                    continue

            training_setups[index] = TrainingSetup(index, list(combination))
            index = index + 1
