import random
from collections import Counter
from copy import deepcopy

import pandas
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from training.logistic_regression_training import (
    LogisticRegressionTraining,
)
from training.mutation_crossover_manager import MutationCrossoverManager
from utils.training_result_utility import TrainingResultUtility


class TrainingManager:
    __training_parameters: TrainingParameters = None

    # Supervised machine learning trainer
    __logistic_regression_training: LogisticRegressionTraining = None
    # Meta training component with genetic algorithm
    __mutation_crossover_manager: MutationCrossoverManager = None

    __x_train: DataFrame = None
    __y_train: DataFrame = None

    __x_validation: DataFrame = None
    __y_validation: DataFrame = None

    __x_test: DataFrame = None
    __y_test: DataFrame = None

    __correlation_matrix = None
    __correlation_to_target_feature: DataFrame = None

    def __init__(
        self,
        training_parameters: TrainingParameters,
    ):
        self.__training_parameters = training_parameters
        self.__logistic_regression_training = LogisticRegressionTraining(
            self.__training_parameters
        )
        self.__mutation_crossover_manager = MutationCrossoverManager(
            self.__training_parameters
        )

    def get_correlation_matrix(self):
        return self.__correlation_matrix

    def prepare_dataset(self):
        print("Loading train dataset.")
        dataset: DataFrame = pandas.read_csv(self.__training_parameters.train_dataset)
        x: DataFrame = dataset.drop(self.__training_parameters.target_feature, axis=1)
        y: DataFrame = dataset[self.__training_parameters.target_feature].astype(int)

        print("Calculate correlation matrix.")
        self.__correlation_matrix = dataset.corr()
        self.__correlation_to_target_feature: DataFrame = self.__correlation_matrix[
            self.__training_parameters.target_feature
        ]

        print("Split dataset into train, validation and test.")
        self.__x_train, x_temp, self.__y_train, y_temp = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=42
        )

        self.__x_validation, self.__x_test, self.__y_validation, self.__y_test = (
            train_test_split(
                x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
        )

        print(
            "Train class distribution:",
            {label: sum(self.__y_train == label) for label in set(self.__y_train)},
        )
        print(
            "Validation class distribution:",
            {
                label: sum(self.__y_validation == label)
                for label in set(self.__y_validation)
            },
        )
        print(
            "Test class distribution:",
            {label: sum(self.__y_test == label) for label in set(self.__y_test)},
        )

    def start_training(
        self, training_setups: dict[int, TrainingResult]
    ) -> dict[int, TrainingResult]:
        train_results = {}
        for index, training_setup in training_setups.items():
            train_results[index] = self.__logistic_regression_training.train(
                index,
                training_setup,
                self.__correlation_to_target_feature,
                self.__x_train,
                self.__y_train,
                self.__x_validation,
                self.__y_validation,
                self.__x_test,
                self.__y_test,
            )
        return train_results

    def start_mutation_and_crossover(
        self,
        current_training_index: int,
        initial_training_top_n_results: dict[int, TrainingResult],
        all_training_results: dict[int, TrainingResult],
    ) -> (dict[int, TrainingResult], dict[int, TrainingResult]):
        final_top_n_training_results: dict[int, TrainingResult] = deepcopy(
            initial_training_top_n_results
        )

        training_setup_workspace: dict[int, TrainingSetup] = {}
        new_candidate_training_index: int = current_training_index
        for _ in range(self.__training_parameters.mutation_and_crossover_iteration):
            if (
                random.uniform(0, 1)
                < self.__training_parameters.mutation_and_crossover_balance
            ):
                new_training_setup: TrainingResult = (
                    self.__mutation_crossover_manager.features_mutation(
                        random.choice(list(final_top_n_training_results.values())),
                    )
                )
            else:
                new_training_setup: TrainingResult = (
                    self.__mutation_crossover_manager.features_crossover(
                        random.choice(list(final_top_n_training_results.values())),
                        random.choice(list(final_top_n_training_results.values())),
                    )
                )

            if (
                self.__is_valid_training_setup(new_training_setup, all_training_results)
                is True
            ):
                new_candidate_training_index += 1
                new_training_setup.index = new_candidate_training_index
                training_setup_workspace[new_candidate_training_index] = (
                    new_training_setup
                )

            if len(training_setup_workspace) == 0:
                continue

            training_results: dict[int, TrainingResult] = self.start_training(
                training_setup_workspace
            )

            for index, training_result in training_results.items():
                all_training_results[index] = training_result
                final_top_n_training_results: dict[int, TrainingResult] = (
                    TrainingResultUtility.merge_training_results(
                        index,
                        training_result,
                        final_top_n_training_results,
                    )
                )
            training_setup_workspace.clear()

            training_setup_candidates: list[TrainingSetup] = (
                self.__logistic_regression_training.suggest_training_setup_candidates(
                    training_results
                )
            )
            for training_setup in training_setup_candidates:
                if (
                    self.__is_valid_training_setup(training_setup, all_training_results)
                    is True
                ):
                    new_candidate_training_index += 1
                    training_setup.index = new_candidate_training_index
                    training_setup_workspace[new_candidate_training_index] = (
                        training_setup
                    )

        print(f"Overall inspected training setups: {new_candidate_training_index}")
        return final_top_n_training_results, all_training_results

    def __is_valid_training_setup(
        self,
        new_training_setup: TrainingSetup,
        all_training_results: dict[int, TrainingResult],
    ) -> bool:
        # Skip if generated an empty feature set.
        if new_training_setup is None or not new_training_setup.features:
            return False

        # Skip if the generated feature set contains an excluded feature combination.
        skip: bool = False
        for exclusion_tuple in self.__training_parameters.excluded_feature_sets:
            if set(exclusion_tuple).issubset(set(new_training_setup.features)):
                skip = True

        if skip:
            return False

        # Skip if this combination is already tried.
        found_same = False
        for training_result in all_training_results.values():
            if Counter(training_result.training_setup.features) == Counter(
                new_training_setup.features
            ):
                found_same = True
                break
        if found_same:
            return False

        return True
