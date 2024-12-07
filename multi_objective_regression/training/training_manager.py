import random
import time
from copy import deepcopy
from typing import Any

import pandas
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from training.logistic_regression_training import (
    LogisticRegressionTraining,
)
from training.mutation_crossover_manager import MutationCrossoverManager
from utils.training_result_utility import TrainingResultUtility


class TrainingManager:
    __training_parameters: TrainingParameters = None
    __logistic_regression_training: LogisticRegressionTraining = None
    __mutation_crossover_manager: MutationCrossoverManager = None

    __scaled_x_train: DataFrame = None
    __y_train: DataFrame = None
    __scaled_x_test: DataFrame = None
    __y_test: DataFrame = None
    __covariance_to_target_feature: DataFrame = None

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

    def __get_standardization_scaler(self, dataset: DataFrame) -> Any:
        standard_scaler = StandardScaler()
        return standard_scaler.fit(dataset)

    def prepare_dataset(self):
        print("Loading train dataset.")
        dataset: DataFrame = pandas.read_csv(self.__training_parameters.train_dataset)
        x: DataFrame = dataset.drop(self.__training_parameters.target_feature, axis=1)
        y: DataFrame = dataset[self.__training_parameters.target_feature].astype(int)

        print("Calculate covariance matrix from standardized dataset.")
        full_dataset_scaler = self.__get_standardization_scaler(dataset)
        scaled_dataset = DataFrame(
            full_dataset_scaler.transform(dataset), columns=dataset.columns
        )
        covariance_matrix = scaled_dataset.cov()
        self.__covariance_to_target_feature: DataFrame = covariance_matrix[
            self.__training_parameters.target_feature
        ]

        print("Split train dataset into train and test.")
        x_train, x_test, self.__y_train, self.__y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=42
        )

        print("Create z-score standardization scaler from train dataset.")
        scaler = self.__get_standardization_scaler(x_train)

        print("Standardize train dataset.")
        scaled_x_train = scaler.transform(x_train)
        self.__scaled_x_train = DataFrame(scaled_x_train, columns=x_train.columns)

        print("Standardize test dataset.")
        scaled_x_test = scaler.transform(x_test)
        self.__scaled_x_test = DataFrame(scaled_x_test, columns=x_test.columns)

    def start_training(
        self, training_setups: dict[int, TrainingResult]
    ) -> dict[int, TrainingResult]:
        start: float = time.perf_counter()

        print(
            f"Start multi-objective logistic regression training on {len(training_setups)} training setups."
        )
        train_results = {}
        for index, training_setup in training_setups.items():
            train_results[index] = self.__logistic_regression_training.train(
                index,
                training_setup,
                self.__covariance_to_target_feature,
                self.__scaled_x_train,
                self.__y_train,
                self.__scaled_x_test,
                self.__y_test,
            )
        elapsed: float = time.perf_counter() - start
        print(f"Whole training done in {elapsed} seconds.")

        return train_results

    def start_mutation_and_crossover(
        self,
        current_training_index: int,
        initial_training_top_n_results: dict[int, TrainingResult],
    ) -> dict[int, TrainingResult]:
        final_training_results: dict[int, TrainingResult] = deepcopy(
            initial_training_top_n_results
        )

        new_training_index: int = current_training_index
        for _ in range(self.__training_parameters.mutation_and_crossover_iteration):
            new_training_index += 1

            if random.uniform(0, 1) < 0.5:
                new_training_setup: TrainingResult = (
                    self.__mutation_crossover_manager.features_mutation(
                        new_training_index,
                        random.choice(list(final_training_results.values())),
                    )
                )
            else:
                new_training_setup: TrainingResult = (
                    self.__mutation_crossover_manager.features_crossover(
                        new_training_index,
                        random.choice(list(final_training_results.values())),
                        random.choice(list(final_training_results.values())),
                    )
                )

            if new_training_setup is None:
                continue

            training_results: dict[int, TrainingResult] = self.start_training(
                {new_training_index: new_training_setup}
            )
            final_training_results: dict[int, TrainingResult] = (
                TrainingResultUtility.merge_training_results(
                    new_training_index,
                    training_results[new_training_index],
                    final_training_results,
                )
            )
        return final_training_results
