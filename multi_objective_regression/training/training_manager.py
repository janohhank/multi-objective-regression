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
from training.mutation_crossover import MutationCrossover
from utils.training_result_utility import TrainingResultUtility


class TrainingManager:
    __training_parameters: TrainingParameters = None
    __logistic_regression_training: LogisticRegressionTraining = None
    __mutation_crossover: MutationCrossover = None

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
        self.__mutation_crossover = MutationCrossover()

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

        print("Start multi-objective regression training.")
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
        self, start_index: int, population_top_n_results: dict[int, TrainingResult]
    ) -> dict[int, TrainingResult]:
        final_training_results: dict[int, TrainingResult] = deepcopy(
            population_top_n_results
        )

        new_train_index: int = start_index + 1
        for _ in range(self.__training_parameters.mutation_and_crossover_iteration):
            training_setups = {}
            for index, training_result in population_top_n_results.items():
                if random.uniform(0, 1) < 0.5:
                    current_result: TrainingResult = (
                        self.__mutation_crossover.features_mutation(training_result)
                    )
                    if current_result is not None:
                        training_setups[new_train_index] = current_result
                else:
                    current_result: TrainingResult = (
                        self.__mutation_crossover.features_crossover(
                            training_result,
                            random.choice(list(population_top_n_results.values())),
                        )
                    )
                    training_setups[new_train_index] = current_result
                new_train_index += 1

            training_results: dict[int, TrainingResult] = self.start_training(
                training_setups
            )
            TrainingResultUtility.merge_training_results(
                training_results, final_training_results
            )

        return final_training_results

    def __get_standardization_scaler(self, dataset: DataFrame) -> Any:
        standard_scaler = StandardScaler()
        return standard_scaler.fit(dataset)
