import time
from typing import Any

import pandas
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from training.logistic_regression_training import (
    LogisticRegressionTraining,
)


class TrainingManager:
    __logistic_regression_training: LogisticRegressionTraining = None
    __training_parameters: TrainingParameters = None
    __training_setups: dict[int, TrainingSetup] = None

    def __init__(
        self,
        training_parameters: TrainingParameters,
        training_setups: dict[int, TrainingSetup],
    ):
        self.__logistic_regression_training = LogisticRegressionTraining(
            training_parameters
        )
        self.__training_parameters = training_parameters
        self.__training_setups = training_setups

    def start_training(self) -> dict[int, TrainingResult]:
        start: float = time.perf_counter()

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
        covariance_to_target_feature = covariance_matrix[
            self.__training_parameters.target_feature
        ]

        print("Split train dataset into train and test.")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=42
        )

        print("Create z-score standardization scaler from train dataset.")
        scaler = self.__get_standardization_scaler(x_train)

        print("Standardize train dataset.")
        scaled_x_train = scaler.transform(x_train)
        scaled_x_train = DataFrame(scaled_x_train, columns=x_train.columns)

        print("Standardize test dataset.")
        scaled_x_test = scaler.transform(x_test)
        scaled_x_test = DataFrame(scaled_x_test, columns=x_test.columns)

        print("Start multi-objective regression training.")
        train_results = {}
        for index, training_setup in self.__training_setups.items():
            train_results[index] = self.__logistic_regression_training.train(
                index,
                training_setup,
                covariance_to_target_feature,
                scaled_x_train,
                y_train,
                scaled_x_test,
                y_test,
            )
        elapsed: float = time.perf_counter() - start
        print(f"Whole training done in {elapsed} seconds.")

        return train_results

    def __get_standardization_scaler(self, dataset: DataFrame) -> Any:
        standard_scaler = StandardScaler()
        return standard_scaler.fit(dataset)
