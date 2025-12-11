from abc import abstractmethod, ABC

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


class TrainingManager(ABC):
    TYPE: str = None

    _training_parameters: TrainingParameters = None

    __x_train: DataFrame = None
    __y_train: DataFrame = None

    __x_validation: DataFrame = None
    __y_validation: DataFrame = None

    __x_test: DataFrame = None
    __y_test: DataFrame = None

    __pearson_correlation_matrix: DataFrame = None
    __pearson_correlation_to_target_feature: DataFrame = None

    def __init__(
        self,
        training_parameters: TrainingParameters,
    ):
        self._training_parameters = training_parameters

    def get_pearson_correlation_matrix(self) -> DataFrame:
        return self.__pearson_correlation_matrix

    def get_pearson_correlation_to_target_feature(self) -> DataFrame:
        return self.__pearson_correlation_to_target_feature

    def get_test_dataset(self):
        return self.__x_test, self.__y_test

    def prepare_dataset(self):
        print("Loading dataset.")

        if (
            self._training_parameters.validation_dataset is not None
            and self._training_parameters.test_dataset is not None
        ):
            print("Predefined train and validation datasets are available.")
            train_dataset: DataFrame = pandas.read_csv(
                self._training_parameters.train_dataset
            )
            self.__x_train = train_dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            self.__y_train = train_dataset[
                self._training_parameters.target_feature
            ].astype(int)

            validation_dataset: DataFrame = pandas.read_csv(
                self._training_parameters.validation_dataset
            )
            self.__x_validation = validation_dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            self.__y_validation = validation_dataset[
                self._training_parameters.target_feature
            ].astype(int)

            test_dataset: DataFrame = pandas.read_csv(
                self._training_parameters.test_dataset
            )
            self.__x_test = test_dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            self.__y_test = test_dataset[
                self._training_parameters.target_feature
            ].astype(int)

            print("Calculate pearson correlation matrix.")
            self.__pearson_correlation_matrix = train_dataset.corr()
            self.__pearson_correlation_to_target_feature: DataFrame = (
                self.__pearson_correlation_matrix[
                    self._training_parameters.target_feature
                ]
            )
        else:
            dataset: DataFrame = pandas.read_csv(
                self._training_parameters.train_dataset
            )
            x: DataFrame = dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            y: DataFrame = dataset[self._training_parameters.target_feature].astype(int)

            print("Calculate pearson correlation matrix.")
            self.__pearson_correlation_matrix = dataset.corr()
            self.__pearson_correlation_to_target_feature: DataFrame = (
                self.__pearson_correlation_matrix[
                    self._training_parameters.target_feature
                ]
            )

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
            "Train dataset target variable distribution:",
            {label: sum(self.__y_train == label) for label in set(self.__y_train)},
        )
        print(
            "Validation dataset target variable distribution:",
            {
                label: sum(self.__y_validation == label)
                for label in set(self.__y_validation)
            },
        )
        print(
            "Test dataset target variable distribution:",
            {label: sum(self.__y_test == label) for label in set(self.__y_test)},
        )

    @abstractmethod
    def start_training(self) -> dict[int, TrainingResult]:
        pass
