from abc import abstractmethod, ABC

import pandas
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from utils.training_result_utility import TrainingResultUtility


class TrainingManager(ABC):
    TYPE: str = None

    _training_parameters: TrainingParameters = None

    _x_train: DataFrame = None
    _y_train: DataFrame = None

    _x_validation: DataFrame = None
    _y_validation: DataFrame = None

    _x_test: DataFrame = None
    _y_test: DataFrame = None

    _pearson_correlation_matrix: DataFrame = None
    _pearson_correlation_to_target_feature: Series = None

    def __init__(
        self,
        training_parameters: TrainingParameters,
    ):
        self._training_parameters = training_parameters

    def get_pearson_correlation_matrix(self) -> DataFrame:
        return self._pearson_correlation_matrix

    def get_pearson_correlation_to_target_feature(self) -> Series:
        return self._pearson_correlation_to_target_feature

    def get_test_dataset(self):
        return self._x_test, self._y_test

    def get_validation_dataset(self):
        return self._x_validation, self._y_validation

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
            self._x_train = train_dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            self._y_train = train_dataset[
                self._training_parameters.target_feature
            ].astype(int)

            validation_dataset: DataFrame = pandas.read_csv(
                self._training_parameters.validation_dataset
            )
            self._x_validation = validation_dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            self._y_validation = validation_dataset[
                self._training_parameters.target_feature
            ].astype(int)

            test_dataset: DataFrame = pandas.read_csv(
                self._training_parameters.test_dataset
            )
            self._x_test = test_dataset.drop(
                self._training_parameters.target_feature, axis=1
            )
            self._y_test = test_dataset[
                self._training_parameters.target_feature
            ].astype(int)

            print("Calculate pearson correlation matrix.")
            self._pearson_correlation_matrix = train_dataset.corr(numeric_only=True)
            self._pearson_correlation_to_target_feature: Series = (
                self._pearson_correlation_matrix[
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
            self._pearson_correlation_matrix = dataset.corr(numeric_only=True)
            self._pearson_correlation_to_target_feature: Series = (
                self._pearson_correlation_matrix[
                    self._training_parameters.target_feature
                ]
            )

            print("Split dataset into train, validation and test.")
            self._x_train, x_temp, self._y_train, y_temp = train_test_split(
                x, y, test_size=0.3, stratify=y, random_state=42
            )

            self._x_validation, self._x_test, self._y_validation, self._y_test = (
                train_test_split(
                    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
                )
            )

        print(
            "Train dataset target variable distribution:",
            {label: sum(self._y_train == label) for label in set(self._y_train)},
        )
        print(
            "Validation dataset target variable distribution:",
            {
                label: sum(self._y_validation == label)
                for label in set(self._y_validation)
            },
        )
        print(
            "Test dataset target variable distribution:",
            {label: sum(self._y_test == label) for label in set(self._y_test)},
        )

    @abstractmethod
    def start_training(self, result_directory: str) -> dict[int, TrainingResult]:
        pass
