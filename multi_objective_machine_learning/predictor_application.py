import argparse
import glob
import os.path

import joblib
import pandas
from numpy import ndarray
from pandas import DataFrame, Series

from dto.training_result import TrainingResult
from training.objective_components import (
    AccuracyComponent,
    PrecisionComponent,
    RecallComponent,
    SpecificityComponent,
    F1ScoreComponent,
    PrAucComponent,
    GiniComponent,
    RocAucComponent,
)


class MultiObjectivePredictorApplication:
    __MODEL_PATH: str = None
    __TEST_DATASET_PATH: str = None
    __THRESHOLD: float = None

    def __init__(
        self, model_path: str, test_dataset_path: str, threshold: float = 0.5
    ) -> None:
        self.__MODEL_PATH = model_path
        self.__TEST_DATASET_PATH = test_dataset_path
        self.__THRESHOLD = threshold

    def start_prediction(self) -> None:
        print(
            f"Getting the model, standard scaler and the selected features from {self.__MODEL_PATH}."
        )
        model = joblib.load(os.path.join(self.__MODEL_PATH, "model.pkl"))
        scaler = joblib.load(os.path.join(self.__MODEL_PATH, "scaler.pkl"))

        pattern = os.path.join(self.__MODEL_PATH, "*_result.json")
        matching_files = glob.glob(pattern)
        with open(matching_files[0]) as file:
            training_result_file_content: str = file.read()

        training_result: TrainingResult = TrainingResult.from_json(
            training_result_file_content
        )

        print(
            f"Loading and preparing the test dataset from {self.__TEST_DATASET_PATH}."
        )
        test_dataset: DataFrame = pandas.read_csv(self.__TEST_DATASET_PATH)
        x_test_reduced: Series = test_dataset[
            training_result.training_setup.features
        ].copy()
        x_test_scaled: ndarray = scaler.transform(x_test_reduced)
        y_test: DataFrame = test_dataset[
            training_result.training_setup.target_feature
        ].copy()

        print("Predicting on the test set.")
        y_probs: ndarray = model.predict_proba(x_test_scaled)[:, 1]

        print(f"Calculating results with threshold {self.__THRESHOLD}.")
        print(
            f"Accuracy: {AccuracyComponent().score(y_test, y_probs, self.__THRESHOLD)}"
        )
        print(
            f"Precision: {PrecisionComponent().score(y_test, y_probs, self.__THRESHOLD)}"
        )
        print(f"Recall: {RecallComponent().score(y_test, y_probs, self.__THRESHOLD)}")
        print(
            f"Specificity: {SpecificityComponent().score(y_test, y_probs, self.__THRESHOLD)}"
        )
        print(
            f"F1 score: {F1ScoreComponent().score(y_test, y_probs, self.__THRESHOLD)}"
        )
        print(f"ROC-AUC: {RocAucComponent().score(y_test, y_probs)}")
        print(f"PR-AUC: {PrAucComponent().score(y_test, y_probs)}")
        print(f"Gini score: {GiniComponent().score(y_test, y_probs)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictor application for multi-objective regression training models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Saved model path.",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
        help="Test dataset path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.5,
        help="Class divisor threshold.",
    )

    args = parser.parse_args()

    application: MultiObjectivePredictorApplication = (
        MultiObjectivePredictorApplication(
            args.model_path, args.test_dataset_path, args.threshold
        )
    )
    application.start_prediction()


if __name__ == "__main__":
    main()
