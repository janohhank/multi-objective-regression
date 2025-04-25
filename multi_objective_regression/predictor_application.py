import argparse
import glob
import os.path

import joblib
import pandas
from pandas import DataFrame
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from dto.training_result import TrainingResult


class MultiObjectivePredictorApplication:
    __MODEL_PATH: str = None
    __DATASET_PATH: str = None

    def __init__(self, model_path: str, dataset_path: str) -> None:
        self.__MODEL_PATH = model_path
        self.__DATASET_PATH = dataset_path

    def start_prediction(self) -> None:
        model = joblib.load(os.path.join(self.__MODEL_PATH, "model.pkl"))
        scaler = joblib.load(os.path.join(self.__MODEL_PATH, "scaler.pkl"))

        pattern = os.path.join(self.__MODEL_PATH, "*_result.json")
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            with open(file_path) as file:
                training_result_file_content = file.read()

        training_result: TrainingResult = TrainingResult.from_json(
            training_result_file_content
        )

        test_dataset: DataFrame = pandas.read_csv(self.__DATASET_PATH)
        x_test_reduced = test_dataset[training_result.training_setup["features"]].copy()
        x_test_scaled = scaler.transform(x_test_reduced)
        y_test: DataFrame = test_dataset[
            training_result.training_setup["target_feature"]
        ].copy()

        y_probs = model.predict_proba(x_test_scaled)[:, 1]
        y_pred = model.predict(x_test_scaled)

        accuracy: float = accuracy_score(y_test, y_pred)
        precision: float = precision_score(y_test, y_pred)
        recall: float = recall_score(y_test, y_pred)
        f1_score_value: float = f1_score(y_test, y_pred)
        roc_auc: float = roc_auc_score(y_test, y_probs)
        pr_auc: float = average_precision_score(y_test, y_probs)
        gini_score: float = 2 * roc_auc - 1

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1_score_value}")
        print(f"ROC-AUC: {roc_auc}")
        print(f"PR-AUC: {pr_auc}")
        print(f"Gini score: {gini_score}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-objective predictor application."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Saved model path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset path.",
    )

    args = parser.parse_args()

    application: MultiObjectivePredictorApplication = (
        MultiObjectivePredictorApplication(args.model_path, args.dataset_path)
    )
    application.start_prediction()


if __name__ == "__main__":
    main()
