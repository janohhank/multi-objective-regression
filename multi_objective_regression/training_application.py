import argparse
import os
import time
from datetime import datetime

from pandas import DataFrame

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_results import TrainingResults
from training.deap.deap_training_manager import (
    DeapTrainingManager,
)
from training.morse.morse_training_manager import (
    MorseTrainingManager,
)
from training.training_manager import TrainingManager
from utils.plot_utility import PlotUtility
from utils.training_parameter_utility import (
    TrainingParameterUtility,
)
from utils.training_result_utility import (
    TrainingResultUtility,
)
from utils.training_utility import TrainingUtility


class MultiObjectiveTrainingApplication:
    __training_parameters: TrainingParameters = None

    def __init__(self, training_parameters_file_path: str) -> None:
        self.__training_parameters: TrainingParameters = (
            TrainingParameterUtility.read_training_parameters(
                training_parameters_file_path
            )
        )

    def start(self):
        training_results: TrainingResults = TrainingResults(
            self.__training_parameters, {}
        )
        for algorithm in self.__training_parameters.algorithms:
            if algorithm == "MORSE":
                training_manager = MorseTrainingManager(self.__training_parameters)
            elif algorithm == "DEAP":
                training_manager = DeapTrainingManager(self.__training_parameters)
            else:
                print(f"[ERROR] Unknown algorithm type: {algorithm.name}")
                continue
            training_results.results[algorithm] = self.__start_training(
                training_manager
            )

    def __start_training(self, training_manager: TrainingManager) -> TrainingResult:
        print("Start training.")
        start: float = time.perf_counter()

        # Main result directory is the date time
        main_result_directory: str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_directory: str = os.path.join(
            main_result_directory, training_manager.TYPE
        )
        os.makedirs(result_directory)

        # Prepare datasets.
        training_manager.prepare_dataset()

        # Start training with meta-optimization.
        top_training_results: dict[int, TrainingResult] = (
            training_manager.start_training(result_directory)
        )

        print("Saving the best model.")
        best_model: TrainingResult = TrainingResultUtility.get_best_training_result(
            top_training_results
        )
        TrainingResultUtility.save_model(
            result_directory,
            "model",
            best_model,
        )

        # Prepare test dataset with selected features
        test_dataset: tuple[DataFrame, DataFrame] = training_manager.get_test_dataset()
        x_test_reduced = test_dataset[0][best_model.training_setup.features].copy()
        x_test_scaled = best_model.scaler.transform(x_test_reduced)

        # Prepare validation dataset with selected features
        validation_dataset: tuple[DataFrame, DataFrame] = (
            training_manager.get_validation_dataset()
        )
        x_validation_reduced = validation_dataset[0][
            best_model.training_setup.features
        ].copy()
        x_validation_scaled = best_model.scaler.transform(x_validation_reduced)

        print("Saving the test dataset.")
        TrainingUtility.save_dataset(
            result_directory,
            "test_dataset",
            test_dataset,
        )

        print("Saving plots.")
        PlotUtility.plot_metric_comparison(result_directory, 4, top_training_results)

        # Using test dataset
        y_probs = best_model.model.predict_proba(x_test_scaled)[:, 1]
        y_pred = best_model.model.predict(x_test_scaled)
        y_test = test_dataset[1]
        PlotUtility.plot_roc_curve(result_directory, "model", y_test, y_probs)
        PlotUtility.plot_pr_curve(result_directory, "model", y_test, y_probs)
        PlotUtility.plot_confusion_matrix(result_directory, "model", y_test, y_pred)

        # Using validation dataset
        y_probs = best_model.model.predict_proba(x_validation_scaled)[:, 1]
        y_validation = validation_dataset[1]
        PlotUtility.plot_specificity_sensitivity_curve(
            result_directory, "model", y_validation, y_probs
        )

        elapsed: float = time.perf_counter() - start
        print(f"Finished training on {elapsed:.2f} seconds.")

        return best_model


def main():
    parser = argparse.ArgumentParser(description="Multi-objective training.")
    parser.add_argument(
        "--training_parameters_path",
        type=str,
        required=True,
        help="Training parameters descriptor (JSON) file.",
    )

    args = parser.parse_args()

    application: MultiObjectiveTrainingApplication = MultiObjectiveTrainingApplication(
        args.training_parameters_path
    )
    application.start()


if __name__ == "__main__":
    main()
