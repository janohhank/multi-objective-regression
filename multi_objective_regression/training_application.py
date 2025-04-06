import argparse
import os
import time
from datetime import datetime

from pandas import DataFrame

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from training.training_manager import TrainingManager
from training.training_setup_generator import (
    TrainingSetupGenerator,
)
from utils.plot_utility import PlotUtility
from utils.training_parameter_utility import (
    TrainingParameterUtility,
)
from utils.training_result_utility import (
    TrainingResultUtility,
)
from utils.training_utility import TrainingUtility


class MultiObjectiveTrainingApplication:
    __VALIDATION_DATASET_PREFIX: str = "Validation dataset"
    __INITIAL_TOP_N_TRAINING_RESULT_DIR_NAME: str = "initial_top_n_training_results"
    __FINAL_TRAINING_RESULT_DIR_NAME: str = "final_top_n_training_results"

    __training_parameters: TrainingParameters = None
    __training_manager: TrainingManager = None

    def __init__(self, training_parameters_file_path: str) -> None:
        self.__training_parameters: TrainingParameters = (
            TrainingParameterUtility.read_training_parameters(
                training_parameters_file_path
            )
        )

        self.__training_manager = TrainingManager(self.__training_parameters)

    def start_training(self):
        print("Starting training")
        start: float = time.perf_counter()

        training_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(training_datetime)

        print(
            f"Generating {self.__training_parameters.initial_training_setup_count} initial training setups."
        )
        training_setups: dict[int, TrainingSetup] = (
            TrainingSetupGenerator.generate_training_setups(self.__training_parameters)
        )

        print("Prepare train, validation and test datasets.")
        self.__training_manager.prepare_dataset()
        PlotUtility.plot_correlation_matrix(
            training_datetime,
            "Pearson",
            self.__training_manager.get_pearson_correlation_matrix(),
        )
        PlotUtility.plot_correlation_matrix(
            training_datetime,
            "Point-biserial",
            DataFrame.from_dict(
                self.__training_manager.get_point_biserial_correlation_matrix(),
                orient="index",
                columns=["Outcome"],
            ),
        )

        print(f"Starting initial training on {len(training_setups)} training setups.")
        training_results: dict[int, TrainingResult] = (
            self.__training_manager.start_training(training_setups)
        )

        print(
            f"Save top {self.__training_parameters.initial_training_top_n_selection_count} initial training results."
        )
        top_initial_training_results: dict[int, TrainingResult] = (
            TrainingResultUtility.get_top_n_training_results(
                training_results,
                self.__training_parameters.initial_training_top_n_selection_count,
            )
        )
        TrainingResultUtility.save_training_results(
            training_datetime,
            MultiObjectiveTrainingApplication.__INITIAL_TOP_N_TRAINING_RESULT_DIR_NAME,
            top_initial_training_results,
        )
        PlotUtility.plot_training_multi_objective_scores(
            training_datetime,
            "initial_top_n",
            top_initial_training_results,
            MultiObjectiveTrainingApplication.__VALIDATION_DATASET_PREFIX,
        )

        print("Starting meta-optimization training.")
        final_top_training_results, all_training_results = (
            self.__training_manager.start_meta_optimization(
                len(training_setups) + 1, top_initial_training_results, training_results
            )
        )

        print("Save final training results.")
        TrainingResultUtility.save_training_results(
            training_datetime,
            MultiObjectiveTrainingApplication.__FINAL_TRAINING_RESULT_DIR_NAME,
            final_top_training_results,
        )
        TrainingResultUtility.save_training_results_report(
            training_datetime,
            final_top_training_results,
        )
        PlotUtility.plot_training_multi_objective_scores(
            training_datetime,
            "final_top_n",
            final_top_training_results,
            MultiObjectiveTrainingApplication.__VALIDATION_DATASET_PREFIX,
        )
        PlotUtility.plot_training_multi_objective_scores(
            training_datetime,
            "all",
            all_training_results,
            MultiObjectiveTrainingApplication.__VALIDATION_DATASET_PREFIX,
        )

        test_dataset: tuple[DataFrame, DataFrame] = (
            self.__training_manager.get_test_dataset()
        )
        best_model: TrainingResult = TrainingResultUtility.get_best_training_result(
            final_top_training_results
        )

        TrainingResultUtility.save_model(
            training_datetime,
            "model",
            best_model,
        )

        TrainingUtility.save_dataset(
            training_datetime,
            "test_dataset",
            test_dataset,
        )

        PlotUtility.plot_metric_comparison(
            training_datetime, 4, final_top_training_results
        )

        x_test_reduced = test_dataset[0][best_model.training_setup.features].copy()
        x_test_scaled = best_model.scaler.transform(x_test_reduced)
        y_probs = best_model.model.predict_proba(x_test_scaled)[:, 1]
        y_pred = best_model.model.predict(x_test_scaled)
        PlotUtility.plot_roc_curve(training_datetime, "model", test_dataset[1], y_probs)
        PlotUtility.plot_pr_curve(training_datetime, "model", test_dataset[1], y_probs)
        PlotUtility.plot_confusion_matrix(
            training_datetime, "model", test_dataset[1], y_pred
        )

        elapsed: float = time.perf_counter() - start
        print(f"Finished training on {elapsed:.2f} seconds.")


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
    application.start_training()


if __name__ == "__main__":
    main()
