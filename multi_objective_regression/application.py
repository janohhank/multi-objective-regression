import argparse
import os
from datetime import datetime

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


class MultiObjectiveRegressionApplication:
    __POPULATION_TRAINING_RESULT_DIR_NAME: str = "population_training_results"
    __POPULATION_TOP_N_TRAINING_RESULT_DIR_NAME: str = (
        "population_top_n_training_results"
    )
    __FINAL_TRAINING_RESULT_DIR_NAME: str = "final_training_results"

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
        training_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(training_datetime)

        print("Generate population.")
        training_setups: dict[int, TrainingSetup] = (
            TrainingSetupGenerator.generate_training_setups(self.__training_parameters)
        )

        print("Prepare train dataset.")
        self.__training_manager.prepare_dataset()

        print("Start training on population.")
        training_results: dict[int, TrainingResult] = (
            self.__training_manager.start_training(training_setups)
        )
        PlotUtility.plot_training_multi_objective_scores(
            training_datetime, "full", training_results
        )

        print(
            f"Save top {self.__training_parameters.select_top_n_training} training results."
        )
        top_training_results: dict[int, TrainingResult] = (
            TrainingResultUtility.get_top_n_training_results(
                training_results, self.__training_parameters.select_top_n_training
            )
        )
        TrainingResultUtility.save_training_results(
            training_datetime,
            MultiObjectiveRegressionApplication.__POPULATION_TRAINING_RESULT_DIR_NAME,
            training_results,
        )
        TrainingResultUtility.save_training_results(
            training_datetime,
            MultiObjectiveRegressionApplication.__POPULATION_TOP_N_TRAINING_RESULT_DIR_NAME,
            top_training_results,
        )
        PlotUtility.plot_training_multi_objective_scores(
            training_datetime, "top_n", top_training_results
        )

        print("Start mutation and crossover.")
        final_training_results: dict[int, TrainingResult] = (
            self.__training_manager.start_mutation_and_crossover(
                len(training_setups) + 1, top_training_results
            )
        )

        print("Save final training results.")
        TrainingResultUtility.save_training_results(
            training_datetime,
            MultiObjectiveRegressionApplication.__FINAL_TRAINING_RESULT_DIR_NAME,
            final_training_results,
        )
        PlotUtility.plot_training_multi_objective_scores(
            training_datetime, "final_top_n", final_training_results
        )


def main():
    parser = argparse.ArgumentParser(description="Multi objective regression training.")
    parser.add_argument(
        "--training_parameters_path",
        type=str,
        required=True,
        help="Training parameters descriptor file (JSON file).",
    )

    args = parser.parse_args()

    application: MultiObjectiveRegressionApplication = (
        MultiObjectiveRegressionApplication(args.training_parameters_path)
    )
    application.start_training()


if __name__ == "__main__":
    main()
