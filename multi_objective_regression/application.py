import argparse
import dataclasses
import json
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


class MultiObjectiveRegressionApplication:
    __training_parameters: TrainingParameters = None
    __training_setups: dict[int, TrainingSetup] = None
    __training_results: dict[int, TrainingResult] = None
    __training_datetime: str = None

    def __init__(self, training_parameters_file_path: str) -> None:
        self.__training_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.__training_parameters: TrainingParameters = self.get_training_parameters(
            training_parameters_file_path
        )

        training_setup_generator = TrainingSetupGenerator(self.__training_parameters)
        self.__training_setups: dict[int, TrainingSetup] = (
            training_setup_generator.generate_training_setups()
        )

    def get_training_parameters(
        self, training_parameters_file_path: str
    ) -> TrainingParameters:
        with open(training_parameters_file_path) as file:
            training_parameters_file_content = file.read()

        training_parameters: TrainingParameters = TrainingParameters.from_json(
            training_parameters_file_content
        )

        return training_parameters

    def start_training(self):
        training_manager = TrainingManager(
            self.__training_parameters, self.__training_setups
        )
        self.__training_results = training_manager.start_training()

    def save_training_results(self):
        os.makedirs(os.path.join(self.__training_datetime, "training_results"))
        os.makedirs(os.path.join(self.__training_datetime, "training_setups"))
        for index, training_result in self.__training_results.items():
            with open(
                os.path.join(
                    self.__training_datetime,
                    "training_results",
                    str(index) + "_result.json",
                ),
                "w",
            ) as file:
                file.write(
                    json.dumps(
                        dataclasses.asdict(training_result), indent=4, default=str
                    )
                )
            with open(
                os.path.join(
                    self.__training_datetime,
                    "training_setups",
                    str(index) + "_training_setup.json",
                ),
                "w",
            ) as file:
                file.write(
                    json.dumps(
                        dataclasses.asdict(self.__training_setups[index]),
                        indent=4,
                        default=str,
                    )
                )

    def render_training_results(self):
        PlotUtility.plot_training_multi_objective_scores(
            self.__training_datetime, self.__training_results
        )

        PlotUtility.plot_top_n_training_multi_objective_scores(
            self.__training_datetime, self.__training_results, 5
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
    application.save_training_results()
    application.render_training_results()


if __name__ == "__main__":
    main()
