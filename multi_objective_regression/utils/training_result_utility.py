import dataclasses
import json
import os
from abc import ABC

from dto.training_result import TrainingResult


class TrainingResultUtility(ABC):
    __POPULATION_TRAINING_RESULT_DIR_NAME: str = "population_training_results"

    @staticmethod
    def get_top_n_training_results(
        training_results: dict[int, TrainingResult], n: int
    ) -> dict[int, TrainingResult]:
        return dict(
            sorted(
                training_results.items(),
                key=lambda item: item[1].multi_objective_score,
                reverse=True,
            )[:n]
        )

    @staticmethod
    def save_training_results(
        training_datetime: str, training_results: dict[int, TrainingResult]
    ) -> None:
        os.makedirs(
            os.path.join(
                training_datetime,
                TrainingResultUtility.__POPULATION_TRAINING_RESULT_DIR_NAME,
            )
        )

        for index, training_result in training_results.items():
            with open(
                os.path.join(
                    training_datetime,
                    TrainingResultUtility.__POPULATION_TRAINING_RESULT_DIR_NAME,
                    str(index) + "_result.json",
                ),
                "w",
            ) as file:
                file.write(
                    json.dumps(
                        dataclasses.asdict(training_result), indent=4, default=str
                    )
                )
