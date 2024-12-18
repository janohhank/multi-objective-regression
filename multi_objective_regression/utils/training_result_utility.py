import dataclasses
import json
import os
from abc import ABC

from dto.training_result import TrainingResult


class TrainingResultUtility(ABC):

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
    def merge_training_results(
        new_training_index: int,
        new_training_result: TrainingResult,
        training_results: dict[int, TrainingResult],
    ) -> dict[int, TrainingResult]:
        training_results = dict(
            sorted(
                training_results.items(),
                key=lambda item: item[1].multi_objective_score,
                reverse=True,
            )
        )

        if (
            training_results[list(training_results.keys())[-1]].multi_objective_score
            < new_training_result.multi_objective_score
        ):
            training_results.pop(list(training_results.keys())[-1])
            training_results[new_training_index] = new_training_result

        return training_results

    @staticmethod
    def save_training_results(
        training_datetime: str, folder: str, training_results: dict[int, TrainingResult]
    ) -> None:
        os.makedirs(
            os.path.join(
                training_datetime,
                folder,
            )
        )

        for index, training_result in training_results.items():
            with open(
                os.path.join(
                    training_datetime,
                    folder,
                    str(index) + "_result.json",
                ),
                "w",
            ) as file:
                file.write(
                    json.dumps(
                        dataclasses.asdict(training_result), indent=4, default=str
                    )
                )

    @staticmethod
    def save_training_results_report(
        training_datetime: str, training_results: dict[int, TrainingResult]
    ) -> None:
        with open(
            os.path.join(
                training_datetime,
                "top_results_report.txt",
            ),
            "w",
        ) as file:
            for index, training_result in training_results.items():
                file.write(
                    f"{index},{training_result.multi_objective_score},[{training_result.training_setup.features}]\n"
                )
