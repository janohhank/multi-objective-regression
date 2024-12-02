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
        new_training_results: dict[int, TrainingResult],
        merge_to: dict[int, TrainingResult],
    ) -> dict[int, TrainingResult]:
        for index, training_result in new_training_results.items():
            if (
                merge_to[list(merge_to.keys())[-1]].multi_objective_score
                < training_result.multi_objective_score
            ):
                merge_to.popitem()
                merge_to[index] = training_result
            merge_to = dict(
                sorted(
                    merge_to.items(),
                    key=lambda item: item[1].multi_objective_score,
                    reverse=True,
                )
            )
        return merge_to

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
