import json
import os
from abc import ABC

import joblib
from dto.training_result import TrainingResult


class TrainingResultUtility(ABC):

    @staticmethod
    def get_best_training_result(
        training_results: dict[int, TrainingResult],
    ) -> TrainingResult:
        return sorted(
            training_results.items(),
            key=lambda item: item[1].validation_results["multi_objective_score"],
            reverse=True,
        )[0][1]

    @staticmethod
    def get_top_n_training_results(
        training_results: dict[int, TrainingResult], n: int
    ) -> dict[int, TrainingResult]:
        return dict(
            sorted(
                training_results.items(),
                key=lambda item: item[1].validation_results["multi_objective_score"],
                reverse=True,
            )[:n]
        )

    @staticmethod
    def merge_training_results(
        new_training_index: int,
        new_training_result: TrainingResult,
        training_results: dict[int, TrainingResult],
    ) -> dict[int, TrainingResult]:
        lowest_key: int = min(
            training_results,
            key=lambda k: training_results[k].validation_results[
                "multi_objective_score"
            ],
        )
        lowest_score: float = training_results[lowest_key].validation_results[
            "multi_objective_score"
        ]

        new_score = new_training_result.validation_results["multi_objective_score"]

        if new_score > lowest_score:
            training_results.pop(lowest_key)
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
                        training_result.to_dict(),
                        indent=4,
                        default=str,
                    )
                )

    @staticmethod
    def save_model(
        training_datetime: str, folder: str, training_result: TrainingResult
    ) -> None:
        os.makedirs(
            os.path.join(
                training_datetime,
                folder,
            )
        )

        with open(
            os.path.join(
                training_datetime,
                folder,
                str(training_result.index) + "_result.json",
            ),
            "w",
        ) as file:
            file.write(
                json.dumps(
                    training_result.to_dict(),
                    indent=4,
                    default=str,
                )
            )

        joblib.dump(
            training_result.model, os.path.join(training_datetime, folder, "model.pkl")
        )
        joblib.dump(
            training_result.scaler,
            os.path.join(training_datetime, folder, "scaler.pkl"),
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
            file.write("Index, Validation MOS, Selected Features")
            for index, training_result in sorted(
                training_results.items(),
                key=lambda item: item[1].validation_results["multi_objective_score"],
                reverse=True,
            ):
                file.write(
                    f"{index},{training_result.validation_results["multi_objective_score"]},[{training_result.training_setup.features}]\n"
                )
