import os
from abc import ABC

import seaborn
from dto.training_result import TrainingResult
from matplotlib import pyplot as plt


class PlotUtility(ABC):

    @staticmethod
    def plot_training_multi_objective_scores(
        folder: str, prefix: str, training_results: dict[int, TrainingResult]
    ) -> None:
        sorted_data = dict(
            sorted(
                training_results.items(),
                key=lambda item: item[1].validation_results["multi_objective_score"],
            )
        )

        keys = list(sorted_data.keys())
        values = [
            item.validation_results["multi_objective_score"]
            for item in sorted_data.values()
        ]

        plt.figure(figsize=(10, 6))
        seaborn.barplot(x=keys, y=values, order=keys)
        plt.xlabel("Training Index")
        plt.ylabel("Multi-objective Score")
        plt.title("Training multi-objective score results.")
        plt.savefig(
            os.path.join(folder, prefix + "_multi_objective_scores.png"),
            format="png",
            dpi=300,
        )
        plt.clf()

    @staticmethod
    def plot_correlation_matrix(folder: str, correlation_matrix) -> None:
        plt.figure(figsize=(14, 14))
        seaborn.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.savefig(
            os.path.join(folder, "correlation_matrix.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()
