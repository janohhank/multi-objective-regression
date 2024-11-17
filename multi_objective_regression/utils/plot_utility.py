import os

import seaborn
from dto.training_result import TrainingResult
from matplotlib import pyplot as plt


class PlotUtility:

    @staticmethod
    def plot_training_multi_objective_scores(
        folder: str, training_results: dict[int, TrainingResult]
    ) -> None:
        sorted_data = dict(
            sorted(
                training_results.items(), key=lambda item: item[1].multi_objective_score
            )
        )

        keys = list(sorted_data.keys())
        values = [item.multi_objective_score for item in sorted_data.values()]

        seaborn.barplot(x=keys, y=values, order=keys)
        plt.xlabel("Training Index")
        plt.ylabel("Multi-objective Score")
        plt.title("Training multi-objective score results.")
        plt.savefig(
            os.path.join(folder, "multi_objective_scores.png"), format="png", dpi=300
        )
        plt.clf()

    @staticmethod
    def plot_top_n_training_multi_objective_scores(
        folder: str, training_results: dict[int, TrainingResult], n: int
    ) -> None:
        sorted_data = dict(
            sorted(
                training_results.items(), key=lambda item: item[1].multi_objective_score
            )
        )

        keys = list(sorted_data.keys())[-n:]
        values = [
            item.multi_objective_score for item in list(sorted_data.values())[-n:]
        ]

        seaborn.barplot(x=keys, y=values, order=keys)
        plt.xlabel("Training Index")
        plt.ylabel("Multi-objective Score")
        plt.title("Training multi-objective score results.")
        plt.savefig(
            os.path.join(folder, "best_" + str(n) + "_multi_objective_scores.png"),
            format="png",
            dpi=300,
        )
        plt.clf()
