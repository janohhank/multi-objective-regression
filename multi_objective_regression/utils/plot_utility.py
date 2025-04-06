import os
from abc import ABC

import pandas as pd
import seaborn
from dto.training_result import TrainingResult
from matplotlib import pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)


class PlotUtility(ABC):

    @staticmethod
    def plot_training_multi_objective_scores(
        folder: str, prefix: str, training_results: dict[int, TrainingResult], name: str
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
        ax = seaborn.barplot(x=keys, y=values, order=keys)
        plt.xlabel("Training Setup")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Multi-objective Score")
        plt.ylim((0.0, 1.0))
        plt.title(name + " training multi-objective score results.")

        if len(keys) < 20:
            ax.bar_label(ax.containers[0])

        if len(keys) > 100:
            for i, label in enumerate(ax.get_xticklabels()):
                if i % 20 != 0:
                    label.set_visible(False)

        plt.savefig(
            os.path.join(folder, prefix + "_multi_objective_scores.pdf"),
            format="pdf",
            dpi=300,
        )
        plt.clf()

    @staticmethod
    def plot_correlation_matrix(folder: str, prefix: str, correlation_matrix) -> None:
        plt.figure(figsize=(14, 14))
        seaborn.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title(prefix + " Correlation Matrix")
        plt.savefig(
            os.path.join(folder, prefix.lower() + "_correlation_matrix.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_metric_comparison(
        folder: str, top_n: int, training_results: dict[int, TrainingResult]
    ):
        sorted_data = dict(
            sorted(
                training_results.items(),
                key=lambda item: item[1].validation_results["multi_objective_score"],
                reverse=True,
            )[:top_n]
        )

        metric_names: list[str] = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "coefficient_sign_diff_score",
        ]
        data = {"Index": [], "Metric": [], "Value": []}
        for metric_name in metric_names:
            for index, training_result in sorted_data.items():
                data["Index"].append(f"Training {index}")
                data["Metric"].append(metric_name)
                data["Value"].append(training_result.validation_results[metric_name])

        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 6))
        seaborn.lineplot(
            data=df,
            x="Metric",
            y="Value",
            hue="Index",
            marker="o",
            linestyle="dotted",
            alpha=0.5,
            sort=False,
        )

        plt.title("Performance Metrics for Different Trainings")
        plt.xlabel("Metrics")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Score")
        plt.ylim(0, 1)

        plt.legend(title="Training Index")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(
            os.path.join(folder, "performance_metrics.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_roc_curve(
        training_datetime: str,
        folder: str,
        y_test,
        y_probs,
    ):
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(training_datetime, folder, "roc_curve.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_pr_curve(
        training_datetime: str,
        folder: str,
        y_test,
        y_probs,
    ):
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        ap_score = average_precision_score(y_test, y_probs)

        plt.figure(figsize=(6, 5))
        plt.plot(
            recall, precision, label=f"PR Curve (AP = {ap_score:.4f})", color="blue"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(training_datetime, folder, "pr_curve.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_confusion_matrix(
        training_datetime: str,
        folder: str,
        y_test,
        y_pred,
    ):
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Negative", "Positive"]  # Adjust as needed

        plt.figure(figsize=(5, 4))
        seaborn.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        plt.savefig(
            os.path.join(training_datetime, folder, "confusion_matrix.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()
