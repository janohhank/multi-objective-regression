import os
import random
from abc import ABC

import numpy as np
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
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_correlation_matrix(
        folder: str, prefix: str, correlation_matrix, fig_size: tuple
    ) -> None:
        plt.figure(figsize=fig_size)
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

        balanced_idx = np.argmin(np.abs(tpr + fpr - 1))
        balanced_fpr = fpr[balanced_idx]
        balanced_tpr = tpr[balanced_idx]

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        plt.scatter(
            balanced_fpr,
            balanced_tpr,
            color="green",
            marker="s",
            label=f"Balanced FPR/TPR = {balanced_tpr:.4f}",
            zorder=5,
        )
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

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]

        balance_idx = np.argmin(np.abs(precision - recall))
        balanced_precision = precision[balance_idx]
        balanced_recall = recall[balance_idx]

        plt.figure(figsize=(6, 5))
        plt.plot(
            recall, precision, label=f"PR Curve (AP = {ap_score:.4f})", color="blue"
        )
        plt.scatter(
            optimal_recall,
            optimal_precision,
            color="red",
            label=f"Max F1 = {optimal_f1:.4f}",
            zorder=5,
        )
        plt.scatter(
            balanced_recall,
            balanced_precision,
            color="green",
            marker="s",
            label=f"Balanced P/R = {balanced_precision:.4f}",
            zorder=5,
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
    def plot_specificity_sensitivity_curve(
        training_datetime: str,
        folder: str,
        y_test,
        y_probs,
    ):
        sorted_indices = np.argsort(y_probs)
        sorted_scores = y_probs[sorted_indices]
        sorted_y_true = y_test.to_numpy()[sorted_indices]

        n = len(y_test)
        sensitivity = np.zeros(n)
        specificity = np.zeros(n)

        for i in range(n):
            predicted_positive = np.zeros(n)
            predicted_positive[i:] = 1

            tp = np.sum((predicted_positive == 1) & (sorted_y_true == 1))
            fn = np.sum((predicted_positive == 0) & (sorted_y_true == 1))
            tn = np.sum((predicted_positive == 0) & (sorted_y_true == 0))
            fp = np.sum((predicted_positive == 1) & (sorted_y_true == 0))

            sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

        diff = np.abs(sensitivity - specificity)
        intersection_idx = np.argmin(diff)
        crossover_value = sorted_scores[intersection_idx]
        crossover_sens = sensitivity[intersection_idx]

        plt.figure(figsize=(6, 5))
        plt.plot(
            range(1, n + 1),
            specificity,
            label="Specificity",
            color="green",
            linestyle="-",
        )
        plt.plot(
            range(1, n + 1),
            sensitivity,
            label="Sensitivity",
            color="blue",
            linestyle="--",
        )
        plt.scatter(
            intersection_idx,
            crossover_sens,
            color="green",
            marker="s",
            label=f"Balanced Sens/Spec = {crossover_sens:.4f}\nThreshold = {crossover_value:.4f}",
            zorder=5,
        )
        plt.xlabel("Test samples in prediction order")
        plt.ylabel("Sensitivity / Specificity")
        plt.ylim(0, 1)
        plt.xlim(1, n)
        plt.xticks([1, n], ["Lowest prediction", "Highest prediction"])
        plt.title("Specificity and sensitivity curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(training_datetime, folder, "spec_sens_curve.pdf"),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()

    @staticmethod
    def plot_objective_space(
        training_datetime: str,
        objective_component_x: str,
        objective_component_y: str,
        all_training_results: dict[int, TrainingResult],
    ):
        all_training_results: dict[int, TrainingResult] = dict(
            sorted(
                all_training_results.items(),
                key=lambda item: item[1].validation_results["multi_objective_score"],
                reverse=True,
            )
        )

        m: float = -0.5 / 0.5
        training_result = random.choice(list(all_training_results.values()))
        x_min = 0.5 * training_result.test_results[objective_component_x] - 1.0 / 2
        x_max = 0.5 * training_result.test_results[objective_component_x] + 1.0 / 2
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = (
            m * (x_vals - 0.5 * training_result.test_results[objective_component_x])
            + 0.5 * training_result.test_results[objective_component_y]
        )

        plt.figure(figsize=(6, 5))
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        for training_result in all_training_results.values():
            plt.scatter(
                0.5 * training_result.test_results[objective_component_x],
                0.5 * training_result.test_results[objective_component_y],
                color="blue",
                zorder=5,
                alpha=0.5,
            )
        plt.plot(x_vals, y_vals, linestyle="--", label=r"$-w_1/w_2$", color="grey")
        plt.scatter(
            0.5
            * list(all_training_results.values())[0].test_results[
                objective_component_x
            ],
            0.5
            * list(all_training_results.values())[0].test_results[
                objective_component_y
            ],
            color="green",
            zorder=5,
            alpha=0.5,
        )
        plt.xlabel(r"$w_1 * \mathcal{L}_{" + objective_component_x + "}(\mathbf{h}) $")
        plt.ylabel(r"$w_2 * \mathcal{L}_{" + objective_component_y + "}(\mathbf{h})$")
        plt.ylim(0.3, 0.5)
        plt.xlim(0.3, 0.5)
        plt.title("Objective Space")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            os.path.join(training_datetime, "objective_space.pdf"),
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
