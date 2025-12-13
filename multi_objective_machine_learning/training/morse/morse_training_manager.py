import random
from collections import Counter
from copy import deepcopy

from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from training.morse.morse_logistic_regression_training import (
    MorseLogisticRegressionTraining,
)
from training.morse.mutation_crossover_manager import MutationCrossoverManager
from training.morse.training_setup_generator import TrainingSetupGenerator
from training.training_manager import TrainingManager
from utils.constants import ConstantUtility
from utils.plot_utility import PlotUtility
from utils.training_result_utility import (
    TrainingResultUtility,
)


class MorseTrainingManager(TrainingManager):
    # MORSE supervised machine learning trainer
    __logistic_regression_training: MorseLogisticRegressionTraining = None
    # MORSE meta-optimization mutation and crossover helper
    __mutation_crossover_manager: MutationCrossoverManager = None

    def __init__(
        self,
        training_parameters: TrainingParameters,
    ):
        super().__init__(training_parameters)

        self.TYPE = "MORSE"
        self.__logistic_regression_training: MorseLogisticRegressionTraining = (
            MorseLogisticRegressionTraining(self._training_parameters)
        )
        self.__mutation_crossover_manager: MutationCrossoverManager = (
            MutationCrossoverManager(self._training_parameters)
        )

    def start_training(self, result_directory: str) -> dict[int, TrainingResult]:
        print(
            f"Generating {self._training_parameters.morse.initial_training_setup_count} initial training setups."
        )
        training_setups: dict[int, TrainingSetup] = (
            TrainingSetupGenerator.generate_training_setups(self._training_parameters)
        )

        print(f"Start initial population training.")
        training_results: dict[int, TrainingResult] = self.__train_logistic_regression(
            training_setups
        )
        top_initial_training_results: dict[int, TrainingResult] = (
            TrainingResultUtility.get_top_n_training_results_on_validation_set(
                training_results,
                self._training_parameters.morse.initial_training_top_n_selection_count,
            )
        )

        print("Start meta-optimization training.")
        final_top_training_results, all_training_results = (
            self.__start_meta_optimization(
                len(training_setups) + 1, top_initial_training_results, training_results
            )
        )

        print("Saving all training multi objective scores.")
        PlotUtility.plot_training_multi_objective_scores(
            result_directory,
            "all",
            all_training_results,
            ConstantUtility.VALIDATION_DATASET_PREFIX,
        )

        # Plot objective space with two objectives
        # PlotUtility.plot_objective_space(
        #    training_datetime, "accuracy", "precision", all_training_results
        # )

        return final_top_training_results

    def __train_logistic_regression(
        self,
        training_setups: dict[int, TrainingSetup],
    ) -> dict[int, TrainingResult]:
        train_results: dict[int, TrainingResult] = {}
        for index, training_setup in training_setups.items():
            train_results[index] = self.__logistic_regression_training.train(
                index,
                training_setup,
                self._pearson_correlation_to_target_feature,
                self._x_train,
                self._y_train,
                self._x_validation,
                self._y_validation,
                self._x_test,
                self._y_test,
            )
        return train_results

    def __start_meta_optimization(
        self,
        current_training_index: int,
        initial_training_top_n_results: dict[int, TrainingResult],
        all_training_results: dict[int, TrainingResult],
    ) -> tuple[dict[int, TrainingResult], dict[int, TrainingResult]]:
        final_top_n_training_results: dict[int, TrainingResult] = deepcopy(
            initial_training_top_n_results
        )

        training_setup_workspace: dict[int, TrainingSetup] = {}
        new_candidate_training_index: int = current_training_index

        # Main genetic algorithm iteration loop
        for _ in range(
            self._training_parameters.morse.mutation_and_crossover_iteration
        ):
            # Do mutation or crossover
            if (
                random.uniform(0, 1)
                < self._training_parameters.morse.mutation_and_crossover_balance
            ):
                new_training_setup: TrainingSetup = (
                    self.__mutation_crossover_manager.features_mutation(
                        random.choice(list(final_top_n_training_results.values())),
                    )
                )
            else:
                new_training_setup: TrainingSetup = (
                    self.__mutation_crossover_manager.features_crossover(
                        random.choice(list(final_top_n_training_results.values())),
                        random.choice(list(final_top_n_training_results.values())),
                    )
                )

            # Check if the new training setup candidate is valid or not
            if (
                self.__is_valid_training_setup(new_training_setup, all_training_results)
                is True
            ):
                new_candidate_training_index += 1
                new_training_setup.index = new_candidate_training_index
                training_setup_workspace[new_candidate_training_index] = (
                    new_training_setup
                )

            if len(training_setup_workspace) == 0:
                continue

            # Use new training setup to train a model
            training_results: dict[int, TrainingResult] = (
                self.__train_logistic_regression(training_setup_workspace)
            )

            # Merge training results
            for index, training_result in training_results.items():
                all_training_results[index] = training_result
                final_top_n_training_results: dict[int, TrainingResult] = (
                    TrainingResultUtility.merge_training_results(
                        index,
                        training_result,
                        final_top_n_training_results,
                    )
                )
            training_setup_workspace.clear()

            # Check if it could suggest new training setup based on the training result
            training_setup_candidates: list[TrainingSetup] = (
                self.__logistic_regression_training.suggest_training_setup_candidates(
                    training_results
                )
            )
            for training_setup in training_setup_candidates:
                if (
                    self.__is_valid_training_setup(training_setup, all_training_results)
                    is True
                ):
                    new_candidate_training_index += 1
                    training_setup.index = new_candidate_training_index
                    training_setup_workspace[new_candidate_training_index] = (
                        training_setup
                    )

        print(f"Overall inspected training setups: {new_candidate_training_index}.")
        return final_top_n_training_results, all_training_results

    def __is_valid_training_setup(
        self,
        new_training_setup: TrainingSetup,
        all_training_results: dict[int, TrainingResult],
    ) -> bool:
        # Skip if generated an empty feature set.
        if new_training_setup is None or not new_training_setup.features:
            return False

        # Skip if the generated feature set contains an excluded feature combination.
        skip: bool = False
        for exclusion_tuple in self._training_parameters.morse.excluded_feature_sets:
            if set(exclusion_tuple).issubset(set(new_training_setup.features)):
                skip = True

        if skip:
            return False

        # Skip if this combination is already tried.
        found_same: bool = False
        for training_result in all_training_results.values():
            if Counter(training_result.training_setup.features) == Counter(
                new_training_setup.features
            ):
                found_same = True
                break
        if found_same:
            return False

        return True
