import time
from copy import deepcopy

import numpy as np
from deap import base, creator, tools, algorithms
from dto.deap_training_results import DeapTrainingResults
from dto.training_parameters import TrainingParameters
from dto.training_result import TrainingResult
from dto.training_setup import TrainingSetup
from sklearn.linear_model import LogisticRegression
from training.model_evaluation_utility import (
    ModelEvaluationUtility,
)
from training.objective_components import ObjectiveComponent
from training.objective_components import create_objective_components
from training.training_manager import TrainingManager
from utils.training_utility import TrainingUtility


class DeapTrainingManager(TrainingManager):
    __objective_components: list[ObjectiveComponent] = None

    def __init__(
        self,
        training_parameters: TrainingParameters,
    ):
        super().__init__(training_parameters)

        self.TYPE = "DEAP"

        self.__objective_components: list[ObjectiveComponent] = (
            create_objective_components(
                self._training_parameters.multi_objective_functions
            )
        )

    def __evaluate(self, individual):
        if sum(individual) == 0:
            return 0.0, 0.0, 0.0

        selected_cols = [f for f, bit in zip(self._x_train.columns, individual) if bit]
        log_regression = LogisticRegression(
            n_jobs=4,
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1024,
            random_state=42,
        )

        # Reduced train datasets.
        x_train_reduced = self._x_train[selected_cols].copy()
        x_validation_reduced = self._x_validation[selected_cols].copy()

        # Create z-score standardization scaler from train dataset.
        scaler = TrainingUtility.fit_standard_scaler(x_train_reduced)

        # Standardize train dataset.
        scaled_x_train = scaler.transform(x_train_reduced)

        # Standardize validation and test dataset.
        scaled_x_validation = scaler.transform(x_validation_reduced)

        log_regression.fit(scaled_x_train, self._y_train)
        y_probs: np.ndarray = log_regression.predict_proba(scaled_x_validation)[:, 1]

        fitness_values: list[float] = []
        for objective_component in self.__objective_components:
            if objective_component.weight == 0.0:
                continue

            fitness_values.append(
                objective_component.score(self._y_validation, y_probs)
            )
        return tuple(fitness_values)

    def start_training(self, result_directory: str) -> dict[int, TrainingResult]:
        feature_names: list = list(self._x_train.columns)
        n_features: int = len(feature_names)

        fitness_weights: list[float] = []
        for objective_component in self.__objective_components:
            if objective_component.weight == 0.0:
                continue
            fitness_weights.append(objective_component.weight)

        creator.create(
            "FitnessMulti",
            base.Fitness,
            weights=tuple(fitness_weights),
        )
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 2)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            n=n_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.__evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        pop = toolbox.population(
            n=self._training_parameters.deap.initial_population_size
        )
        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        _, _ = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=self._training_parameters.deap.initial_population_size,
            lambda_=self._training_parameters.deap.initial_population_size * 2,
            cxpb=0.5,
            mutpb=0.3,
            ngen=self._training_parameters.deap.iteration,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        print("Pareto optimal sets:")
        for ind in hof:
            selected = [f for f, use in zip(feature_names, ind) if use]
            print(f"{ind.fitness.values} | Features: {selected}")

        print("Get the best model.")
        best_individual = max(
            hof,
            key=lambda ind: sum(
                w * v for w, v in zip(fitness_weights, ind.fitness.values)
            ),
        )
        selected_features = [f for f, use in zip(feature_names, best_individual) if use]

        start: float = time.perf_counter()

        # Reduced train datasets.
        x_train_reduced = self._x_train[selected_features].copy()
        x_validation_reduced = self._x_validation[selected_features].copy()
        x_test_reduced = self._x_test[selected_features].copy()

        # Create z-score standardization scaler from train dataset.
        scaler = TrainingUtility.fit_standard_scaler(x_train_reduced)

        # Standardize train dataset.
        scaled_x_train = scaler.transform(x_train_reduced)

        # Standardize validation and test dataset.
        scaled_x_validation = scaler.transform(x_validation_reduced)
        scaled_x_test = scaler.transform(x_test_reduced)

        log_regression = LogisticRegression(
            n_jobs=4,
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1024,
            random_state=42,
        )
        log_regression.fit(scaled_x_train, self._y_train)

        coefficients: dict[str, float] = dict(
            zip(selected_features, log_regression.coef_[0])
        )

        elapsed_sec: float = time.perf_counter() - start

        training_result = DeapTrainingResults(
            index=0,
            training_setup=TrainingSetup(
                index=0,
                features=selected_features,
                target_feature=self._training_parameters.target_feature,
            ),
            validation_results=ModelEvaluationUtility.evaluate_log_regression(
                self.__objective_components,
                selected_features,
                self._pearson_correlation_to_target_feature,
                log_regression,
                scaled_x_validation,
                self._y_validation,
            ),
            test_results=ModelEvaluationUtility.evaluate_log_regression(
                self.__objective_components,
                selected_features,
                self._pearson_correlation_to_target_feature,
                log_regression,
                scaled_x_test,
                self._y_test,
            ),
            training_time_seconds=elapsed_sec,
            model=deepcopy(log_regression),
            scaler=deepcopy(scaler),
        )

        return {0: training_result}
