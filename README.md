# multi_objective_machine_learning

Binary classification-focused, multi-objective supervised machine learning with genetic feature selection.  
The project provides:
- A custom MORSE meta-optimization algorithm for feature selection and model search
- An NSGA-II implementation via DEAP
- Multiple objective components (ROC-AUC, PR-AUC, F1, etc.) combined through user-defined weights
- Reproducible, CLI-driven training and prediction pipelines

Core model: scikit-learn LogisticRegression with z-score standardization.  
Scope: Binary classification.

## Highlights
- Multi-objective optimization over thresholded and score-based metrics:
  - Thresholded: accuracy, precision, recall (sensitivity), specificity, F1
  - Score-based: ROC-AUC, PR-AUC (average precision), Gini
  - Structural: coefficient_sign_diff (encourages consistency between learned coefficient signs and correlation signs)
- Two meta-optimizers:
  - MORSE (custom): mutation/crossover over feature subsets, excludes specified feature combinations
  - DEAP/NSGA-II: Pareto-based evolutionary optimization with tunable population size and iterations
- Automatic dataset split or use pre-defined splits
- Saved artifacts: best model + scaler, JSON results, and PDF plots

## Project structure
- multi_objective_machine_learning/
  - training_application.py — CLI training entrypoint
  - predictor_application.py — CLI prediction/evaluation on a saved model
  - dto/ — dataclasses (parameters, setups, results)
  - training/
    - objective_components.py — objective metrics and factory
    - training_manager.py — dataset prep and flow
    - deap/deap_training_manager.py — NSGA-II via DEAP
    - morse/morse_training_manager.py — MORSE optimizer and training loop
  - utils/
    - plot_utility.py — ROC/PR/confusion and other plots
    - training_result_utility.py — saving/merging results
    - training_parameter_utility.py — parameter loading
- test_data/
  - diabetes.csv, diabetes_modified.csv
  - test_train_configuration_diabetes.json — example training configuration

## Installation
Requirements:
- Python 3.10+ (uses modern typing like `int | float`)
- Recommended: virtual environment

Install packages:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install numpy pandas scikit-learn imbalanced-learn deap joblib matplotlib seaborn orjson
```

Optional (for notebooks): `pip install jupyter`

## Configuration (JSON)
Training is driven by a JSON file describing data sources, algorithms, objectives, and hyperparameters.

Fields:
- train_dataset: CSV path with features + binary target
- validation_dataset (optional): CSV path; if omitted, split is created from train_dataset
- test_dataset (optional): CSV path; if omitted, split is created from train_dataset
- algorithms: ["MORSE", "DEAP"] — choose one or both
- target_feature: column name of the binary target (0/1)
- features: list of feature column names
- morse: MORSE-specific parameters
  - excluded_feature_sets: list of lists; any subset here is disallowed together (e.g., raw + log of same variable)
  - initial_training_setup_generator_type: e.g. "RANDOM_COMBINATIONS"
  - initial_training_setup_count: int
  - initial_training_top_n_selection_count: int
  - mutation_and_crossover_iteration: int
  - mutation_and_crossover_balance: float in [0,1]
  - mutation_feature_change_probability: float in [0,1]
  - crossover_feature_selection_probability: float in [0,1]
- deap: NSGA-II params
  - initial_population_size: int
  - iteration: number of generations
- multi_objective_functions: mapping of metric name -> weight (float)
  - Available names: accuracy, precision, recall, specificity, f1_score, roc_auc, pr_auc, gini, coefficient_sign_diff
  - Set weight to 0.0 to ignore a metric

Example (provided):
```json
{
  "train_dataset": "../test_data/diabetes_modified.csv",
  "target_feature": "Outcome",
  "features": [
    "Pregnancies",
    "ln_pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "ln_insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "ln_diabetes_pedigree_function",
    "Age",
    "ln_age"
  ],
  "algorithms": ["MORSE","DEAP"],
  "morse": {
    "excluded_feature_sets": [
      ["ln_pregnancies","Pregnancies"],
      ["ln_insulin","Insulin"],
      ["ln_diabetes_pedigree_function","DiabetesPedigreeFunction"],
      ["ln_age","Age"]
    ],
    "initial_training_setup_generator_type": "RANDOM_COMBINATIONS",
    "initial_training_setup_count": 48,
    "initial_training_top_n_selection_count": 8,
    "mutation_and_crossover_iteration": 8192,
    "mutation_and_crossover_balance": 0.5,
    "mutation_feature_change_probability": 0.3,
    "crossover_feature_selection_probability": 0.5
  },
  "deap": {
    "initial_population_size": 48,
    "iteration": 5
  },
  "multi_objective_functions": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "specificity": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.3,
    "pr_auc": 0.5,
    "gini": 0.0,
    "coefficient_sign_diff": 0.2
  }
}
```

Notes:
- `coefficient_sign_diff` scores 1.0 when coefficient sign agrees with correlation sign across features (penalizes disagreement/near-zero agreement).
- Thresholded metrics (accuracy/precision/recall/specificity/F1) default to 0.5 internally when evaluated at train/validation/test time. ROC/PR/Gini use continuous scores.

## Training
Run from repository root.

Option A (module):
```bash
python -m multi_objective_machine_learning.training_application \
  --training_parameters_path test_data/test_train_configuration_diabetes.json
```

Option B (script path):
```bash
python multi_objective_machine_learning/training_application.py \
  --training_parameters_path test_data/test_train_configuration_diabetes.json
```

What happens:
1. Load dataset(s) and either use provided validation/test CSVs or split the train CSV into train/validation/test (stratified, random_state=42).
2. Compute Pearson correlation matrix and per-feature correlation to target.
3. For each selected algorithm:
   - Optimize feature subsets w.r.t your `multi_objective_functions` weights
   - Train LogisticRegression on selected features (z-score standardized)
   - Evaluate on validation and test
4. Pick the best model by validation multi-objective score and save artifacts
5. Generate plots

Outputs (in a timestamped directory, e.g. `2025-12-13-17-22-01/`):
- overall_results.json — aggregated results for all algorithms
- For each algorithm (MORSE/DEAP):
  - model/
    - model.pkl — trained LogisticRegression
    - scaler.pkl — fitted StandardScaler
    - 0_result.json — details (selected features, metrics, multi-objective score)
  - performance_metrics.pdf — top-N validation metrics comparison
  - <prefix>_multi_objective_scores.pdf — bars of validation multi-objective scores
  - roc_curve.pdf, pr_curve.pdf, confusion_matrix.pdf
  - spec_sens_curve.pdf — sensitivity/specificity curve with balanced point

## Prediction (evaluate a saved model)
Provide the saved model directory created by training (the folder containing `model.pkl` and `scaler.pkl`) and a CSV test dataset.

Option A (module):
```bash
python -m multi_objective_machine_learning.predictor_application \
  --model_path 2025-12-13-17-22-01/MORSE/model \
  --test_dataset_path test_data/diabetes_modified.csv \
  --threshold 0.5
```

Option B (script path):
```bash
python multi_objective_machine_learning/predictor_application.py \
  --model_path 2025-12-13-17-22-01/MORSE/model \
  --test_dataset_path test_data/diabetes_modified.csv \
  --threshold 0.5
```

Printed metrics:
- Accuracy, Precision, Recall, Specificity, F1 (at given `--threshold`)
- ROC-AUC, PR-AUC, Gini (threshold-free)

## How objectives are combined
- You define weights via `multi_objective_functions`. Only metrics with weight > 0 influence optimization in DEAP and selection of the best model.
- An internal multi-objective score is computed on validation to rank and pick the best model to save.

## Reproducibility and defaults
- LogisticRegression: solver=lbfgs, penalty=l2, C=1.0, max_iter=1024, random_state=42, n_jobs=4
- Splits: train/validation/test use `random_state=42` and are stratified
- Standardization: z-score fitted on train, applied to validation/test

## Limitations
- Binary classification only (target values must be 0/1)
- Current learner is LogisticRegression
- If you use `coefficient_sign_diff`, ensure features are numeric and correlations are meaningful

## License
Not specified.

## Citation
If you use MORSE or this repository in academic work, please consider citing the repository and describing MORSE and your objective configuration.

## Acknowledgements
- DEAP: NSGA-II implementation (https://github.com/DEAP/deap)
- scikit-learn, imbalanced-learn, pandas, numpy, seaborn, matplotlib
