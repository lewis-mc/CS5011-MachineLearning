import sys
import time
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer,TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Global constants for ordinal ordering.
ORDERING_DICT = {
    "water_quality": [
        "soft",
        "milky",
        "coloured",
        "fluoride",
        "salty",
        "fluoride abandoned",
        "salty abandoned",
        "unknown",
    ],
    "quality_group": ["good", "milky", "colored", "fluoride", "salty", "unknown"],
    "quantity": ["enough", "seasonal", "insufficient", "dry", "unknown"],
    "quantity_group": ["enough", "seasonal", "insufficient", "dry", "unknown"],
}
ORDINAL_COLUMNS = list(ORDERING_DICT.keys())


def process_datetime_features(df, date_format="%Y-%m-%d"):
    """
    For each non-ID column of object type, try converting to datetime.
    If more than 50% of entries can be parsed, extract year, month, and day
    as new columns and drop the original column.
    """
    df = df.copy()
    for col in df.columns:
        if col.lower() == "id":
            continue

        if df[col].dtype == "object":
            dt_series = pd.to_datetime(df[col], errors="coerce", format=date_format)
            if dt_series.notna().mean() > 0.5:
                df[f"{col}_year"] = dt_series.dt.year
                df[f"{col}_month"] = dt_series.dt.month
                df[f"{col}_day"] = dt_series.dt.day
                df.drop(columns=[col], inplace=True)
    return df


def objective(trial, X, y, num_features, cat_features):
    """
    Optuna objective function for hyper-parameter optimization.
    """

    # --- Numerical Preprocessing ---
    num_preproc_choice = trial.suggest_categorical(
        "num_preprocessing", ["StandardScaler", "None"]
    )
    num_transformer = StandardScaler() if num_preproc_choice == "StandardScaler" else "passthrough"

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", num_transformer),
        ]
    )

    # --- Model Selection and Hyper-Parameters ---
    model_choice = trial.suggest_categorical(
        "model_type",
        [
            "LogisticRegression",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "MLPClassifier",
        ],
    )

    sparse_param = True
    if model_choice == "LogisticRegression":
        # Three hyperparameters: C, penalty, and tolerance.
        C = trial.suggest_float("lr_C", 1e-3, 1e3, log=True)
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2"])
        tol = trial.suggest_float("lr_tol", 1e-5, 1e-2, log=True)
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        classifier = LogisticRegression(C=C, penalty=penalty, tol=tol, solver=solver, max_iter=1000)

    elif model_choice == "RandomForestClassifier":
        # Three hyperparameters: n_estimators, max_depth, and min_samples_split.
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        max_depth = trial.suggest_int("rf_max_depth", 2, 20)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        sparse_param = False

    elif model_choice == "GradientBoostingClassifier":
        # Three hyperparameters: n_estimators, learning_rate, and max_depth.
        n_estimators = trial.suggest_int("gb_n_estimators", 50, 300)
        learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 1.0, log=True)
        max_depth = trial.suggest_int("gb_max_depth", 2, 10)
        classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        sparse_param = False


    elif model_choice == "HistGradientBoostingClassifier":
        # Three hyperparameters: max_iter, learning_rate, and max_depth.
        max_iter = trial.suggest_int("hgb_max_iter", 50, 300)
        learning_rate = trial.suggest_float("hgb_learning_rate", 0.01, 1.0, log=True)
        max_depth = trial.suggest_int("hgb_max_depth", 2, 20)
        classifier = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        sparse_param = False

    else:  # MLPClassifier
        # Three hyperparameters: hidden_layer_sizes, alpha, and solver.
        hidden_layer_sizes = trial.suggest_categorical("mlp_hidden_layer_sizes", [(50,), (100,), (50, 50)])
        alpha = trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True)
        solver = trial.suggest_categorical("mlp_solver", ["adam", "sgd"])
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            solver=solver,
            max_iter=1000,
            random_state=42,
        )
        sparse_param = False

    # --- Categorical Preprocessing ---
    cat_preproc_choice = trial.suggest_categorical(
        "cat_preprocessing", ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]
    )
    if cat_preproc_choice == "OneHotEncoder":
        cat_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=sparse_param, min_frequency=0.05
        )

    elif cat_preproc_choice == "OrdinalEncoder":
        available_ordinal = [col for col in cat_features if col in ORDINAL_COLUMNS]
        if not available_ordinal:
            raise optuna.exceptions.TrialPruned(
                "No ordinal features available for OrdinalEncoder."
            )
        categories_list = [ORDERING_DICT[col] for col in available_ordinal]
        cat_transformer = OrdinalEncoder(categories=categories_list)
        # Use only the available ordinal features.
        cat_features = available_ordinal

    else:  # TargetEncoder
        smooth_val = trial.suggest_float("targetencoder_smooth", 0.01, 1.0)
        cat_transformer = TargetEncoder(smooth=smooth_val)

    to_str = FunctionTransformer(lambda X: X.astype(str))
    cat_pipeline = Pipeline(
        steps=[
            ("convert", to_str),
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", cat_transformer),
        ]
    )

    # --- Combine Preprocessing Steps ---
    transformers = []
    if len(num_features) > 0:
        transformers.append(("num", num_pipeline, num_features))
    if len(cat_features) > 0:
        transformers.append(("cat", cat_pipeline, cat_features))
    if not transformers:
        raise ValueError("No features available for transformation!")
    preprocessor = ColumnTransformer(transformers=transformers, verbose=True)

    # --- Build and Evaluate Pipeline ---
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()


def main():
    """
    Main function for loading data, performing hyperparameter optimization (if requested),
    training the pipeline, and saving predictions.
    """
    # --- Parse Command-Line Arguments ---
    if len(sys.argv) < 5:
        print(
            "Usage: python train-and-predict.py <train-input-file> <train-labels-file> "
            "<test-input-file> <test-prediction-output-file> [--hpo]"
        )
        sys.exit(1)
    else:
        _, arg_train_input, arg_train_label, arg_test_input, arg_test_pred_output, *extra = sys.argv
        run_hpo = "--hpo" in extra

    # --- Data Loading ---
    data_dir = "../data"
    train_input_values = pd.read_csv(f"{data_dir}/{arg_train_input}")
    train_input_labels = pd.read_csv(f"{data_dir}/{arg_train_label}")
    test_values = pd.read_csv(f"{data_dir}/{arg_test_input}")

    # --- Process Datetime Features ---
    train_input_values = process_datetime_features(train_input_values)

    # --- Merge Training Data ---
    if "id" in train_input_values.columns and "id" in train_input_labels.columns:
        merged_train = pd.merge(train_input_values, train_input_labels, on="id", how="inner")
    else:
        merged_train = train_input_values.copy()
        if "status_group" in train_input_labels.columns:
            merged_train["status_group"] = train_input_labels["status_group"]
        else:
            merged_train["status_group"] = train_input_labels.iloc[:, 0]

    if "status_group" not in merged_train.columns:
        print("Error: 'status_group' column not found in merged training data.")
        sys.exit(1)

    # --- Split Features and Target ---
    y = merged_train["status_group"]
    X = merged_train.drop(columns=["id", "status_group"], errors="ignore")

    # --- Identify Numeric and Categorical Features ---
    numeric_features = selector(dtype_include=["int64", "float64"])(X)
    categorical_features = selector(dtype_include=["object", "bool"])(X)

    # --- Hyper-Parameter Optimization (HPO) ---
    
    print("Running hyper-parameter optimisation using Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, numeric_features, categorical_features),
        n_trials=150,
    )
    best_trial = study.best_trial

    print("HPO complete. Best trial:")
    print("  Accuracy:", best_trial.value)
    print("  Params:")
   
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
            
    # Save CV results for further analysis.
    trials_df = study.trials_dataframe()
    trials_df.to_csv("../evaluation/hpo_cv_results.csv", index=False)
    print("Cross validation results saved to ../evaluation/hpo_cv_results.csv")

if __name__ == "__main__":
    main()
