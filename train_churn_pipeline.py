"""Train, tune, evaluate, and export a Telco churn ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from churn_pipeline import RANDOM_STATE, TARGET_COLUMN, build_param_grid, build_pipeline


DEFAULT_DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a reusable Telco customer churn pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Telco churn CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the model and reports will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for final testing.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds for GridSearchCV.",
    )
    parser.add_argument(
        "--scoring",
        default="roc_auc",
        help="GridSearchCV scoring metric.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs used by GridSearchCV. Use -1 to use all available CPU cores.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column: {TARGET_COLUMN}")

    return data


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if target.isna().any():
        invalid_values = sorted(data.loc[target.isna(), TARGET_COLUMN].dropna().unique())
        raise ValueError(f"Unexpected target values found: {invalid_values}")

    return features, target.astype(int)


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
            output_dict=True,
        ),
    }


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(to_json_safe(payload), indent=2), encoding="utf-8")


def to_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "get_params") and hasattr(value, "__class__"):
        return value.__class__.__name__

    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data_path)
    X, y = split_features_and_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv_strategy = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=build_param_grid(),
        scoring=args.scoring,
        cv=cv_strategy,
        n_jobs=args.n_jobs,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test)
    metrics.update(
        {
            "best_cv_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "scoring": args.scoring,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
    )

    model_path = args.output_dir / "telco_churn_pipeline.joblib"
    metrics_path = args.output_dir / "metrics.json"
    grid_results_path = args.output_dir / "grid_search_results.csv"

    joblib.dump(best_pipeline, model_path)
    save_json(metrics, metrics_path)
    pd.DataFrame(grid_search.cv_results_).to_csv(grid_results_path, index=False)

    print("Training complete")
    print(f"Best model: {best_pipeline.named_steps['classifier'].__class__.__name__}")
    print(f"Best CV {args.scoring}: {grid_search.best_score_:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved grid results: {grid_results_path}")


if __name__ == "__main__":
    main()
