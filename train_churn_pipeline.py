"""Train, tune, evaluate, and export a Telco churn ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from churn_pipeline import (
    DEFAULT_CHURN_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    RANDOM_STATE,
    RISK_LEVELS,
    TARGET_COLUMN,
    assign_risk_levels,
    build_param_grid,
    build_pipeline,
    predict_churn_labels,
)


DEFAULT_DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
BASELINE_METRICS_FILENAME = "baseline_metrics_before_tuning.json"
PRE_FEATURE_ENGINEERING_METRICS_FILENAME = "pre_feature_engineering_metrics.json"


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
    parser.add_argument(
        "--churn-threshold",
        type=float,
        default=DEFAULT_CHURN_THRESHOLD,
        help="Probability threshold used for final churn labels and metrics.",
    )
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=HIGH_RISK_THRESHOLD,
        help="Probability threshold used for High Risk labels.",
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


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    churn_threshold: float = DEFAULT_CHURN_THRESHOLD,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        predict_churn_labels(probabilities, threshold=churn_threshold)
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )
    confusion_counts = get_confusion_counts(y_test, predictions)
    risk_tier_summary = build_risk_tier_summary(
        y_test,
        probabilities,
        churn_threshold=churn_threshold,
        high_risk_threshold=high_risk_threshold,
    )
    threshold_metrics = build_threshold_metrics(y_test, probabilities)

    metrics = {
        "churn_threshold": churn_threshold,
        "high_risk_threshold": high_risk_threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "confusion_matrix_breakdown": confusion_counts,
        "churn_focus": build_churn_focus_metrics(y_test, predictions, confusion_counts),
        "risk_tier_summary": risk_tier_summary.to_dict(orient="records"),
        "threshold_metrics": threshold_metrics.to_dict(orient="records"),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
            output_dict=True,
        ),
    }

    return metrics, risk_tier_summary, threshold_metrics


def get_confusion_counts(y_true: pd.Series, predictions: pd.Series) -> dict[str, int]:
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "true_no_churn": int(true_negative),
        "false_churn_alerts": int(false_positive),
        "missed_churn_customers": int(false_negative),
        "correctly_identified_churn": int(true_positive),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_churn_focus_metrics(
    y_true: pd.Series,
    predictions: pd.Series,
    confusion_counts: dict[str, int],
) -> dict[str, float | int]:
    actual_churn_customers = int(y_true.sum())
    actual_no_churn_customers = int(len(y_true) - actual_churn_customers)
    predicted_churn_customers = int(predictions.sum())
    predicted_no_churn_customers = int(len(predictions) - predicted_churn_customers)

    true_positive = confusion_counts["correctly_identified_churn"]
    false_positive = confusion_counts["false_churn_alerts"]
    false_negative = confusion_counts["missed_churn_customers"]
    true_negative = confusion_counts["true_no_churn"]

    return {
        "actual_churn_customers": actual_churn_customers,
        "actual_no_churn_customers": actual_no_churn_customers,
        "predicted_churn_customers": predicted_churn_customers,
        "predicted_no_churn_customers": predicted_no_churn_customers,
        "correctly_identified_churn": true_positive,
        "missed_churn_customers": false_negative,
        "false_churn_alerts": false_positive,
        "true_no_churn": true_negative,
        "churn_capture_rate_recall": safe_divide(true_positive, actual_churn_customers),
        "churn_alert_precision": safe_divide(true_positive, predicted_churn_customers),
        "churn_miss_rate": safe_divide(false_negative, actual_churn_customers),
        "false_alarm_rate": safe_divide(false_positive, actual_no_churn_customers),
        "predicted_churn_rate": safe_divide(predicted_churn_customers, len(predictions)),
        "actual_churn_rate": safe_divide(actual_churn_customers, len(y_true)),
    }


def build_risk_tier_summary(
    y_true: pd.Series,
    probabilities: object,
    churn_threshold: float = DEFAULT_CHURN_THRESHOLD,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
) -> pd.DataFrame:
    probability_series = pd.Series(probabilities)
    risk_order = [
        RISK_LEVELS["low"],
        RISK_LEVELS["moderate"],
        RISK_LEVELS["high"],
    ]

    evaluation = pd.DataFrame(
        {
            "actual_churn": y_true.reset_index(drop=True),
            "churn_probability": probability_series,
            "risk_level": assign_risk_levels(
                probability_series,
                churn_threshold=churn_threshold,
                high_risk_threshold=high_risk_threshold,
            ),
        }
    )
    evaluation["risk_level"] = pd.Categorical(
        evaluation["risk_level"],
        categories=risk_order,
        ordered=True,
    )

    summary = (
        evaluation.groupby("risk_level", observed=False)
        .agg(
            total_customers=("actual_churn", "size"),
            actual_churners=("actual_churn", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
            min_churn_probability=("churn_probability", "min"),
            max_churn_probability=("churn_probability", "max"),
        )
        .reset_index()
    )
    summary["actual_non_churners"] = summary["total_customers"] - summary["actual_churners"]
    summary["actual_churn_rate"] = summary.apply(
        lambda row: safe_divide(row["actual_churners"], row["total_customers"]),
        axis=1,
    )
    summary["share_of_test_customers"] = summary["total_customers"] / len(evaluation)
    summary["share_of_actual_churners"] = summary["actual_churners"] / max(
        int(evaluation["actual_churn"].sum()),
        1,
    )

    return summary[
        [
            "risk_level",
            "total_customers",
            "actual_churners",
            "actual_non_churners",
            "actual_churn_rate",
            "share_of_test_customers",
            "share_of_actual_churners",
            "avg_churn_probability",
            "min_churn_probability",
            "max_churn_probability",
        ]
    ]


def build_threshold_metrics(
    y_true: pd.Series,
    probabilities: object,
    thresholds: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
    ),
) -> pd.DataFrame:
    probability_series = pd.Series(probabilities)
    rows = []

    threshold_values = sorted(set(thresholds + (DEFAULT_CHURN_THRESHOLD,)))

    for threshold in threshold_values:
        predictions = probability_series.ge(threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
        predicted_churn_customers = int(predictions.sum())

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "predicted_churn_customers": predicted_churn_customers,
                "correctly_identified_churn": int(true_positive),
                "missed_churn_customers": int(false_negative),
                "false_churn_alerts": int(false_positive),
                "true_no_churn": int(true_negative),
                "predicted_churn_rate": safe_divide(
                    predicted_churn_customers,
                    len(predictions),
                ),
                "churn_miss_rate": safe_divide(false_negative, int(y_true.sum())),
            }
        )

    return pd.DataFrame(rows)


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(to_json_safe(payload), indent=2), encoding="utf-8")


def load_json(input_path: Path) -> dict[str, object]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def get_nested_metric(metrics: dict[str, object], metric_path: str) -> float | int | None:
    current: object = metrics

    for key in metric_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    if isinstance(current, (int, float)):
        return current

    return None


def build_metric_comparison(
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    before_label: str = "before_tuning",
    after_label: str = "after_tuning",
) -> pd.DataFrame:
    comparison_metrics = {
        "accuracy": "Accuracy",
        "precision": "Churn Precision",
        "recall": "Churn Recall",
        "f1": "Churn F1",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average Precision",
        "churn_focus.missed_churn_customers": "Missed Churn Customers",
        "churn_focus.false_churn_alerts": "False Churn Alerts",
        "churn_focus.correctly_identified_churn": "Correctly Identified Churn",
        "churn_focus.churn_miss_rate": "Churn Miss Rate",
        "churn_focus.false_alarm_rate": "False Alarm Rate",
    }
    rows = []

    for metric_path, label in comparison_metrics.items():
        before_value = get_nested_metric(before_metrics, metric_path)
        after_value = get_nested_metric(after_metrics, metric_path)

        if before_value is None or after_value is None:
            continue

        rows.append(
            {
                "metric": label,
                before_label: before_value,
                after_label: after_value,
                "absolute_change": after_value - before_value,
                "relative_change_percent": safe_divide(
                    after_value - before_value,
                    before_value,
                )
                * 100,
            }
        )

    return pd.DataFrame(rows)


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
    metrics, risk_tier_summary, threshold_metrics = evaluate_model(
        best_pipeline,
        X_test,
        y_test,
        churn_threshold=args.churn_threshold,
        high_risk_threshold=args.high_risk_threshold,
    )
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
    risk_tier_path = args.output_dir / "risk_tier_summary.csv"
    threshold_metrics_path = args.output_dir / "threshold_metrics.csv"
    baseline_metrics_path = args.output_dir / BASELINE_METRICS_FILENAME
    pre_feature_metrics_path = args.output_dir / PRE_FEATURE_ENGINEERING_METRICS_FILENAME
    metric_comparison_path = args.output_dir / "metric_comparison_before_after_tuning.csv"
    feature_engineering_comparison_path = (
        args.output_dir / "feature_engineering_comparison.csv"
    )

    joblib.dump(best_pipeline, model_path)
    save_json(metrics, metrics_path)
    pd.DataFrame(grid_search.cv_results_).to_csv(grid_results_path, index=False)
    risk_tier_summary.to_csv(risk_tier_path, index=False)
    threshold_metrics.to_csv(threshold_metrics_path, index=False)

    if baseline_metrics_path.exists():
        before_metrics = load_json(baseline_metrics_path)
        metric_comparison = build_metric_comparison(before_metrics, metrics)
        metric_comparison.to_csv(metric_comparison_path, index=False)

    if pre_feature_metrics_path.exists():
        pre_feature_metrics = load_json(pre_feature_metrics_path)
        feature_engineering_comparison = build_metric_comparison(
            pre_feature_metrics,
            metrics,
            before_label="before_feature_engineering",
            after_label="after_feature_engineering",
        )
        feature_engineering_comparison.to_csv(
            feature_engineering_comparison_path,
            index=False,
        )

    print("Training complete")
    print(f"Best model: {best_pipeline.named_steps['classifier'].__class__.__name__}")
    print(f"Best CV {args.scoring}: {grid_search.best_score_:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved grid results: {grid_results_path}")
    print(f"Saved risk tier summary: {risk_tier_path}")
    print(f"Saved threshold metrics: {threshold_metrics_path}")
    if baseline_metrics_path.exists():
        print(f"Saved before/after comparison: {metric_comparison_path}")
    if pre_feature_metrics_path.exists():
        print(f"Saved feature engineering comparison: {feature_engineering_comparison_path}")


if __name__ == "__main__":
    main()
