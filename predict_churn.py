"""Load the exported churn pipeline and generate predictions for new rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn_pipeline import ID_COLUMN, TARGET_COLUMN, assign_risk_levels, predict_churn_labels


DEFAULT_MODEL_PATH = Path("outputs") / "telco_churn_pipeline.joblib"
DEFAULT_INPUT_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_OUTPUT_PATH = Path("outputs") / "churn_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate customer churn predictions with the exported pipeline."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained joblib pipeline.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV file containing Telco customer rows.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file where predictions will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_path}")

    model = joblib.load(args.model_path)
    data = pd.read_csv(args.input_path)
    prediction_features = data.drop(columns=[TARGET_COLUMN], errors="ignore")

    churn_probability = model.predict_proba(prediction_features)[:, 1]

    identifier_columns = [
        column for column in [ID_COLUMN, TARGET_COLUMN] if column in data.columns
    ]

    output = data[identifier_columns].copy()
    output["churn_probability"] = churn_probability
    output["predicted_churn"] = predict_churn_labels(churn_probability)
    output["risk_level"] = assign_risk_levels(churn_probability)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)

    print(f"Saved predictions: {args.output_path}")


if __name__ == "__main__":
    main()
