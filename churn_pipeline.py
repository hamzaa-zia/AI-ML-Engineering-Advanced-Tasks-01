"""Reusable pipeline components for the Telco customer churn task."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@dataclass
class TelcoDataCleaner(BaseEstimator, TransformerMixin):
    """Clean raw Telco churn rows before model preprocessing."""

    id_column: str = ID_COLUMN
    total_charges_column: str = "TotalCharges"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TelcoDataCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = X.copy()

        if self.total_charges_column in cleaned.columns:
            cleaned[self.total_charges_column] = pd.to_numeric(
                cleaned[self.total_charges_column],
                errors="coerce",
            )

        if self.id_column in cleaned.columns:
            cleaned = cleaned.drop(columns=[self.id_column])

        return cleaned


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer for numeric and categorical data."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline() -> Pipeline:
    """Build a full trainable pipeline with a default classifier."""

    return Pipeline(
        steps=[
            ("cleaner", TelcoDataCleaner()),
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_param_grid() -> list[dict[str, object]]:
    """Return model and hyperparameter options for GridSearchCV."""

    logistic_regression = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    random_forest = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return [
        {
            "classifier": [logistic_regression],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"],
        },
        {
            "classifier": [random_forest],
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_leaf": [1, 2],
        },
    ]
