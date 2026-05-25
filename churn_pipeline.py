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
DEFAULT_CHURN_THRESHOLD = 0.443
HIGH_RISK_THRESHOLD = 0.80

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count",
    "support_service_count",
    "streaming_service_count",
    "monthly_charge_per_tenure_month",
    "total_to_monthly_charge_ratio",
    "is_new_customer",
    "is_high_monthly_charge",
    "is_high_value_short_tenure",
    "is_auto_payment",
    "has_streaming_service",
    "has_support_service",
    "has_fiber_optic_internet",
    "month_to_month_short_tenure",
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
    "tenure_group",
    "contract_tenure_group",
    "contract_payment_profile",
    "internet_support_profile",
    "phone_internet_bundle",
    "billing_contract_profile",
]

OPTIONAL_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SUPPORT_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

STREAMING_SERVICE_COLUMNS = [
    "StreamingTV",
    "StreamingMovies",
]

RISK_LEVELS = {
    "low": "Low Risk",
    "moderate": "Moderate Risk",
    "high": "High Risk",
}


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


@dataclass
class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create churn-focused behavior and interaction features."""

    new_customer_months: int = 12
    high_charge_quantile: float = 0.75

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TelcoFeatureEngineer":
        monthly_charges = self._numeric_column(X, "MonthlyCharges")
        high_monthly_charge_cutoff = monthly_charges.quantile(self.high_charge_quantile)
        if pd.isna(high_monthly_charge_cutoff):
            high_monthly_charge_cutoff = 0.0
        self.high_monthly_charge_cutoff_ = float(high_monthly_charge_cutoff)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        engineered = X.copy()

        tenure = self._numeric_column(engineered, "tenure").fillna(0)
        monthly_charges = self._numeric_column(engineered, "MonthlyCharges").fillna(0)
        total_charges = self._numeric_column(engineered, "TotalCharges").fillna(0)
        contract = self._text_column(engineered, "Contract")
        payment_method = self._text_column(engineered, "PaymentMethod")
        internet_service = self._text_column(engineered, "InternetService")
        tech_support = self._text_column(engineered, "TechSupport")
        phone_service = self._text_column(engineered, "PhoneService")
        paperless_billing = self._text_column(engineered, "PaperlessBilling")

        service_count = self._yes_count(engineered, OPTIONAL_SERVICE_COLUMNS)
        support_service_count = self._yes_count(engineered, SUPPORT_SERVICE_COLUMNS)
        streaming_service_count = self._yes_count(engineered, STREAMING_SERVICE_COLUMNS)
        tenure_group = self._tenure_group(tenure)
        high_monthly_charge_cutoff = getattr(self, "high_monthly_charge_cutoff_", 0.0)

        engineered["service_count"] = service_count
        engineered["support_service_count"] = support_service_count
        engineered["streaming_service_count"] = streaming_service_count
        engineered["monthly_charge_per_tenure_month"] = monthly_charges.div(
            tenure.clip(lower=1)
        )
        engineered["total_to_monthly_charge_ratio"] = total_charges.div(
            monthly_charges.where(monthly_charges > 0)
        ).fillna(0)
        engineered["is_new_customer"] = tenure.le(self.new_customer_months).astype(int)
        engineered["is_high_monthly_charge"] = monthly_charges.ge(
            high_monthly_charge_cutoff
        ).astype(int)
        engineered["is_high_value_short_tenure"] = (
            engineered["is_new_customer"].eq(1)
            & engineered["is_high_monthly_charge"].eq(1)
        ).astype(int)
        engineered["is_auto_payment"] = payment_method.str.contains(
            "automatic",
            case=False,
            na=False,
        ).astype(int)
        engineered["has_streaming_service"] = streaming_service_count.gt(0).astype(int)
        engineered["has_support_service"] = support_service_count.gt(0).astype(int)
        engineered["has_fiber_optic_internet"] = internet_service.eq("Fiber optic").astype(
            int
        )
        engineered["month_to_month_short_tenure"] = (
            contract.eq("Month-to-month") & tenure.le(self.new_customer_months)
        ).astype(int)

        engineered["tenure_group"] = tenure_group
        engineered["contract_tenure_group"] = contract + "_" + tenure_group
        engineered["contract_payment_profile"] = contract + "_" + payment_method
        engineered["internet_support_profile"] = internet_service + "_" + tech_support
        engineered["phone_internet_bundle"] = phone_service + "_" + internet_service
        engineered["billing_contract_profile"] = paperless_billing + "_" + contract

        return engineered

    def _numeric_column(self, data: pd.DataFrame, column: str) -> pd.Series:
        if column not in data.columns:
            return pd.Series(0.0, index=data.index)
        return pd.to_numeric(data[column], errors="coerce")

    def _text_column(self, data: pd.DataFrame, column: str) -> pd.Series:
        if column not in data.columns:
            return pd.Series("Unknown", index=data.index)
        return data[column].fillna("Unknown").astype(str)

    def _yes_count(self, data: pd.DataFrame, columns: list[str]) -> pd.Series:
        yes_flags = [
            self._text_column(data, column).eq("Yes").astype(int) for column in columns
        ]
        return pd.concat(yes_flags, axis=1).sum(axis=1)

    def _tenure_group(self, tenure: pd.Series) -> pd.Series:
        grouped = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, float("inf")],
            labels=["0-12", "13-24", "25-48", "49+"],
        )
        return grouped.astype("object").fillna("Unknown").astype(str)


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
            ("feature_engineer", TelcoFeatureEngineer()),
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
            "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__class_weight": ["balanced"],
        },
        {
            "classifier": [random_forest],
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [6, 10, 14, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__class_weight": ["balanced"],
        },
    ]


def predict_churn_labels(
    probabilities: object,
    threshold: float = DEFAULT_CHURN_THRESHOLD,
) -> pd.Series:
    """Convert churn probabilities into Yes/No labels using a decision threshold."""

    probability_series = pd.Series(probabilities)
    return probability_series.ge(threshold).map({False: "No", True: "Yes"})


def assign_risk_levels(
    probabilities: object,
    churn_threshold: float = DEFAULT_CHURN_THRESHOLD,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
) -> pd.Series:
    """Convert churn probabilities into low, moderate, and high risk levels."""

    probability_series = pd.Series(probabilities)
    risk_levels = pd.Series(RISK_LEVELS["high"], index=probability_series.index)
    risk_levels = risk_levels.mask(
        probability_series < high_risk_threshold,
        RISK_LEVELS["moderate"],
    )
    risk_levels = risk_levels.mask(
        probability_series < churn_threshold,
        RISK_LEVELS["low"],
    )
    return risk_levels
