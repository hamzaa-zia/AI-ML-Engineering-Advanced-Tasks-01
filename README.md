# 📊 Telco Customer Churn ML Pipeline

This project builds a reusable machine learning pipeline for predicting customer churn from the Telco Churn Dataset. It started as a standard Scikit-learn pipeline task, then interpretability was added through churn probability risk levels, hyperparameters were tuned with `class_weight="balanced"`, and feature engineering was added to reduce false churn alerts without increasing missed churners.

---

## 🧭 Project Flow Diagram

The project flow below was created with Excalidraw for the original pipeline stages. The current feature-engineered workflow is:

```text
Telco CSV -> Clean data -> Engineer churn features -> Preprocess -> GridSearchCV -> Threshold 0.443 -> Export + predictions
```

![Project workflow diagram](<Project workflow diagram.png>)

---

## 🎯 Project Objective

The goal of this task was to build an end-to-end churn prediction workflow using the Scikit-learn `Pipeline` API.

The project covers:

- Data cleaning and preprocessing
- Numeric scaling and categorical encoding
- Logistic Regression and Random Forest model training
- Hyperparameter tuning with `GridSearchCV`
- Model export with `joblib`
- Churn probability interpretation with risk levels
- Before-and-after metric comparison after tuning
- Feature engineering and comparison against the last version

---

## 🧩 How The Project Evolved

### 1. Base ML Pipeline

The first version focused on building the core machine learning workflow. The dataset is loaded, cleaned, split into features and target labels, preprocessed, trained, evaluated, and exported as a reusable pipeline.

The base workflow:

```text
Telco CSV -> Data cleaning -> Preprocessing -> Model training -> GridSearchCV -> Exported pipeline
```

### 2. Interpretability Added

After the base model worked, interpretability was added so predictions are easier to understand. Instead of only returning `Yes` or `No`, the prediction script now returns:

- `churn_probability`
- `predicted_churn`
- `risk_level`

The latest version uses a churn threshold of `0.443`. This threshold was selected after feature engineering because it reduces false churn alerts while keeping missed churners unchanged compared with the previous `0.40` version.

Risk-level logic:

```text
0.000 to 0.442 -> predicted_churn = No  -> Low Risk
0.443 to 0.799 -> predicted_churn = Yes -> Moderate Risk
0.80 to 1.00 -> predicted_churn = Yes -> High Risk
```

This does not change the model itself. It makes the model output easier to read and more useful for churn analysis.

### 3. Hyperparameters Fine Tuned

The final improvement focused on tuning the model while keeping `class_weight="balanced"`. This was important because churn customers are the minority class, and removing balanced class weights caused the model to miss more real churners.

The final tuning command used recall as the scoring metric and a lower decision threshold:

```powershell
python .\train_churn_pipeline.py --scoring recall --churn-threshold 0.443 --n-jobs -1
```

### 4. Feature Engineering Added

Feature engineering was added after the recall-tuned version. The goal was to reduce false churn alerts without increasing missed churners.

The new engineered features include:

- `service_count`: number of optional services used by the customer
- `support_service_count`: number of security, backup, protection, and support services
- `streaming_service_count`: number of streaming services
- `tenure_group`: grouped customer tenure ranges
- `monthly_charge_per_tenure_month`: charge intensity compared with tenure
- `total_to_monthly_charge_ratio`: approximate relationship between total and monthly charges
- `is_high_value_short_tenure`: marks newer customers with high monthly charges
- Interaction features such as `contract_tenure_group`, `contract_payment_profile`, and `internet_support_profile`

After feature engineering and threshold adjustment, false churn alerts changed from `373` to `369`, while missed churners stayed at `50`.

---

## 📁 Dataset

Dataset file:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains:

- `7,043` customer records
- `21` columns
- Target column: `Churn`
- Target values: `Yes` and `No`

Important cleaning decisions:

- `customerID` is removed because it is an identifier, not a predictive feature.
- `TotalCharges` is converted from text to numeric values.
- Blank `TotalCharges` values are treated as missing and handled by the pipeline.
- `Churn` is converted into numeric labels: `No = 0`, `Yes = 1`.

---

## 🛠️ Libraries Used And Their Functions

### `pandas`

Used for loading, cleaning, analyzing, and saving tabular data.

- `pd.read_csv()` loads the dataset.
- `DataFrame.drop()` separates features from the target column.
- `Series.map()` converts `Churn` labels into numeric values.
- `pd.to_numeric()` converts `TotalCharges` into numeric format.
- `DataFrame.groupby()` builds risk-tier evaluation summaries.
- `DataFrame.to_csv()` saves prediction outputs and evaluation reports.

### `scikit-learn`

Used for preprocessing, model training, hyperparameter tuning, evaluation, and pipeline construction.

- `Pipeline` chains cleaning, preprocessing, and model training into one reusable workflow.
- `BaseEstimator` and `TransformerMixin` make custom cleaning and feature engineering steps work inside a Scikit-learn pipeline.
- `ColumnTransformer` applies separate transformations to numeric and categorical columns.
- `SimpleImputer` fills missing values.
- `StandardScaler` scales numeric columns.
- `OneHotEncoder` converts categorical text columns into numeric encoded columns.
- `LogisticRegression` trains a linear churn classifier.
- `RandomForestClassifier` trains an ensemble tree-based classifier.
- `GridSearchCV` tests hyperparameter combinations.
- `StratifiedKFold` keeps churn/non-churn class balance during cross-validation.
- `train_test_split` creates training and test datasets.
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, and `average_precision_score` evaluate model performance.
- `classification_report` and `confusion_matrix` provide class-level evaluation.

### `joblib`

Used for model persistence.

- `joblib.dump()` saves the complete trained pipeline.
- `joblib.load()` loads the saved pipeline for prediction.

### `argparse`

Used to make scripts configurable from the command line.

- `ArgumentParser()` defines command-line options.
- `parse_args()` reads selected options when the script runs.

### `pathlib`

Used for clean file and folder path handling.

- `Path()` creates path objects.
- `Path.exists()` checks whether required files exist.
- `Path.mkdir()` creates output folders.

### `json`

Used to save evaluation metrics in a readable structured format.

- `json.dumps()` converts metrics dictionaries into JSON text.

### `dataclasses`

Used to keep the custom data cleaner and feature engineer simple and readable.

- `@dataclass` defines the `TelcoDataCleaner` and `TelcoFeatureEngineer` transformer settings.

---

## ⚙️ Code Workflow

### 1. Load Data

`train_churn_pipeline.py` loads the Telco churn CSV file with `pandas`.

### 2. Split Features And Target

The script separates:

- Features: all columns except `Churn`
- Target: `Churn`, mapped from `Yes/No` to `1/0`

### 3. Build The Pipeline

`churn_pipeline.py` builds this workflow:

```text
TelcoDataCleaner -> TelcoFeatureEngineer -> ColumnTransformer -> Classifier
```

### 4. Engineer Churn Features

`TelcoFeatureEngineer` creates behavior and interaction features before scaling and encoding. These features help the model separate customers who only look risky from customers who are more likely to churn.

Examples:

- Service usage counts
- Support service counts
- Tenure groups
- High-charge and short-tenure flags
- Contract, payment, internet, support, and billing interaction features

### 5. Preprocess The Data

Numeric features:

- Missing values are filled with the median.
- Values are scaled with `StandardScaler`.

Categorical features:

- Missing values are filled with the most frequent value.
- Text categories are converted with `OneHotEncoder`.

### 6. Train And Tune Models

`GridSearchCV` tunes:

- Logistic Regression
- Random Forest

The final tuning grid keeps:

```text
class_weight = balanced
```

This helps the model pay more attention to churn customers.

### 7. Evaluate The Model

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision
- Confusion matrix
- Churn-focused metrics
- Risk-tier summary
- Threshold comparison

### 8. Export And Reuse

The best pipeline is exported as:

```text
outputs/telco_churn_pipeline.joblib
```

`predict_churn.py` loads the saved pipeline and generates prediction output.

---

## 📈 Final Recall-Tuned Results

The latest tuned model uses feature engineering, recall scoring, and balanced class weights. The decision threshold is `0.443`, selected to reduce false churn alerts without increasing missed churners compared with the previous version.

Best model:

```text
LogisticRegression
C = 0.01
penalty = l1
class_weight = balanced
```

Key metrics:

```text
Decision threshold: 0.443
Accuracy:           0.7026
Churn precision:    0.4675
Churn recall:       0.8663
F1-score:           0.6073
ROC-AUC:           0.8449
Average precision: 0.6580
```

Comparison with the last `0.40` version:

```text
Missed churners:             50 -> 50
Correctly identified churn:  324 -> 324
False churn alerts:          373 -> 369
Churn precision:             0.4648 -> 0.4675
Average precision:           0.6361 -> 0.6580
```

This version keeps the same churn capture level while slightly reducing false alerts. The improvement is small, but it matches the project constraint: do not increase missed churners.

---

## 📂 Project Structure

```text
.
|-- outputs/
|   |-- telco_churn_pipeline.joblib
|   |-- metrics.json
|   |-- grid_search_results.csv
|   |-- risk_tier_summary.csv
|   |-- threshold_metrics.csv
|   |-- baseline_metrics_before_tuning.json
|   |-- pre_feature_engineering_metrics.json
|   |-- pre_feature_engineering_risk_tier_summary.csv
|   |-- pre_feature_engineering_threshold_metrics.csv
|   |-- feature_engineering_comparison.csv
|   `-- metric_comparison_before_after_tuning.csv
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- Project workflow diagram.png
|-- churn_pipeline.py
|-- train_churn_pipeline.py
|-- predict_churn.py
|-- requirements.txt
`-- README.md
```

---

## ▶️ How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train, tune, evaluate, and export the pipeline:

```powershell
python .\train_churn_pipeline.py
```

Run recall-focused tuning to miss fewer churn customers:

```powershell
python .\train_churn_pipeline.py --scoring recall --churn-threshold 0.443 --n-jobs -1
```

Generate predictions:

```powershell
python .\predict_churn.py
```

Generate predictions with a custom churn threshold:

```powershell
python .\predict_churn.py --churn-threshold 0.443
```

Run with custom paths:

```powershell
python .\train_churn_pipeline.py --data-path .\WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir .\outputs
python .\predict_churn.py --model-path .\outputs\telco_churn_pipeline.joblib --input-path .\WA_Fn-UseC_-Telco-Customer-Churn.csv --output-path .\outputs\churn_predictions.csv
```

---

## 📤 Outputs

Main generated files:

- `outputs/telco_churn_pipeline.joblib`: complete trained pipeline
- `outputs/metrics.json`: detailed model evaluation
- `outputs/grid_search_results.csv`: GridSearchCV results
- `outputs/risk_tier_summary.csv`: churn distribution by risk level
- `outputs/threshold_metrics.csv`: threshold-level precision, recall, and miss-rate comparison
- `outputs/churn_predictions.csv`: prediction output with probability, churn label, and risk level
- `outputs/metric_comparison_before_after_tuning.csv`: before-and-after tuning comparison
- `outputs/pre_feature_engineering_metrics.json`: saved metrics from the last version before feature engineering
- `outputs/feature_engineering_comparison.csv`: comparison between the last version and the feature-engineered version

---

## ✅ Production-Oriented Practices

- Preprocessing is saved inside the model pipeline.
- Feature engineering is saved inside the model pipeline, so prediction data gets the same engineered columns as training data.
- New categorical values are handled with `OneHotEncoder(handle_unknown="ignore")`.
- The dataset is split before fitting to reduce data leakage.
- Cross-validation is stratified to respect class imbalance.
- Random states are fixed for reproducible training.
- The trained model is exported and reused without retraining.
- Churn-focused reports are saved for model review.
