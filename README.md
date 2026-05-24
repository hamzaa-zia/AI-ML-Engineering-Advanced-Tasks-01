# Telco Customer Churn ML Pipeline

This project builds an end-to-end machine learning pipeline for predicting customer churn using the Telco Churn Dataset. The workflow uses the Scikit-learn `Pipeline` API so preprocessing, model training, hyperparameter tuning, evaluation, model export, and prediction interpretation stay reusable and consistent.

## Objective

- Build a production-ready churn prediction pipeline.
- Preprocess numeric and categorical columns using Scikit-learn transformers.
- Train and tune Logistic Regression and Random Forest models.
- Select the best model using `GridSearchCV`.
- Export the complete fitted pipeline with `joblib`.
- Add tiered churn-risk interpretation from model probabilities.
- Evaluate churn behavior in more depth than accuracy alone.

## Dataset

Dataset file:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains 7,043 customer records and 21 columns. The target column is `Churn`, with values `Yes` and `No`.

Important data handling:

- `customerID` is removed because it is an identifier, not a predictive feature.
- `TotalCharges` is stored as text in the CSV, so the pipeline converts it to numeric values.
- Blank `TotalCharges` values are converted to missing values and imputed during preprocessing.
- The target column is converted from `No` and `Yes` into `0` and `1`.

## Project Structure

```text
.
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- churn_pipeline.py
|-- train_churn_pipeline.py
|-- predict_churn.py
|-- requirements.txt
`-- README.md
```

Generated after training and prediction:

```text
outputs/
|-- telco_churn_pipeline.joblib
|-- metrics.json
|-- grid_search_results.csv
|-- risk_tier_summary.csv
|-- threshold_metrics.csv
|-- baseline_metrics_before_tuning.json
|-- accuracy_tuned_metrics.json
|-- recall_tuned_balanced_metrics.json
|-- metric_comparison_before_after_tuning.csv
`-- churn_predictions.csv
```

## Libraries Used and Their Functions

### pandas

`pandas` is used for loading, cleaning, analyzing, and saving tabular data.

Functions used:

- `pd.read_csv()` loads the Telco churn CSV file.
- `DataFrame.drop()` separates features from the target and removes unwanted columns.
- `Series.map()` converts `Churn` values from `Yes` and `No` into numeric labels.
- `pd.to_numeric()` converts `TotalCharges` from text into numeric values.
- `DataFrame.groupby()` builds the risk-tier evaluation summary.
- `DataFrame.to_csv()` saves grid-search results, risk reports, threshold reports, and prediction outputs.

### scikit-learn

`scikit-learn` is used for preprocessing, model training, hyperparameter tuning, evaluation, and pipeline construction.

Main components used:

- `Pipeline` chains the data cleaner, preprocessing steps, and model into one reusable object.
- `ColumnTransformer` applies different preprocessing to numeric and categorical columns.
- `SimpleImputer` fills missing numeric and categorical values.
- `StandardScaler` scales numeric features for models such as Logistic Regression.
- `OneHotEncoder` converts categorical columns into numeric encoded columns.
- `LogisticRegression` trains a linear classification model for churn prediction.
- `RandomForestClassifier` trains an ensemble tree-based classification model.
- `GridSearchCV` tests multiple model and hyperparameter combinations.
- `StratifiedKFold` keeps the churn and non-churn class ratio balanced during cross-validation.
- `train_test_split` creates separate training and testing datasets.
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, and `average_precision_score` evaluate model performance.
- `classification_report` and `confusion_matrix` provide detailed class-level results.

### joblib

`joblib` is used to save and load the trained pipeline.

Functions used:

- `joblib.dump()` exports the best fitted pipeline to a `.joblib` file.
- `joblib.load()` loads the saved pipeline for future predictions.

### argparse

`argparse` is used to make the scripts configurable from the command line.

Functions used:

- `ArgumentParser()` defines command-line options such as dataset path, output path, test size, and cross-validation folds.
- `parse_args()` reads the selected options when the script runs.

### pathlib

`pathlib` is used for clean and reliable file path handling.

Functions used:

- `Path()` creates file and folder paths.
- `Path.exists()` checks whether input files are available.
- `Path.mkdir()` creates the output folder if it does not already exist.

### json

`json` is used to save model metrics in a structured format.

Functions used:

- `json.dumps()` converts the metrics dictionary into readable JSON text.

### dataclasses

`dataclasses` is used to define the custom `TelcoDataCleaner` transformer cleanly.

Functions used:

- `@dataclass` creates a lightweight class for storing transformer settings.

## Code Workflow

### 1. Load the dataset

`train_churn_pipeline.py` loads `WA_Fn-UseC_-Telco-Customer-Churn.csv` with `pandas`.

### 2. Split features and target

The script separates the input features from the target column:

- Features: all columns except `Churn`
- Target: `Churn`, converted into `0` for `No` and `1` for `Yes`

### 3. Create train and test sets

`train_test_split` divides the data into training and testing sets. Stratification is used so both sets keep a similar churn ratio.

### 4. Build the complete pipeline

`churn_pipeline.py` creates the reusable pipeline:

```text
TelcoDataCleaner -> ColumnTransformer preprocessing -> Classifier
```

The cleaner converts `TotalCharges` and removes `customerID`. The preprocessor imputes missing values, scales numeric columns, and one-hot encodes categorical columns.

### 5. Tune models with GridSearchCV

`GridSearchCV` tests:

- Logistic Regression with a wider range of regularization strengths and penalties while keeping `class_weight="balanced"`.
- Random Forest with different tree counts, tree depths, leaf settings, and feature-selection settings while keeping `class_weight="balanced"`.

The best model is selected using ROC-AUC by default.

### 6. Evaluate the best model

The best pipeline is tested on the holdout test set. The script saves standard metrics and deeper churn-focused reports.

### 7. Export the pipeline

The best complete pipeline is saved as:

```text
outputs/telco_churn_pipeline.joblib
```

This exported file includes the cleaner, preprocessing steps, and trained classifier.

### 8. Reuse the pipeline

`predict_churn.py` loads the `.joblib` file and generates churn probabilities, predicted churn labels, and risk levels for a CSV file with the same feature columns. The prediction output keeps `customerID` and actual `Churn` when they are available, then adds the interpretation fields.

## Risk Level Interpretation

The model returns a churn probability between `0.00` and `1.00`. The prediction script converts that probability into a business-readable risk level:

```text
0.00 to 0.49 -> predicted_churn = No  -> risk_level = Low Risk
0.50 to 0.79 -> predicted_churn = Yes -> risk_level = Moderate Risk
0.80 to 1.00 -> predicted_churn = Yes -> risk_level = High Risk
```

This does not change the model's learned probabilities. It makes the prediction output easier to interpret.

## Deep Churn Evaluation

The training script now creates three evaluation views:

- `metrics.json` stores accuracy, precision, recall, F1-score, ROC-AUC, average precision, confusion matrix details, churn-focused metrics, risk-tier summary, and threshold metrics.
- `risk_tier_summary.csv` shows how many customers and actual churners fall into each risk level.
- `threshold_metrics.csv` compares precision, recall, F1-score, missed churn customers, and false churn alerts at different probability thresholds.
- `metric_comparison_before_after_tuning.csv` compares the saved baseline metrics against the latest tuned model metrics.

Important churn-focused metrics:

- `churn_capture_rate_recall`: how many actual churn customers the model caught.
- `churn_alert_precision`: how many predicted churn alerts were truly churn customers.
- `churn_miss_rate`: how many actual churn customers were missed.
- `false_alarm_rate`: how many non-churn customers were incorrectly flagged as churn.
- `average_precision`: how well the model ranks churn customers when the churn class is imbalanced.

## How to Run

Install the required libraries:

```powershell
pip install -r requirements.txt
```

Train, tune, evaluate, and export the model:

```powershell
python .\train_churn_pipeline.py
```

Run the accuracy-focused hyperparameter tuning comparison:

```powershell
python .\train_churn_pipeline.py --scoring accuracy --n-jobs -1
```

Run the recall-focused tuning mode when the goal is to miss fewer churn customers. This keeps `class_weight="balanced"` in the tuning grid:

```powershell
python .\train_churn_pipeline.py --scoring recall --n-jobs -1
```

The current baseline metrics were preserved in:

```text
outputs/baseline_metrics_before_tuning.json
```

After training, the before-and-after comparison is saved in:

```text
outputs/metric_comparison_before_after_tuning.csv
```

Run prediction with the saved pipeline:

```powershell
python .\predict_churn.py
```

Use custom paths if needed:

```powershell
python .\train_churn_pipeline.py --data-path .\WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir .\outputs
python .\predict_churn.py --model-path .\outputs\telco_churn_pipeline.joblib --input-path .\WA_Fn-UseC_-Telco-Customer-Churn.csv --output-path .\outputs\churn_predictions.csv
```

To use all available CPU cores during grid search:

```powershell
python .\train_churn_pipeline.py --n-jobs -1
```

## Production-Readiness Practices

- The preprocessing logic is part of the saved pipeline, not a separate manual step.
- The code handles unseen categorical values with `OneHotEncoder(handle_unknown="ignore")`.
- The dataset is split before model fitting to avoid test-data leakage.
- Cross-validation is stratified to handle the churn class imbalance more reliably.
- Random states are fixed for reproducible results.
- Metrics, grid-search results, risk-tier summaries, and threshold reports are saved for review.
- The prediction script loads the exported pipeline instead of retraining the model.

## Next Steps

- Try wider hyperparameter tuning with `RandomizedSearchCV`.
- Compare threshold choices based on business goals.
- Add feature importance reporting for the Random Forest model.
- Add a small sample input CSV for deployment testing.
- Package the scripts into a module if the model will be used in an API or dashboard.
