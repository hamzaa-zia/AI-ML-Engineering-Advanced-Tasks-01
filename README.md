# Telco Customer Churn ML Pipeline

This project builds an end-to-end machine learning pipeline for predicting customer churn using the Telco Churn Dataset. The workflow uses the Scikit-learn `Pipeline` API so preprocessing, model training, hyperparameter tuning, evaluation, and model export are handled in a reusable structure.

## Objective

- Build a production-ready churn prediction pipeline.
- Preprocess numeric and categorical columns using Scikit-learn transformers.
- Train and tune Logistic Regression and Random Forest models.
- Select the best model using `GridSearchCV`.
- Export the complete fitted pipeline with `joblib`.
- Reuse the saved pipeline for future predictions.

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
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_pipeline.py
├── train_churn_pipeline.py
├── predict_churn.py
├── requirements.txt
└── README.md
```

Generated after training:

```text
outputs/
├── telco_churn_pipeline.joblib
├── metrics.json
├── grid_search_results.csv
└── churn_predictions.csv
```

## Libraries Used and Their Functions

### pandas

`pandas` is used for loading, inspecting, cleaning, and saving tabular data.

Functions used:

- `pd.read_csv()` loads the Telco churn CSV file.
- `DataFrame.drop()` separates features from the target and removes unwanted columns.
- `Series.map()` converts `Churn` values from `Yes` and `No` into numeric labels.
- `pd.to_numeric()` converts `TotalCharges` from text into numeric values.
- `DataFrame.to_csv()` saves grid-search results and prediction outputs.

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
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `roc_auc_score` evaluate model performance.
- `classification_report` and `confusion_matrix` provide detailed classification results.

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
TelcoDataCleaner
→ ColumnTransformer preprocessing
→ Classifier
```

The cleaner converts `TotalCharges` and removes `customerID`. The preprocessor imputes missing values, scales numeric columns, and one-hot encodes categorical columns.

### 5. Tune models with GridSearchCV

`GridSearchCV` tests:

- Logistic Regression with different regularization settings.
- Random Forest with different tree depth, number of trees, and split settings.

The best model is selected using ROC-AUC by default.

### 6. Evaluate the best model

The best pipeline is tested on the holdout test set. The script saves:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### 7. Export the pipeline

The best complete pipeline is saved as:

```text
outputs/telco_churn_pipeline.joblib
```

This exported file includes the cleaner, preprocessing steps, and trained classifier.

### 8. Reuse the pipeline

`predict_churn.py` loads the `.joblib` file and generates churn predictions for a CSV file with the same feature columns.

## How to Run

Install the required libraries:

```powershell
pip install -r requirements.txt
```

Train, tune, evaluate, and export the model:

```powershell
python .\train_churn_pipeline.py
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
- Metrics and grid-search results are saved for review.
- The prediction script loads the exported pipeline instead of retraining the model.

## Next Steps

- Compare threshold tuning if recall for churned customers is more important than accuracy.
- Add feature importance reporting for the Random Forest model.
- Add a small sample input CSV for deployment testing.
- Package the scripts into a module if the model will be used in an API or dashboard.
