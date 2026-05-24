# 📊 Telco Customer Churn ML Pipeline

This project builds a reusable machine learning pipeline for predicting customer churn from the Telco Churn Dataset. It started as a standard Scikit-learn pipeline task, then interpretability was added through churn probability risk levels, and finally the hyperparameters were tuned with `class_weight="balanced"` to reduce missed churn customers.

---

## 🧭 Project Flow Diagram

The project flow below was created with Excalidraw and shows how the work evolved from the base ML pipeline to interpretability and recall-focused fine-tuning.

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

Risk-level logic:

```text
0.00 to 0.49 -> predicted_churn = No  -> Low Risk
0.50 to 0.79 -> predicted_churn = Yes -> Moderate Risk
0.80 to 1.00 -> predicted_churn = Yes -> High Risk
```

This does not change the model itself. It makes the model output easier to read and more useful for churn analysis.

### 3. Hyperparameters Fine Tuned

The final improvement focused on tuning the model while keeping `class_weight="balanced"`. This was important because churn customers are the minority class, and removing balanced class weights caused the model to miss more real churners.

The final tuning command used recall as the scoring metric:

```powershell
python .\train_churn_pipeline.py --scoring recall --n-jobs -1
```

The recall-focused tuned model reduced missed churners from `81` to `79` while keeping the class-balancing behavior.

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

Used to keep the custom data cleaner simple and readable.

- `@dataclass` defines the `TelcoDataCleaner` transformer settings.

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
TelcoDataCleaner -> ColumnTransformer -> Classifier
```

### 4. Preprocess The Data

Numeric features:

- Missing values are filled with the median.
- Values are scaled with `StandardScaler`.

Categorical features:

- Missing values are filled with the most frequent value.
- Text categories are converted with `OneHotEncoder`.

### 5. Train And Tune Models

`GridSearchCV` tunes:

- Logistic Regression
- Random Forest

The final tuning grid keeps:

```text
class_weight = balanced
```

This helps the model pay more attention to churn customers.

### 6. Evaluate The Model

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

### 7. Export And Reuse

The best pipeline is exported as:

```text
outputs/telco_churn_pipeline.joblib
```

`predict_churn.py` loads the saved pipeline and generates prediction output.

---

## 📈 Final Recall-Tuned Results

The latest tuned model was selected using recall scoring while keeping class weights balanced.

Best model:

```text
LogisticRegression
C = 0.3
penalty = l1
class_weight = balanced
```

Key metrics:

```text
Accuracy:          0.7388
Churn precision:   0.5051
Churn recall:      0.7888
F1-score:          0.6159
ROC-AUC:           0.8421
Average precision: 0.6361
```

Churn-focused comparison:

```text
Missed churners:             81 -> 79
Correctly identified churn:  293 -> 295
False churn alerts:          286 -> 289
```

This version slightly improves churn capture while accepting a small increase in false churn alerts.

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
|   |-- accuracy_tuned_metrics.json
|   |-- recall_tuned_balanced_metrics.json
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
python .\train_churn_pipeline.py --scoring recall --n-jobs -1
```

Generate predictions:

```powershell
python .\predict_churn.py
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

---

## ✅ Production-Oriented Practices

- Preprocessing is saved inside the model pipeline.
- New categorical values are handled with `OneHotEncoder(handle_unknown="ignore")`.
- The dataset is split before fitting to reduce data leakage.
- Cross-validation is stratified to respect class imbalance.
- Random states are fixed for reproducible training.
- The trained model is exported and reused without retraining.
- Churn-focused reports are saved for model review.
