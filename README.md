# 📊 Telco Customer Churn ML Pipeline

This project builds an end-to-end machine learning pipeline to predict customer churn using the Telco Churn Dataset. The pipeline handles cleaning, feature engineering, preprocessing, model tuning, evaluation, and model export.

---

## 🎯 Objective

The goal was to create a reusable churn prediction pipeline using the Scikit-learn `Pipeline` API.

Main focus:

- Predict whether a customer is likely to churn
- Keep `class_weight="balanced"` to avoid missing churners
- Reduce false churn alerts without increasing missed churners
- Export the full trained pipeline with `joblib`

---

## 🧭 Approaches Taken Step By Step

### 1. Built The Base ML Pipeline

The first version created the core workflow:

```text
CSV data -> Clean data -> Preprocess -> Train models -> Tune with GridSearchCV -> Export pipeline
```

The pipeline trains Logistic Regression and Random Forest models, then selects the best model using `GridSearchCV`.

### 2. Added Interpretability

The prediction output was improved from only `Yes/No` to:

- `churn_probability`
- `predicted_churn`
- `risk_level`

Current risk logic:

```text
0.000 to 0.442 -> No churn -> Low Risk
0.443 to 0.799 -> Churn    -> Moderate Risk
0.800 to 1.000 -> Churn    -> High Risk
```

### 3. Tuned Hyperparameters

The model was tuned using recall-focused scoring because missing real churners is costly.

Important decision:

```text
class_weight = balanced
```

This keeps the model more sensitive to churn customers, who are the minority class.

### 4. Added Feature Engineering

Feature engineering was added inside the Scikit-learn pipeline so training and prediction use the same transformations.

New engineered features include:

- Service count
- Support service count
- Streaming service count
- Tenure group
- Monthly charge per tenure month
- High-value short-tenure flag
- Contract, payment, internet, and support interaction features

Final workflow:

```text
TelcoDataCleaner -> TelcoFeatureEngineer -> ColumnTransformer -> Classifier
```

### 5. Tuned The Threshold

After feature engineering, the churn threshold was set to `0.443`.

This reduced false churn alerts while keeping missed churners unchanged compared with the previous `0.40` threshold version.

---

## 🛠️ Libraries Used

- `pandas`: used to load the Telco CSV dataset, clean columns, convert `TotalCharges` into numeric values, split features from the target, create evaluation summaries, and save prediction/report CSV files.

- `scikit-learn`: used for the complete ML workflow. `Pipeline` connects cleaning, feature engineering, preprocessing, and model training. `ColumnTransformer` handles numeric and categorical columns separately. `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` prepare the data. `LogisticRegression`, `RandomForestClassifier`, and `GridSearchCV` train and tune the models. Metric functions such as accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report evaluate performance.

- `joblib`: used to export the complete trained pipeline into a `.joblib` file and load it again for prediction without retraining.

- `argparse`: used to make the training and prediction scripts configurable from the command line, such as choosing the churn threshold, data path, output path, and scoring metric.

- `pathlib`: used for clean and reliable file path handling when reading the dataset, creating output folders, and saving model/report files.

- `json`: used to save model metrics in a structured format so results can be reviewed and compared later.

- `dataclasses`: used to keep custom pipeline transformers such as `TelcoDataCleaner` and `TelcoFeatureEngineer` simple, readable, and easy to configure.

---

## 📈 Final Results

Best model:

```text
LogisticRegression
C = 0.01
penalty = l1
class_weight = balanced
threshold = 0.443
```

Final metrics:

```text
Accuracy:           0.7026
Churn precision:    0.4675
Churn recall:       0.8663
F1-score:           0.6073
ROC-AUC:            0.8449
Average precision:  0.6580
```

Comparison with the previous version:

```text
Missed churners:             50 -> 50
Correctly identified churn:  324 -> 324
False churn alerts:          373 -> 369
```

The final version reduced false alerts by `4` without increasing missed churners.

---

## ▶️ How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train and export the pipeline:

```powershell
python .\train_churn_pipeline.py --scoring recall --churn-threshold 0.443 --n-jobs -1
```

Generate predictions:

```powershell
python .\predict_churn.py
```

---

## 📤 Outputs

- `outputs/telco_churn_pipeline.joblib`: exported complete pipeline
- `outputs/metrics.json`: final evaluation metrics
- `outputs/churn_predictions.csv`: churn probabilities, predicted churn labels, and risk levels
- `outputs/feature_engineering_comparison.csv`: before/after feature engineering comparison
- `outputs/threshold_metrics.csv`: threshold-level metric comparison

---

## ✅ Production-Oriented Practices

- Cleaning, feature engineering, preprocessing, and model training are inside one reusable pipeline.
- New categorical values are handled with `OneHotEncoder(handle_unknown="ignore")`.
- `class_weight="balanced"` is kept to reduce missed churners.
- The model is exported with `joblib` and reused by the prediction script.
- Metrics are saved for review and comparison.
