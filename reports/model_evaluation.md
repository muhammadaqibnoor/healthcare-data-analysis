
# Healthcare Readmission Prediction
## Machine Learning Model Evaluation Report

### 1. Project Objective

The objective of this project is to analyze historical hospital encounter
data and develop machine learning models to estimate the likelihood of
a patient's readmission within 30 days.

This is an educational research project and is not intended for clinical
decision-making.

### 2. Dataset Overview

- Dataset: Diabetes 130-US Hospitals for Years 1999-2008
- Source: UCI Machine Learning Repository
- Original encounters: 101,766
- Unique patients: 71,518
- Target: Readmission within 30 days

### 3. Data Preparation

The following preprocessing operations were performed:

- Removed the weight column because approximately 96.86% of its values
  were missing.
- Replaced question-mark markers with Unknown in relevant categorical
  columns.
- Replaced missing max_glu_serum and A1Cresult values with Not Tested.
- Created readable admission type, admission source, and discharge
  category labels.
- Removed identifier columns from model features.
- Excluded readmission outcome and discharge disposition information
  from the predictors.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.

The binary target was defined as:

- 1: Readmitted within 30 days (<30)
- 0: Not readmitted within 30 days (NO or >30)

Positive class: 11,357 encounters (11.16%)
Negative class: 90,409 encounters (88.84%)

### 4. Model Evaluation Methodology

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

A patient-level train/test split was performed using GroupShuffleSplit.

- Training encounters: 81,613
- Testing encounters: 20,153
- Training patients: 57,214
- Testing patients: 14,304
- Overlapping patients: 0

The patient identifier was used only for grouping and was not included
as a model feature.

Preprocessing was fitted on the training data and applied to the
testing data. All three models used a classification threshold of 0.5.

### 5. Patient-Level Model Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6389 | 0.1539 | 0.5300 | 0.2386 | 0.6295 |
| Random Forest | 0.6378 | 0.1581 | 0.5532 | 0.2459 | 0.6440 |
| XGBoost | 0.6583 | 0.1621 | 0.5281 | 0.2481 | 0.6438 |

### 6. Model Interpretation

- XGBoost achieved the highest accuracy (65.83%), precision (16.21%),
  and F1-score (0.2481).
- Random Forest achieved the highest recall (55.32%) and ROC-AUC (0.6440).
- Logistic Regression provided a baseline for comparison.
- Random Forest and XGBoost showed very similar ROC-AUC values.

XGBoost achieved the highest F1-score, but the difference from Random
Forest was small. Therefore, neither model can be considered a
definitively superior or clinically reliable solution.

### 7. Important Limitations

- The dataset is historical and covers hospital encounters from
  1999 to 2008.
- The target classes are imbalanced.
- Positive-class precision is low, meaning many predicted readmissions
  are false positives.
- The evaluation uses a random patient-level split, not a temporal
  split. Performance on future time periods has not been established.
- Some predictors, such as length of stay and treatment activity,
  become available during the hospital encounter. The model should
  therefore be interpreted as a retrospective discharge-time
  risk-estimation prototype, not an admission-time prediction system.
- The test results were used to compare the three candidate models.
  An independent external dataset or a separate final holdout would
  be needed for an unbiased final performance estimate.
- The model has not undergone clinical validation.

### 8. Future Improvements

- Perform patient-grouped cross-validation.
- Evaluate threshold selection using training/validation data only.
- Investigate class imbalance and precision-recall trade-offs.
- Explore feature selection and hyperparameter optimization.
- Perform temporal and external validation.
- Improve interpretability using SHAP or other explainability methods.

### 9. Conclusion

This project demonstrates a complete machine learning workflow for
historical hospital readmission analysis, including data cleaning,
exploratory analysis, feature preprocessing, model training,
patient-level evaluation, and model comparison.

XGBoost achieved the highest F1-score in the current patient-level
comparison, while Random Forest achieved the highest recall and
ROC-AUC. The results demonstrate the challenges of predicting
30-day readmission in an imbalanced healthcare dataset.

The project is an educational research prototype and requires
further validation before any real-world healthcare application.
