# Healthcare 30-Day Readmission Prediction

An educational Data Science and Machine Learning project that analyzes hospital encounter data and builds machine learning models to estimate the likelihood of diabetes-related hospital readmission within 30 days.

## Project Overview

This project uses the **Diabetes 130-US Hospitals dataset (1999–2008)** from the UCI Machine Learning Repository.

- Total hospital encounters: 101,766
- Unique patients: 71,518
- Original dataset features: 50
- Target: Readmission within 30 days
- Machine learning models: Logistic Regression, Random Forest, XGBoost

## Project Objectives

- Perform data cleaning and exploratory data analysis.
- Investigate factors associated with hospital readmission.
- Build a binary classification pipeline.
- Evaluate models using patient-level train/test splitting.
- Visualize model performance and feature importance.
- Develop an interactive Streamlit dashboard.

## Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6389 | 0.1539 | 0.5300 | 0.2386 | 0.6295 |
| Random Forest | 0.6378 | 0.1581 | 0.5532 | 0.2459 | 0.6440 |
| XGBoost | 0.6583 | 0.1621 | 0.5281 | 0.2481 | 0.6438 |

**Evaluation note:** Patients were separated between training and test sets using a group-based split to prevent patient overlap.

## Exploratory Data Analysis

- Readmission distribution
- Age and gender analysis
- Hospital stay analysis
- Previous inpatient, outpatient and emergency visits
- Medication and diagnosis analysis
- Admission and discharge patterns
- Correlation analysis

## Interactive Dashboard

The Streamlit dashboard provides:

- Patient information inputs
- 30-day readmission score estimation
- Selection between three trained models
- Model performance comparison
- XGBoost feature importance
- ROC curve visualization
- Confusion matrices
- Project methodology and limitations

## Project Structure

```text
healthcare-data-analysis/
├── dashboard/
│   └── app.py
├── data/
│   ├── diabetic_data.csv
│   └── diabetic_data_cleaned.csv
├── images/
├── notebooks/
│   └── 01_healthcare_data_analysis.ipynb
├── reports/
│   ├── eda_summary.md
│   ├── model_evaluation.md
│   ├── patient_level_model_comparison.csv
│   ├── patient_level_xgboost_feature_importance.csv
│   └── trained models and preprocessing files
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/muhammadaqibnoor/healthcare-data-analysis.git
cd healthcare-data-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Dataset Source

UCI Machine Learning Repository:

[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Limitations and Disclaimer

- The dataset contains historical records from 1999–2008.
- Positive-class precision is relatively low.
- Model selection used patient-level test metrics; a separate untouched external test dataset is still needed for stronger generalization evaluation.
- Model scores are not clinically calibrated probabilities.
- The models have not undergone clinical validation.

**This project is for educational and research purposes only. It must not be used for diagnosis, treatment or real-world clinical decision-making.**

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Plotly, Joblib, Jupyter Notebook and Streamlit.