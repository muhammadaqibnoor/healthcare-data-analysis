import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Healthcare Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

# Dashboard title
st.title("🏥 Healthcare Readmission Prediction System")

st.markdown("""
### 30-Day Hospital Readmission Risk Estimation

This educational machine learning application estimates
the likelihood of hospital readmission within 30 days.

**Note:** This application is for educational and research
purposes only. It is not intended for clinical decision-making.
""")

st.divider()

st.success("Healthcare dashboard initialized successfully!")

# Sidebar Navigation

st.sidebar.title("🏥 Healthcare Dashboard")

st.sidebar.markdown("### Navigation")

st.sidebar.markdown("""
- [Project Overview](#project-overview)
- [Patient Information](#patient-information)
- [Risk Prediction](#readmission-risk-prediction)
- [Model Performance](#model-performance-comparison)
- [Feature Importance](#xgboost-feature-importance)
- [ROC Curve](#roc-curve-comparison)
- [Confusion Matrices](#confusion-matrices)
""")

st.sidebar.divider()

st.sidebar.subheader("📁 Dataset Information")

st.sidebar.write("Dataset: Diabetes 130-US Hospitals")
st.sidebar.write("Total Encounters: 101,766")
st.sidebar.write("Unique Patients: 71,518")

st.sidebar.divider()

st.sidebar.info(
    "This project is developed for "
    "educational and research purposes."
)

# Load trained machine learning models
@st.cache_resource
def load_models():
    preprocessor = joblib.load(
        REPORTS_DIR / "patient_level_preprocessor.pkl"
    )

    logistic_model = joblib.load(
        REPORTS_DIR / "patient_level_logistic_regression.pkl"
    )

    random_forest_model = joblib.load(
        REPORTS_DIR / "patient_level_random_forest.pkl"
    )

    xgboost_model = joblib.load(
        REPORTS_DIR / "patient_level_xgboost.pkl"
    )

    return (
        preprocessor,
        logistic_model,
        random_forest_model,
        xgboost_model
    )


try:
    (
        preprocessor,
        logistic_model,
        random_forest_model,
        xgboost_model
    ) = load_models()

    st.success("All trained models loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    
st.divider()

# Project overview
st.header("📊 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Encounters", "101,766")

with col2:
    st.metric("Unique Patients", "71,518")

with col3:
    st.metric("Readmission Rate", "11.16%")

st.divider()

st.subheader("Available Machine Learning Models")

st.write("1. Logistic Regression")
st.write("2. Random Forest")
st.write("3. XGBoost")

st.divider()

st.header("🧑‍⚕️ Patient Readmission Risk Prediction")

st.write(
    "Enter patient information below to estimate "
    "the likelihood of readmission within 30 days."
)

# Patient Information Inputs

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox(
        "Patient Age Group",
        [
            "[0-10)", "[10-20)", "[20-30)",
            "[30-40)", "[40-50)", "[50-60)",
            "[60-70)", "[70-80)", "[80-90)",
            "[90-100)"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Unknown/Invalid"]
    )

    race = st.selectbox(
        "Race",
        ["Caucasian", "AfricanAmerican", "Asian",
         "Hispanic", "Other", "Unknown"]
    )

with col2:
    time_in_hospital = st.number_input(
        "Days in Hospital",
        min_value=1,
        max_value=14,
        value=4
    )

# Previous Hospital Visits

st.subheader("Previous Hospital Visits")

col1, col2, col3 = st.columns(3)

with col1:
    number_outpatient = st.number_input(
        "Previous Outpatient Visits",
        min_value=0,
        max_value=100,
        value=0
    )

with col2:
    number_emergency = st.number_input(
        "Previous Emergency Visits",
        min_value=0,
        max_value=100,
        value=0
    )

with col3:
    number_inpatient = st.number_input(
        "Previous Inpatient Visits",
        min_value=0,
        max_value=100,
        value=0
    )
    
# Medical Activity

st.subheader("Medical Activity")

col1, col2, col3, col4 = st.columns(4)

with col1:
    num_lab_procedures = st.number_input(
        "Lab Procedures",
        min_value=0,
        max_value=200,
        value=44
    )

with col2:
    num_procedures = st.number_input(
        "Procedures",
        min_value=0,
        max_value=10,
        value=1
    )

with col3:
    num_medications = st.number_input(
        "Medications",
        min_value=0,
        max_value=100,
        value=15
    )

with col4:
    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=1,
        max_value=16,
        value=8
    )
    
# Diabetes Medication Information

st.subheader("Diabetes Medication Information")

col1, col2 = st.columns(2)

with col1:
    diabetesMed = st.selectbox(
        "Diabetes Medication",
        ["Yes", "No"]
    )

with col2:
    change = st.selectbox(
        "Medication Changed During Stay?",
        ["Ch", "No"]
    )
    
# Admission Information

st.subheader("Admission Information")

col1, col2 = st.columns(2)

with col1:
    admission_type = st.selectbox(
        "Admission Type",
        [
            "Emergency",
            "Urgent",
            "Elective",
            "Not Available",
            "NULL",
            "Not Mapped",
            "Trauma Center",
            "Newborn"
        ]
    )

with col2:
    admission_source = st.selectbox(
        "Admission Source",
        [
            "Emergency Room",
            "Physician Referral",
            "Clinic Referral",
            "HMO Referral",
            "Transfer from Hospital",
            "Transfer from Healthcare Facility",
            "Transfer from Skilled Nursing Facility",
            "Not Available",
            "Not Mapped",
            "Unknown"
        ]
    )
    
# Laboratory Test Results

st.subheader("Laboratory Test Results")

col1, col2 = st.columns(2)

with col1:
    max_glu_serum = st.selectbox(
        "Maximum Glucose Serum",
        ["Not Tested", "Norm", ">200", ">300"]
    )

with col2:
    A1Cresult = st.selectbox(
        "HbA1c Test Result",
        ["Not Tested", "Norm", ">7", ">8"]
    )
    
# Diagnosis Information

st.subheader("Diagnosis Information")

col1, col2, col3 = st.columns(3)

with col1:
    diag_1 = st.text_input(
        "Primary Diagnosis Code",
        value="250.8"
    )

with col2:
    diag_2 = st.text_input(
        "Secondary Diagnosis Code",
        value="Unknown"
    )

with col3:
    diag_3 = st.text_input(
        "Additional Diagnosis Code",
        value="Unknown"
    )
    
# Additional Patient Information

st.subheader("Additional Patient Information")

col1, col2 = st.columns(2)

with col1:
    medical_specialty = st.text_input(
        "Medical Specialty",
        value="Unknown",
        help="Enter the specialty, e.g. Cardiology or InternalMedicine"
    )

with col2:
    payer_code = st.text_input(
        "Payer Code",
        value="Unknown",
        help="Enter the payer code, or Unknown"
    )
    
# Medication Details

st.subheader("Medication Details")

medication_columns = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

medication_inputs = {}

medication_cols = st.columns(3)

for i, medication in enumerate(medication_columns):
    with medication_cols[i % 3]:
        medication_inputs[medication] = st.selectbox(
            medication.replace("-", " ").title(),
            ["No", "Steady", "Up", "Down"],
            key=f"med_{medication}"
        )
        
# Prepare Patient Data for Prediction

patient_data = pd.DataFrame([{
    "race": race,
    "gender": gender,
    "age": age,
    "time_in_hospital": time_in_hospital,
    "payer_code": payer_code.strip() or "Unknown",
    "medical_specialty": medical_specialty.strip() or "Unknown",
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "diag_1": diag_1.strip() or "Unknown",
    "diag_2": diag_2.strip() or "Unknown",
    "diag_3": diag_3.strip() or "Unknown",
    "number_diagnoses": number_diagnoses,
    "max_glu_serum": max_glu_serum,
    "A1Cresult": A1Cresult,
    **medication_inputs,
    "change": change,
    "diabetesMed": diabetesMed,
    "admission_type": admission_type,
    "admission_source": admission_source
}])

# Ensure exact training feature order
feature_order = [
    "race", "gender", "age", "time_in_hospital",
    "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed", "admission_type", "admission_source"
]

patient_data = patient_data[feature_order]

st.write("Patient input data prepared successfully!")
st.write("Input shape:", patient_data.shape)

# 30-Day Readmission Prediction

st.divider()
st.subheader("🔍 Readmission Risk Prediction")

selected_model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "XGBoost",
        "Random Forest",
        "Logistic Regression"
    ]
)

model_options = {
    "XGBoost": xgboost_model,
    "Random Forest": random_forest_model,
    "Logistic Regression": logistic_model
}

selected_model = model_options[selected_model_name]

if st.button("Predict 30-Day Readmission Risk", type="primary"):

    try:
        # Transform patient input
        patient_processed = preprocessor.transform(patient_data)

        # Generate prediction probability
        risk_score = selected_model.predict_proba(
        patient_processed
        )[0, 1]

        # Apply default threshold
        prediction = int(risk_score >= 0.5)

        # Display estimated score
        st.metric(
            "Model-Estimated Readmission Score",
            f"{risk_score * 100:.2f}%"
        )

        if prediction == 1:
            st.warning(
                "Model predicts Readmission Within 30 Days."
            )
        else:
            st.success(
                "Model predicts No Readmission Within 30 Days."
            )

        st.caption(
            "Educational prototype only. "
            "This score is not a clinically validated probability "
            "and must not be used for medical decisions."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        
# Model Performance Comparison

st.divider()
st.header("📈 Model Performance Comparison")

try:
    model_results = pd.read_csv(
        REPORTS_DIR / "patient_level_model_comparison.csv"
    )

    st.subheader("Patient-Level Test Results")

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Performance Metrics Visualization")

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    chart_data = model_results.set_index("Model")[metric_columns]

    st.bar_chart(chart_data)

except Exception as e:
    st.error(f"Could not load model comparison results: {e}")
    
# XGBoost Feature Importance

st.divider()
st.header("🔍 XGBoost Feature Importance")

try:
    feature_importance = pd.read_csv(
        REPORTS_DIR / "patient_level_xgboost_feature_importance.csv"
    )

    top_features = feature_importance.head(20).copy()

    st.subheader("Top 20 Important Features")

    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Feature Importance Visualization")

    importance_chart = top_features.set_index(
        top_features.columns[0]
    )

    importance_chart = importance_chart.iloc[:, [0]]

    st.bar_chart(importance_chart)

except Exception as e:
    st.error(f"Could not load feature importance: {e}")
    
# Patient-Level ROC Curve Comparison

st.divider()
st.header("📉 ROC Curve Comparison")

roc_image_path = BASE_DIR / "images" / "patient_level_roc_comparison.png"

if roc_image_path.exists():
    st.image(
        str(roc_image_path),
        caption="Patient-Level ROC Curves — All Three Models",
        use_container_width=True
    )
else:
    st.warning("ROC comparison image not found.")
    
# Confusion Matrix Comparison

st.divider()
st.header("📊 Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Logistic Regression")
    logistic_cm = BASE_DIR / "images" / "patient_logistic_confusion_matrix.png"

    if logistic_cm.exists():
        st.image(str(logistic_cm), use_container_width=True)

with col2:
    st.subheader("Random Forest")
    rf_cm = BASE_DIR / "images" / "patient_random_forest_confusion_matrix.png"

    if rf_cm.exists():
        st.image(str(rf_cm), use_container_width=True)

st.subheader("XGBoost")
xgb_cm = BASE_DIR / "images" / "patient_xgboost_confusion_matrix.png"

if xgb_cm.exists():
    st.image(str(xgb_cm), use_container_width=True)
    
# Project Methodology

st.divider()
st.header("📚 Project Methodology")

st.subheader("Dataset Information")

st.write("""
This project uses the Diabetes 130-US Hospitals dataset
covering hospital encounters from 1999 to 2008.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", "101,766")

with col2:
    st.metric("Unique Patients", "71,518")

with col3:
    st.metric("Original Features", "50")

st.subheader("Data Preparation")

st.markdown("""
- Removed the weight column because of excessive missing values.
- Replaced unknown categorical markers with Unknown.
- Replaced missing laboratory results with Not Tested.
- Checked duplicate records.
- Prepared a binary 30-day readmission target.
- Excluded patient identifiers and discharge information from predictors.
""")

st.subheader("Machine Learning Workflow")

st.markdown("""
1. Data cleaning and exploratory data analysis.
2. Patient-level train/test split.
3. Numerical feature scaling.
4. Categorical feature encoding.
5. Model training:
   - Logistic Regression
   - Random Forest
   - XGBoost
6. Model evaluation using Precision, Recall, F1-Score and ROC-AUC.
7. Feature importance and prediction dashboard.
""")

# Model Limitations and Ethical Considerations

st.divider()
st.header("⚠️ Model Limitations & Ethical Considerations")

st.subheader("Important Limitations")

st.markdown("""
- The dataset contains historical hospital records from 1999–2008.
- Model performance is moderate and requires further improvement.
- Positive-class precision is relatively low.
- The models have not undergone external clinical validation.
- The displayed scores are not calibrated clinical probabilities.
- The 50% prediction threshold is a default classification threshold.
- The dataset may not represent current hospital populations.
""")

st.subheader("Evaluation Considerations")

st.markdown("""
- Patient-level splitting was used to prevent the same patient
  from appearing in both training and test sets.
- Model selection was based on patient-level test metrics.
- A separate untouched external dataset is required
  for stronger generalization evaluation.
""")

st.subheader("Ethical Disclaimer")

st.warning("""
This application is an educational machine learning prototype.
It must not be used for diagnosis, treatment, patient management,
or real-world clinical decision-making.
""")

# Dataset Source and Attribution

st.divider()
st.header("🔗 Dataset Source & Attribution")

st.markdown("""
**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008

**Source:** UCI Machine Learning Repository

**Original Research:** The dataset contains hospital
encounters involving patients with diabetes.

**Dataset Link:**

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
""")

st.divider()

st.caption(
    "Healthcare Readmission Prediction System | "
    "Developed as an educational Data Science and Machine Learning project."
)