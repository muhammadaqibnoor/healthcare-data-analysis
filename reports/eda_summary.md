
# Healthcare Data Analysis — Exploratory Data Analysis

## 1. Project Overview
This project analyzes the Diabetes 130-US Hospitals dataset
covering hospital encounters from 1999 to 2008.

## 2. Dataset Overview
- Total hospital encounters: 101,766
- Unique patients: 71,518
- Original features: 50
- Duplicate records: 0

## 3. Data Cleaning
- Removed the weight column because approximately 97% of its values were missing.
- Replaced '?' markers with 'Unknown' in relevant categorical columns.
- Replaced missing max_glu_serum and A1Cresult values with 'Not Tested'.
- Preserved the original dataset.
- Saved the cleaned dataset separately.

## 4. Readmission Analysis
Readmission distribution:
- No readmission: 54,864
- Readmitted after 30 days: 35,545
- Readmitted within 30 days: 11,357

## 5. Key Findings

### Age
Older age groups represent a substantial portion of hospital encounters.
Readmission patterns vary across age groups.

### Gender
Female patients had a slightly higher readmission percentage
than male patients in this dataset.

### Hospital Stay
- Average hospital stay: approximately 4.40 days.
- Median hospital stay: 4 days.
- Most common hospital stay: 3 days.

Patients readmitted within 30 days had a higher average
hospital stay than patients who were not readmitted.

### Previous Hospital Visits
Patients with more previous inpatient and emergency visits
generally showed higher readmission percentages.

### Medical Activity
Hospital stays, medications, lab procedures, and diagnoses
showed different relationships with readmission.

### Medication
Patients receiving diabetes medication and patients whose
medication was changed showed different readmission patterns.

### Admission and Discharge
Emergency admissions and certain discharge categories
showed varying readmission patterns.

## 6. Correlation Analysis
Some notable positive correlations:
- Hospital stay and number of medications: approximately 0.47
- Number of procedures and medications: approximately 0.39
- Hospital stay and lab procedures: approximately 0.32

These correlations do not establish causation.

## 7. Important Limitations
- The dataset contains historical hospital records from 1999–2008.
- The analysis is observational and does not establish causal relationships.
- Some categories have very small sample sizes.
- Missing clinical test results may reflect tests that were not performed.
- Readmission patterns should not be interpreted as individual patient risk predictions.

## 8. Next Phase
Prepare the cleaned dataset for machine learning,
define the prediction target, and develop a baseline model.
