# DS422: Machine Learning I &mdash; Lab Project Report

**Project Title:** Benchmarking Supervised Classifiers for Diabetes Risk Screening Using CDC BRFSS Health Indicators  
**Course:** DS422 (Machine Learning I)  
**Section:** E1  
**Instructor:** Teacher FH  
**Student ID:** 232-35-733 (Shihab)  
**Submission Date:** 22 November 2026  
**Live Showcase:** [https://mlids422.vercel.app](https://mlids422.vercel.app)  
**Online Report:** [https://mlids422.vercel.app/report.html](https://mlids422.vercel.app/report.html)  
**Direct PDF Download:** [https://mlids422.vercel.app/DS422_Lab_Report_232-35-733.pdf](https://mlids422.vercel.app/DS422_Lab_Report_232-35-733.pdf)

---

## Abstract
Diabetes mellitus represents one of the most critical global chronic healthcare burdens, with millions remaining undiagnosed until severe microvascular and macrovascular complications develop. This investigation conducts an end-to-end empirical benchmark of supervised machine learning architectures—**Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**—applied to 253,680 public health survey records from the CDC's Behavioral Risk Factor Surveillance System (BRFSS 2015). Addressing a realistic 84.7% to 15.3% class imbalance, we implement rigorous deduplication (removing 24,206 duplicate profiles to prevent leakage), percentile-based BMI outlier mitigation, leak-free standardized feature pipelines, and cost-sensitive loss weighting. Evaluated on a 20% stratified test set (45,895 samples), **Random Forest** emerged as the premier screening classifier, attaining an **ROC-AUC of 0.8176**, a **Balanced Accuracy of 0.7399**, and an affirmative clinical **Recall of 75.69%** (identifying 5,313 out of 7,019 diabetic patients). Feature attribution identified General Health perception (`GenHlth`), High Blood Pressure (`HighBP`), Body Mass Index (`BMI`), Age bracket (`Age`), and High Cholesterol (`HighChol`) as the dominant non-invasive risk drivers.

---

## 1. Introduction & Problem Formulation
According to the Centers for Disease Control and Prevention (CDC), over 37 million Americans suffer from diabetes, while an estimated 96 million adults exhibit prediabetes. Because early-stage hyperglycemia is frequently asymptomatic, non-invasive triage models capable of flagging high-risk individuals using routine lifestyle, clinical history, and demographic indicators are vital for early intervention.

From a machine learning engineering standpoint, disease screening presents asymmetric error costs. A **False Negative (Type II error)**—failing to detect an undiagnosed diabetic patient—leads to disease progression and costly acute emergencies. Conversely, a **False Positive (Type I error)** merely prompts a confirmatory blood test (e.g., HbA1c or fasting plasma glucose). Consequently, standard classification Accuracy is a deceptive performance indicator due to severe class imbalance (~85% healthy vs. ~15% diabetic). This research formulates diabetes detection as an imbalanced, cost-sensitive classification problem prioritizing **Sensitivity (Recall)**, **Balanced Accuracy**, and **Area Under the ROC Curve (ROC-AUC)** over raw accuracy.

---

## 2. Dataset Description & Engineering
The study employs the CDC BRFSS 2015 health indicators dataset mirrored on the UCI Machine Learning Repository (ID: 891). The dataset comprises 253,680 raw respondent profiles across 21 predictive variables capturing clinical diagnoses, dietary habits, physical mobility, and socioeconomic indicators, coupled with a binary target: `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes / Diabetes).

### 2.1 Preprocessing Pipeline
1. **Exact Deduplication:** Comprehensive audit identified 24,206 duplicate rows arising from survey response permutations. Dropping these exact duplicates reduced the cohort to **229,474 unique records**, strictly avoiding data leakage between training and evaluation partitions.
2. **Outlier Mitigation:** Self-reported Body Mass Index (`BMI`) reached biologically implausible values up to 98. Extreme values were capped at the 1st and 99th percentiles (`[18.0, 50.0]`), preserving sample integrity while stabilizing gradient-based and linear estimators.
3. **Stratified Partitioning:** An 80/20 train/test split was executed with target stratification (`random_state=42`), yielding **183,579 training records** and **45,895 test records**, preserving the exact 15.29% prevalence across sets.
4. **Leak-Free Scaling:** Continuous and ordinal features were scaled using `StandardScaler` strictly within a cross-validation pipeline fitted solely on training partitions.

### 2.2 Feature Dictionary
| Attribute | Type | Category | Description & Encoding Scale |
| :--- | :--- | :--- | :--- |
| `Diabetes_binary` | Binary | Target | 0 = Healthy, 1 = Prediabetes or Diagnosed Diabetes |
| `HighBP` | Binary | Clinical | Adults diagnosed with hypertension (0=No, 1=Yes) |
| `HighChol` | Binary | Clinical | Adults told their blood cholesterol is elevated (0=No, 1=Yes) |
| `CholCheck` | Binary | Clinical | Cholesterol checked within the past 5 years (0=No, 1=Yes) |
| `BMI` | Continuous | Clinical | Body Mass Index (clipped to 1st–99th percentile: 18.0–50.0) |
| `Smoker` | Binary | Lifestyle | Smoked at least 100 cigarettes in lifetime (0=No, 1=Yes) |
| `Stroke` | Binary | Clinical | Ever told respondent had a stroke (0=No, 1=Yes) |
| `HeartDiseaseorAttack` | Binary | Clinical | Coronary heart disease (CHD) or myocardial infarction (0=No, 1=Yes) |
| `PhysActivity` | Binary | Lifestyle | Physical activity in past 30 days excluding job (0=No, 1=Yes) |
| `Fruits` / `Veggies` | Binary | Lifestyle | Consumes fruit / vegetables &ge;1 time per day (0=No, 1=Yes) |
| `HvyAlcoholConsump` | Binary | Lifestyle | Heavy drinker: Men &gt;14 drinks/wk, Women &gt;7 drinks/wk |
| `GenHlth` | Ordinal | Clinical | Subjective health rating: 1=Excellent, 2=Very Good ... 5=Poor |
| `MentHlth` / `PhysHlth` | Discrete | Clinical | Days of poor mental / physical health in past 30 days (0–30) |
| `DiffWalk` | Binary | Clinical | Difficulty walking or climbing stairs (0=No, 1=Yes) |
| `Sex` | Binary | Demographic | Respondent biological sex (0 = Female, 1 = Male) |
| `Age` | Ordinal | Demographic | 13-level bracket: 1=18–24 up to 13=80+ years |
| `Education` / `Income` | Ordinal | Socioeconomic | Levels 1–6 (Education) and 1–8 (Household income) |

---

## 3. Exploratory Data Analysis (EDA)
- **Target Imbalance:** 194,377 healthy cases (84.71%) vs. 35,097 diabetic cases (15.29%). An unadjusted model predicting class 0 would achieve 84.7% accuracy with 0% utility.
- **Monotonic Risk Factors:** Diabetes prevalence scales sharply with poorer `GenHlth` (from 2.1% in excellent health to 38.4% in poor health), doubles in the presence of `HighBP` (6.2% vs. 27.2%), and steadily increases across `Age` categories.
- **Linear Correlations:** Highest linear correlations with diabetes diagnosis are observed in `GenHlth` (r = 0.29), `HighBP` (r = 0.26), `DiffWalk` (r = 0.21), `BMI` (r = 0.21), and `HighChol` (r = 0.20).

---

## 4. Model Architectures & Cost-Sensitive Setup
Four models were trained with cost-sensitive class balancing on 183,579 training records:
1. **Logistic Regression:** Scaled linear baseline with L2 penalty and `class_weight="balanced"`.
2. **Decision Tree:** Regularized CART classifier (`max_depth=10`, `min_samples_leaf=50`, `class_weight="balanced"`).
3. **Random Forest:** Bagged ensemble of 200 trees (`max_depth=14`, `min_samples_leaf=20`, `class_weight="balanced"`).
4. **XGBoost:** Gradient boosted decision trees (250 trees, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=5.54`).

---

## 5. Empirical Results (Test Set: 45,895 instances)

| Classifier Model | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | F1-Score | Balanced Acc | Raw Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Best)** | **0.8176** | 0.4431 | **0.7569** | **0.3303** | **0.4599** | 0.7399 | **0.7281** |
| **XGBoost** | 0.8170 | **0.4437** | 0.7786 | 0.3201 | 0.4537 | **0.7400** | 0.7132 |
| **Logistic Regression** | 0.8120 | 0.4268 | 0.7619 | 0.3191 | 0.4498 | 0.7342 | 0.7149 |
| **Decision Tree** | 0.8046 | 0.4210 | 0.7671 | 0.3078 | 0.4393 | 0.7278 | 0.7005 |

### Confusion Matrix Breakdown (Random Forest)
- **True Negatives (TN):** 28,103
- **False Positives (FP):** 10,773
- **False Negatives (FN):** 1,706
- **True Positives (TP):** 5,313
- **Clinical Sensitivity:** **75.69%** (5,313 out of 7,019 diabetic patients correctly flagged).

---

## 6. Feature Importance & Interpretability
The top 12 predictive features identified by Gini importance in the Random Forest model:
1. `GenHlth` (14.2%): Strongest single non-invasive risk predictor.
2. `HighBP` (11.8%): Strongly tied to metabolic syndrome and microvascular strain.
3. `BMI` (11.2%): Adiposity and insulin resistance biomarker.
4. `Age` (9.8%): Age-related pancreatic beta-cell decline.
5. `HighChol` (7.8%): Dyslipidemia component.
6. `Income` (6.2%) & `Education` (3.8%): Social determinants of health.
7. `PhysHlth` (5.5%) & `DiffWalk` (4.8%): Physical functioning impairment.
8. `HeartDiseaseorAttack` (4.2%): Cardiovascular comorbidity.
9. `MentHlth` (3.6%) & `PhysActivity` (3.2%): Psychological well-being and lifestyle.

---

## 7. Conclusions & Next Steps
- **Engineering Verdict:** Cost-sensitive ensemble learning (Random Forest) is superior to single decision trees and linear baselines, achieving an ROC-AUC of 0.8176 and capturing over 75% of positive cases without invasive clinical lab tests.
- **Future Work:** Clinical decision threshold optimization (&theta; tuning), integration with electronic health records (EHR), and SHAP local explainability.

---
*Report generated for DS422: Machine Learning I &middot; Section E1 &middot; Faculty: FH &middot; Student ID: 232-35-733*
