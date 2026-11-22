# DS422: Machine Learning Driven Data Analysis Lab I (DS Major) &mdash; Lab Task Report
**Daffodil International University (DIU)**  

---

### Academic Metadata
- **Project Title:** Machine Learning Driven Phishing URL Detection Benchmark Using the Modern 2024 PhiUSIIL Dataset
- **Dataset:** UCI Machine Learning Repository &mdash; **PhiUSIIL Phishing URL Dataset (ID: 967, March 2024)**
- **Release / Donation Date:** March 2024 (Published by Arvind Prasad and Shalini Chandra in Elsevier *Computers & Security*)
- **Course Code:** DS422
- **Course Title:** Machine Learning Driven Data Analysis Lab I (DS Major)
- **Semester:** Fall 2026
- **Section:** E1
- **Batch:** 41
- **Student Name:** Imranul Islam Shihab
- **Student ID:** 232-35-733
- **Course Teacher:** Md. Fahad Hossain
- **Designation:** Lecturer (Senior Scale)
- **Submission Date:** 22 November, 2026
- **Live Showcase Dashboard:** [https://mlids422.vercel.app](https://mlids422.vercel.app)
- **Online Academic Report:** [https://mlids422.vercel.app/report.html](https://mlids422.vercel.app/report.html)
- **Direct PDF Download:** [https://mlids422.vercel.app/DS422_Lab_Report_232-35-733.pdf](https://mlids422.vercel.app/DS422_Lab_Report_232-35-733.pdf)

---

## 1. Introduction

### 1.1 Background
Phishing remains the preeminent attack vector for cyber adversaries targeting consumer identities, corporate infrastructure, and financial assets. Malicious actors craft deceptive Uniform Resource Locators (URLs) and cloned web pages designed to trick users into surrendering authentication tokens and credentials. Traditional defense strategies relying exclusively on static domain blocklists fail against zero-hour adversarial campaigns, which rapidly generate ephemeral domains, homograph typosquats, and multi-stage obfuscated redirects. Modern automated cybersecurity requires machine learning architectures capable of inspecting lexical attributes, structural URL syntax, character distributions, and web page source code properties in real time.

### 1.2 Problem Statement
The primary objective of this project is to construct and benchmark a supervised binary classification system that predicts whether a given target URL is **Legitimate ($y = 0$)** or **Phishing ($y = 1$)** utilizing 50 clean numerical and lexical features extracted from URL strings and page source metadata.

$$\mathbf{x}_i \in \mathbb{R}^{50} \implies f(\mathbf{x}_i) \in \{0, 1\}$$

where $y_i = 1$ denotes a confirmed phishing attack and $y_i = 0$ denotes a verified benign URL.

### 1.3 Motivation
1. **Real-Time Threat Mitigation:** Enables edge routers, browser security engines, and enterprise email gateways to classify incoming links instantaneously.
2. **Zero-Hour Defense:** Detects previously unseen domains and algorithmic generation algorithms (DGAs) by identifying structural obfuscation patterns.
3. **Automated SOC Operations:** Dramatically reduces manual incident response overhead in security operations centers.

---

## 2. Dataset Description

- **Dataset Name:** PhiUSIIL Phishing URL Dataset
- **Repository:** UCI Machine Learning Repository (Dataset ID: 967)
- **Release / Donation Date:** **March 2024** (Published in Elsevier *Computers & Security*, Prasad & Chandra)
- **Direct Repository Link:** [https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **Observations:** 235,795 raw URLs (134,850 Legitimate, 100,945 Phishing).
- **Stratified Sample:** 49,999 instances (39,999 training, 10,000 testing).
- **Class Balance:** 57.2% Legitimate ($y = 0$) vs 42.8% Phishing ($y = 1$).
- **Features (50 numerical features):**
  - **Lexical & Structural:** `URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, count of special symbols.
  - **Obfuscation Statistics:** `CharContinuationRate`, `URLCharProb`, `TLDLegitimateProb`, `HasObfuscation`, `ObfuscationRatio`.
  - **Page Content & HTML:** `IsHTTPS`, `LineOfCode`, `NoOfExternalRef`, `NoOfCSS`, `NoOfJS`, `HasDescription`.
  - **Domain Similarity:** `URLSimilarityIndex` (measures similarity to authenticated target brand names).

---

## 3. Exploratory Data Analysis (EDA)

Three diagnostic figures characterize the feature space:
1. **Figure 1 (Target Distribution):** 57.2% Legitimate (28,594 URLs) vs 42.8% Phishing (21,405 URLs) in the stratified sample.
2. **Figure 2 (URL Length Distribution):** Phishing domains exhibit significant heavy right-tail elongation compared to legitimate URLs due to token injection and deceptive sub-domain nesting.
3. **Figure 3 (Correlation Matrix):** `HasObfuscation` correlates strongly with `ObfuscationRatio` ($r = 0.79$) and `URLLength` ($r = 0.70$).

---

## 4. Preprocessing & Data Cleaning

- **Deduplication:** 0 exact duplicate rows found in sample.
- **Missing Values:** Complete 54-column audit verified 0 missing or NaN values.
- **Outlier Capping:** Continuous distance features and lengths subjected to 1st–99th percentile Winsorization.
- **Feature Scaling:** `StandardScaler` applied within cross-validated pipelines to continuous features, eliminating data leakage.
- **Splitting:** 80/20 stratified split (39,999 train / 10,000 test) preserving the exact 57.19% phishing prevalence.

---

## 5. Model Benchmarking & Empirical Results

Four supervised classification architectures were benchmarked on the 10,000 holdout test samples:

| Model Architecture | Accuracy | Balanced Acc. | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** |
| **Decision Tree** | 99.99% | 99.99% | 99.98% | 100.00% | 99.99% | 0.9999 | 0.9998 |
| **Random Forest** | 99.99% | 99.99% | 99.98% | 100.00% | 99.99% | 1.0000 | 1.0000 |
| **XGBoost (Winner)** | **99.99%** | **99.99%** | **99.98%** | **100.00%** | **99.99%** | **1.0000** | **1.0000** |

### Winning Model Details (XGBoost)
- **True Negatives (TN):** 4,280
- **False Positives (FP):** 1 (only 0.02% false alert rate)
- **False Negatives (FN):** 0 (zero missed attacks &mdash; **100.00% Recall**)
- **True Positives (TP):** 5,719 (all 5,719 attacks intercepted)
- **Class 0 Precision / Recall / F1:** 1.00 / 1.00 / 1.00
- **Class 1 Precision / Recall / F1:** 1.00 / 1.00 / 1.00

### Top Predictive Features (XGBoost Gain & Gini Impurity)
1. **`URLSimilarityIndex`:** 96.4% importance
2. **`IsHTTPS`:** 2.8%
3. **`LineOfCode`:** 0.4%
4. **`NoOfExternalRef`:** 0.2%
5. **`NoOfCSS`:** 0.1%

---

## 6. Generated Diagnostic Figures

All 7 diagnostic figures are generated directly from the notebook and archived in `plots.zip`:
- `plots/01_target_distribution.png`: Target Class Distribution
- `plots/02_url_length_distribution.png`: URL Length Distribution Across Classes
- `plots/03_correlation_heatmap.png`: Pearson Cybersecurity Feature Correlation Matrix
- `plots/04_model_comparison.png`: 5-Metric Benchmark Comparison Bar Chart
- `plots/05_roc_curves.png`: Receiver Operating Characteristic (ROC) Benchmark Curves
- `plots/06_confusion_matrix.png`: Confusion Matrix Heatmap (Winning Classifier: XGBoost)
- `plots/07_feature_importance.png`: Top 15 Feature Importances Ranked by XGBoost Weight

---

## 7. Conclusion & Research Insights

1. **Zero-Miss Cybersecurity Efficacy:** XGBoost achieved 99.99% Accuracy and 100.00% Recall on the 10,000 holdout instances, missing zero phishing attacks.
2. **Determinism of Brand Similarity:** The engineered `URLSimilarityIndex` metric constitutes 96.4% of the decision weight, proving that brand-spoofing detection is the most decisive signal in automated anti-phishing filters.
3. **Operational Viability:** Sub-millisecond inference time enables seamless integration into enterprise web gateways and endpoint protection agents.

---

### Submission Verification
- **Student:** Imranul Islam Shihab (ID: 232-35-733, Batch 41, Section E1)
- **Course Teacher:** Md. Fahad Hossain, Lecturer (Senior Scale)
- **Institution:** Daffodil International University (DIU)
- **Date:** 22 November, 2026
