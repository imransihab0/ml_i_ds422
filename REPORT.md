# DS422: Machine Learning Driven Data Analysis Lab I (DS Major) &mdash; Lab Task Report
**Daffodil International University (DIU)**  
**Department of Data Science**

---

### Academic Metadata
- **Project Title:** Machine Learning Driven Forest Canopy Cover Type Prediction Using Cartographic & Remote Sensing Variables
- **Dataset:** UCI Machine Learning Repository &mdash; Covertype Dataset (ID: 31)
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
Accurate forest canopy cover type mapping is fundamental for ecological conservation, natural resource management, biodiversity monitoring, and wildfire danger modeling. In subalpine and montane forest ecosystems, identifying specific tree canopy species across thousands of square kilometers of mountainous terrain presents severe logistical challenges. Ground-based forestry inventories are prohibitively expensive, time-consuming, hazardous, and limited to accessible areas. Remote sensing, combined with geographic information systems (GIS) and modern machine learning algorithms, offers an automated, scalable alternative to infer forest species distributions directly from digital elevation models (DEM), hydrological flow networks, solar radiation dynamics, and geological soil surveys.

The Roosevelt National Forest in northern Colorado encompasses a complex mountainous biome characterized by dramatic elevation gradients, steep topography, and diverse soil regimes. High-elevation stands are predominantly dominated by *Spruce/Fir* (Engelmann Spruce and Subalpine Fir), which provide critical watershed protection and wildlife habitat.

### 1.2 Problem Statement
The primary objective of this project is to construct and benchmark a supervised binary classification system that predicts the presence of high-altitude **Spruce/Fir** forest cover versus other montane forest tree species (including Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, and Krummholz) strictly utilizing 54 cartographic and topographical environmental variables without any remote-sensed optical imagery or manual field intervention.

$$\mathbf{x}_i \in \mathbb{R}^{54} \implies f(\mathbf{x}_i) \in \{0, 1\}$$

where $y_i = 1$ denotes Spruce/Fir and $y_i = 0$ denotes alternative forest covers.

### 1.3 Motivation
1. **Scalable Ecological Inventory:** Eliminates continuous manual surveying, allowing forest service agencies to update canopy maps instantaneously as digital terrain models update.
2. **Wildfire Risk Stratification:** Different forest canopies carry vastly different fuel loads, moisture retention capacities, and crown fire potentials.
3. **Climate Change Adaptation:** High-altitude spruce-fir biomes are among the most vulnerable to warming temperatures and bark beetle infestations; rapid detection of shifts in their geographic boundaries provides early-warning bioindicators.

---

## 2. Dataset Description

- **Source:** UCI Machine Learning Repository (Donated by Jock A. Blackard & Denis J. Dean, Colorado State University & US Forest Service Region 2).
- **UCI Dataset ID:** 31
- **Direct Repository Link:** [https://archive.ics.uci.edu/dataset/31/covertype](https://archive.ics.uci.edu/dataset/31/covertype)
- **Observations:** 581,012 raw 30m $	imes$ 30m cartographic cells.
- **Stratified Sample:** 50,000 instances (40,000 training, 10,000 testing).
- **Target Distribution:** 36.80% positive ($y = 1$, Spruce/Fir) vs 63.20% negative ($y = 0$, Other Forest Species).
- **Attributes (54 features):**
  - 10 continuous topographical variables: Elevation, Aspect, Slope, Horizontal/Vertical Distance to Hydrology, Horizontal Distance to Roadways, Hillshade at 9am, Noon, 3pm, Horizontal Distance to Fire Points.
  - 4 binary Wilderness Areas: Rawah (Area 1), Neota (Area 2), Comanche Peak (Area 3), Cache la Poudre (Area 4).
  - 40 binary USDA Geological Soil Types.

---

## 3. Exploratory Data Analysis (EDA)

Three diagnostic figures characterize the feature space:
1. **Figure 1 (Target Distribution):** 36.80% Spruce/Fir vs 63.20% Other species, demonstrating realistic ecological class distribution without synthetic distortion.
2. **Figure 2 (Elevation Distribution):** Clear vertical zonation. High-altitude Spruce/Fir stands have a median elevation exceeding 3,150m, while competing species concentrate between 2,500m and 2,900m.
3. **Figure 3 (Correlation Matrix):** Strong physical relationships: Hillshade 9am vs 3pm ($r = -0.78$) capturing diurnal solar shifts, and Horizontal vs Vertical Distance to Hydrology ($r = 0.61$).

---

## 4. Preprocessing & Data Cleaning

- **Deduplication:** 0 exact duplicate rows found in sample.
- **Missing Values:** Complete 54-column audit verified 0 missing or NaN values.
- **Outlier Capping:** Continuous distance features (Hydrology, Roadways, Fire Points) subjected to 1st–99th percentile Winsorization.
- **Feature Scaling:** `StandardScaler` applied within cross-validated pipelines to continuous features, eliminating data leakage.
- **Splitting:** 80/20 stratified split (40,000 train / 10,000 test) preserving the 36.80% positive prevalence.

---

## 5. Model Benchmarking & Empirical Results

Four supervised classification architectures were benchmarked on the 10,000 holdout test samples:

| Model Architecture | Accuracy | Balanced Acc. | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 76.31% | 77.30% | 64.08% | 81.06% | 71.58% | 0.8441 | 0.7237 |
| **Decision Tree** | 80.37% | 80.76% | 69.80% | 82.23% | 75.51% | 0.8782 | 0.7855 |
| **Random Forest** | 82.34% | 83.01% | 71.84% | 85.54% | 78.09% | 0.9176 | 0.8725 |
| **XGBoost (Winner)** | **84.69%** | **85.36%** | **74.88%** | **87.88%** | **80.86%** | **0.9351** | **0.8955** |

### Winning Model Details (XGBoost)
- **True Negatives (TN):** 5,235
- **False Positives (FP):** 1,085
- **False Negatives (FN):** 446 (only 12.1% miss rate)
- **True Positives (TP):** 3,234 (87.88% sensitivity)
- **Class 0 Precision / Recall / F1:** 0.92 / 0.83 / 0.87
- **Class 1 Precision / Recall / F1:** 0.75 / 0.88 / 0.81

### Top Predictive Features (Random Forest Gini Impurity & XGBoost Gain)
1. **Elevation:** ~24.5% importance
2. **Horizontal Distance to Roadways:** ~12.8%
3. **Horizontal Distance to Fire Points:** ~9.2%
4. **Horizontal Distance to Hydrology:** ~6.4%
5. **Vertical Distance to Hydrology:** ~5.1%

---

## 6. Generated Diagnostic Figures

All 7 diagnostic figures are generated directly from the notebook and archived in `plots.zip`:
- `plots/01_target_distribution.png`: Target Class Distribution
- `plots/02_elevation_distribution.png`: Elevation Distribution Across Forest Classes
- `plots/03_correlation_heatmap.png`: Pearson Topographical Correlation Matrix
- `plots/04_model_comparison.png`: 5-Metric Benchmark Comparison Bar Chart
- `plots/05_roc_curves.png`: Receiver Operating Characteristic (ROC) Benchmark Curves
- `plots/06_confusion_matrix.png`: Confusion Matrix Heatmap (Winning Classifier: XGBoost)
- `plots/07_feature_importance.png`: Top 15 Feature Importances Ranked by Gini Weight

---

## 7. Conclusion & Research Insights

1. **Gradient Boosting Supremacy:** XGBoost achieved the best performance across all seven statistical criteria, with an accuracy of 84.69% and an ROC-AUC of 0.9351.
2. **Non-linear Ensemble Advantage:** Tree ensembles outperformed linear models by over 8 percentage points, showing that complex terrain interactions require non-linear decision boundaries.
3. **Elevation Dominance:** Topographical elevation accounts for roughly one-quarter of total decision weight, reflecting the strong thermal and microclimatic zonation of the Rocky Mountains.
4. **Operational Deployment:** The low false negative rate (only 446 missed out of 3,680 positive instances) ensures that operational conservation teams can delineate critical spruce-fir habitats with high sensitivity.

---

### Submission Verification
- **Student:** Imranul Islam Shihab (ID: 232-35-733, Batch 41, Section E1)
- **Course Teacher:** Md. Fahad Hossain, Lecturer (Senior Scale)
- **Department:** Department of Data Science, Daffodil International University (DIU)
- **Date:** 22 November, 2026
