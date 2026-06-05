# HEARS Model — Hearing Exposure AI Risk Score

> **Development, Internal Validation, and Risk Stratification of the HEARS Model  
> for Noise-Induced Hearing Loss (NIHL) among Malaysian Workers**

---

## Study Information

| Item | Detail |
|---|---|
| **Author** | Dr. Syuaib Aiman Amir Kamarudin |
| **Degree** | DrPH — Universiti Sains Malaysia |
| **Department** | Community Medicine, School of Medical Sciences, USM |
| **Supervisor** | Dr Afiq Izzudin A. Rahim
                   Prof. Dr. Aziah Daud
                   Dr Shawalludin Husin|
| **JEPeM Code** | USM/JEPeM/KK/26010129 |
| **Ethics approval** | Granted 6th May 2026 — valid until 5th May 2027 |
| **Data source** | MySMART-OH (OH Digital Solution Sdn Bhd + DOSH Malaysia) |

---

## Key Results

| Metric | Value |
|---|---|
| Best algorithm | **LightGBM** |
| Macro-averaged AUC | **0.844** |
| Calibration slope | **1.001** (ideal = 1.0) |
| Brier Score (High risk) | **0.116** |
| F1-macro | **0.655** |
| Overall accuracy | **65.9%** |
| AUC — Low risk | 0.837 |
| AUC — Moderate risk | 0.789 |
| AUC — High risk | 0.905 |
| DeLong test (LightGBM vs LR) | z = 8.42, p < 0.001 |
| Kappa (simulated OHD) | 0.880 (Phase 2 pending) |

---

## Risk Stratification Output

| Risk Category | n (Test set) | % | Action required |
|---|---|---|---|
| Low risk | 1,007 | 36.0% | Routine audiometric surveillance |
| Moderate risk | 767 | 27.4% | Enhanced monitoring + prevention |
| **High risk** | **1,026** | **36.6%** | **Urgent OHD intervention** |

---

## SHAP Feature Importance

| Rank (Gini) | Rank (SHAP, High risk) | Predictor | SHAP |value| (High) |
|---|---|---|---|
| 1 | 6 | Age | 0.085 |
| 2 | **1** | Noise LEX | **2.796** |
| 3 | 2 | Noise Lmax | 0.312 |
| 4 | 7 | Noise Lpeak | 0.059 |
| 5 | 3 | HPD type | 0.145 |

> **Key finding:** LEX is SHAP rank 1 for High risk (mean |SHAP| = 2.796),  
> nearly 9× larger than the second predictor. Age is Gini rank 1 but  
> SHAP rank 6 — LEX is the primary individual-level High risk driver.

---

## Repository Structure

```
HEARS-model/
├── HEARS_Model_Development.Rmd    # Complete R Markdown — run this
├── HEARS_Model_Development.md     # GitHub-rendered output
├── HEARS_Final_Clean_v3.xlsx      # Dataset (place in working directory)
├── HEARS_LightGBM_equivalent_final.model  # Saved model (generated on run)
├── HEARS_imputation_values.rds    # Imputation statistics (generated on run)
├── HEARS_scaler.rds               # Standardisation scaler (generated on run)
├── HEARS_algorithm_comparison.csv # Table 4.2 results (generated on run)
└── README.md                      # This file
```

---

## How to Reproduce

### Requirements

- R >= 4.3.0
- RStudio (recommended)
- Packages: `tidyverse`, `readxl`, `caret`, `pROC`, `irr`, `randomForest`,
  `xgboost`, `lightgbm`, `e1071`, `nnet`, `ggplot2`, `knitr`, `kableExtra`

### Steps

```r
# 1. Clone this repository
#    git clone https://github.com/YOUR_USERNAME/HEARS-model.git
#    cd HEARS-model

# 2. Place HEARS_Final_Clean_v3.xlsx in the working directory

# 3. Open RStudio and set working directory
setwd("path/to/HEARS-model")

# 4. Knit the R Markdown file
rmarkdown::render("HEARS_Model_Development.Rmd",
                  output_format = "html_document",
                  output_file   = "HEARS_Model_Development.html")

# 5. For GitHub Markdown output
rmarkdown::render("HEARS_Model_Development.Rmd",
                  output_format = "md_document",
                  output_file   = "HEARS_Model_Development.md")
```

### Push to GitHub

```bash
git init
git add HEARS_Model_Development.Rmd README.md
git commit -m "HEARS model development — LightGBM AUC 0.844"
git remote add origin https://github.com/YOUR_USERNAME/HEARS-model.git
git branch -M main
git push -u origin main
```

---

## Analytical Pipeline

```
Step 1  Load dataset (n = 9,530)
Step 2  Three-class risk labelling → n = 9,331
Step 3  Encode 12 variables + median/mode imputation
Step 4  Stratified 70/30 train-test split (seed = 42)
Step 5  5-fold cross-validation setup
Steps 6–11  Train 6 algorithms (LR, SVM, ANN, RF, XGBoost, LightGBM)
Step 12 Evaluate all on test set — Table 4.2
Step 13 Best model (LightGBM) detailed evaluation — Table 4.3
Step 14 Risk proportions — Table 4.4
Step 15 Gini + SHAP feature importance — Table 4.5
Step 16 ROC curves — Figure 4.1
Step 17 Calibration plot — Figure 4.1(e)
Step 18 Kappa agreement (simulated) — Table 4.6
Step 19 Hypothesis testing (H1, H2, H3)
Step 20 Save model and outputs
```

---

## Primary Analysis Platform

> All key results reported in this README and in the thesis Chapter 4 were
> produced using **Python** (scikit-learn v1.3, LightGBM v4.0, SHAP v0.46).
> The R Markdown file (`HEARS_Model_Development.Rmd`) demonstrates the
> analytical pipeline for reproducibility verification. Minor numerical
> differences between R and Python outputs are expected and normal in
> cross-platform machine learning research.

---

## Important Notes

1. **Kappa values in Table 4.6 are simulated** — Phase 2 external validation  
   with actual OHD clinical assessments is pending. See Section 3.11.10.

2. **SHAP analysis** was conducted in Python (shap v0.46, TreeExplainer)  
   on n = 500 test workers. SHAP values in the Rmd are the Python-derived values  
   (hardcoded for consistency). See thesis Section 3.11.11.

3. **Random seed = 42** is fixed throughout for full reproducibility.

4. **LightGBM in R:** lgb.train() v4.6.0 on Apple Silicon Macs has a known  
   multiclass training issue. The Rmd uses xgb.train() as an equivalent  
   gradient boosting alternative. Python LightGBM results are definitive.

4. **External validation (Phase 2)** will use MySMART-OH records  
   January–April 2026 with 3 blinded OHDs for Fleiss' and Cohen's Kappa.

---

## Citation

```
Amir Kamarudin, S.A. (2026) Development, Validation, and Risk Stratification
of the Hearing Exposure AI Risk Score (HEARS) Model for Noise-Induced Hearing
Loss (NIHL) among Malaysian Workers. DrPH Thesis, Universiti Sains Malaysia.
JEPeM Code: USM/JEPeM/KK/26010129.
```

---

## Ethics Statement

This study was approved by the Universiti Sains Malaysia Human Research Ethics
Committee (JEPeM-USM), Protocol Code: **USM/JEPeM/KK/26010129**, approved 6th May 2026,
valid until 5th May 2027. Data access was granted by OH Digital Solution Sdn Bhd
and the Department of Occupational Safety and Health (DOSH) Malaysia.

---

*Department of Community Medicine · School of Medical Sciences · Universiti Sains Malaysia*
