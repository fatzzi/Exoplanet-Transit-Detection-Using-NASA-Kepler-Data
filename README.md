# 🔭 Exoplanet Transit Detection Using NASA Kepler Data
### A Multi-Method AI Approach with Classification, Clustering, and Probabilistic Reasoning

**Eesha Fatima (31647) · Fatima Kaleem (31620) · Uroos Fatima (31094)**
Institute of Business Administration, Karachi
Spring 2026 — Introduction to Artificial Intelligence — Dr. Syed Ali Raza

---

## 📌 Project Overview

This project builds a multi-method AI pipeline to classify stellar light curves from the **NASA Kepler Cumulative KOI (Kepler Objects of Interest)** dataset. The goal is to identify which unconfirmed candidate signals are most likely to be genuine exoplanet transits, using:

- Decision Trees
- Naive Bayes
- K-Means Clustering
- Bayesian Probabilistic Reasoning

All methods are **implemented from scratch** using NumPy.

---
## 📁 Project Structure

```
Exoplanet-Transit-Detection-Using-NASA-Kepler-Data/
│
├── cumulative_2026.04.12_06.34.10.csv   ← Raw NASA dataset
├── koi_clean.csv                         ← Cleaned dataset (Phase 2 output)
│
├── cleaningdata.py                       ← Phase 2: data cleaning script
├── preprocessing.py                      ← Phase 3: preprocessing script
├── decision_tree.py                      ← Phase 4: Decision Tree classifier (from scratch)
│
├── requirements.txt                      ← Required Python libraries
├── .gitignore
├── LICENSE
│
├── X_train.npy                           ← Balanced training features (post-SMOTE)
├── y_train.npy                           ← Balanced training labels
├── X_val.npy                             ← Validation features
├── y_val.npy                             ← Validation labels
├── X_test.npy                            ← Test features
├── y_test.npy                            ← Test labels
├── X_candidates.npy                      ← 1,979 candidate features (inference)
│
└── README.md
```

## 📊 Performance Summary

| Metric | Score |
|---|---|
| Validation Accuracy | 95.61% |
| Test Accuracy | 94.55% |
| AUC-ROC | 0.9437 |
| Precision | 91% |
| Recall | 94% |

> **Key Finding:** The model correctly identifies 94% of confirmed planets (Recall) while maintaining 91% precision.

---



---

## 🚀 Progress Log

### ✅ Phase 1 — Dataset Acquisition & Understanding

**Dataset:** NASA Kepler Cumulative KOI Table (`cumulative_2026.04.12_06.34.10.csv`)
**Source:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)

The raw dataset contains approximately **9,564 KOI entries**, each representing a stellar signal flagged by the Kepler pipeline. Each entry carries over 40 engineered photometric and orbital features including transit depth, orbital period, transit duration, stellar radius, SNR, and centroid offset metrics.

**Class Distribution:**

| Label | Count | Role |
|---|---|---|
| FALSE POSITIVE | 4,839 | Training data (negative class) |
| CONFIRMED | 2,746 | Training data (positive class) |
| CANDIDATE | 1,979 | Inference targets (unknowns) |

> **Key Insight:** The 1,979 CANDIDATE entries have never been confirmed or ruled out — primarily because the Kepler mission ended in 2018, ground-based telescope time is limited, and many candidates have weak signals or orbit faint stars. These are the **primary scientific target** of this pipeline.

---

### ✅ Phase 2 — Data Cleaning (`cleaningdata.py`)

1. Loaded raw CSV using `pd.read_csv(..., comment='#')` to skip header comment rows
2. Inspected missing values — identified columns with >50% null rates
3. Dropped high-missingness columns (>50% null threshold)
4. Dropped non-feature columns (identifiers, provenance fields, administrative metadata):

```
rowid, kepid, kepoi_name, koi_vet_stat, koi_vet_date,
koi_pdisposition, koi_disp_prov, koi_comment, koi_fittype,
koi_limbdark_mod, koi_parm_prov, koi_tce_delivname,
koi_quarters, koi_trans_mod, koi_datalink_dvr,
koi_datalink_dvs, koi_sparprov, koi_eccen
```

5. Saved cleaned dataset as `koi_clean.csv`

**Output shape:** 9,564 rows × 103 columns (102 features + 1 target)

**False positive flags retained as features:**

| Column | Meaning |
|---|---|
| `koi_fpflag_nt` | Not transit-like shape |
| `koi_fpflag_ss` | Secondary eclipse (eclipsing binary) |
| `koi_fpflag_co` | Centroid offset (background contamination) |
| `koi_fpflag_ec` | Ephemeris match to known false positive |

---

### ✅ Phase 3 — Preprocessing (`preprocessing.py`)

#### Train / Candidate Split
CANDIDATE rows were separated **before** any fitting to prevent data leakage.

```
Training pool  →  7,585 rows  (CONFIRMED + FALSE POSITIVE)
Candidates     →  1,979 rows  (held out entirely)
```

#### Steps Performed

| Step | Method |
|---|---|
| Missing values | Median imputation (fit on train only) |
| Label encoding | CONFIRMED → 0, FALSE POSITIVE → 1 |
| Feature scaling | Z-score standardization (fit on train only) |
| Train/Val/Test split | Stratified 70% / 15% / 15% |
| Class rebalancing | SMOTE oversampling |

**After SMOTE:**
```
CONFIRMED = 3,387 | FALSE POSITIVE = 3,387
```

**Saved splits:**
```
X_train.npy  y_train.npy
X_val.npy    y_val.npy
X_test.npy   y_test.npy
X_candidates.npy
```

---

## 🛠️ Model Logic — Decision Tree (Phase 4)

The classifier is built **from scratch using NumPy**.

### Core Functions

#### `entropy(y)`
Measures the disorder/impurity of a dataset.

$$H(X) = -\sum p_i \log_2(p_i)$$

Entropy = **0** when all labels are the same; entropy = **1** at a 50/50 split.

#### `information_gain(...)`
Measures the reduction in entropy after splitting on a feature threshold.

$$\text{Gain} = H(\text{parent}) - \text{Weighted Entropy of Children}$$

#### `best_split(X, y)`
Iterates through every feature and every unique threshold value to find the split that maximizes Information Gain.

#### `build_tree(...)`
Recursively grows the tree. Stops when:
- A node is pure (only one class remains), **or**
- `max_depth = 10` is reached (to prevent overfitting)

#### `predict(...)`
Traverses the finished tree for new data until it reaches a leaf node (0 = Confirmed, 1 = False Positive).

### Node Structure

Each `Node` object contains:
- **Feature / Threshold** — the question the node asks (e.g., *Is transit depth < 0.05?*)
- **Left / Right** — pointers to child branches
- **Value** — only present in leaf nodes; the final classification

---

## ⚠️ Feature Selection — Removing "Cheat Codes"

Initially, the model achieved **99.03% accuracy** — but analysis of decision paths revealed it was primarily using NASA-derived flags:

```
koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec, koi_score
```

These flags are assigned **after** scientists already know the classification. A model using them isn't predicting planets — it's memorising NASA's notes.

**Solution:** These columns were removed, forcing the model to learn from raw physical observations only:

| Feature | Description |
|---|---|
| Transit Depth | How much light the planet blocks |
| Orbital Period | How long it takes to orbit the star |
| Stellar Radius | The size of the parent star |

This dropped accuracy to **94.55%**, but produced a far more robust and scientifically honest model capable of generalising to new stars where these flags don't yet exist.

---

## 📈 Results

### Test Set Performance (1,138 unseen samples)

| Metric | Value |
|---|---|
| Accuracy | 94.55% |
| Precision | 0.91 |
| Recall | 0.94 |
| Total errors | 62 / 1,138 |

### Confusion Matrix

| | Predicted Confirmed | Predicted False Positive |
|---|---|---|
| **Actual Confirmed** | 386 ✅ | 36 ❌ |
| **Actual False Positive** | 26 ❌ | 690 ✅ |

> **Recall is the priority metric** — a missed genuine planet (false negative) is more scientifically costly than a false alarm.

### Candidate Predictions (1,979 unresolved signals)

| Prediction | Count |
|---|---|
| 🟢 CONFIRMED (likely planet) | 743 |
| 🔴 FALSE POSITIVE | 1,236 |

This provides a prioritised list of **743 high-probability candidates** for astronomers to focus follow-up observations on.

---

## 🗺️ Upcoming Phases

| Phase | Task | Status |
|---|---|---|
| Phase 5 | Naive Bayes classifier (Gaussian, from scratch) | 🔲 Pending |
| Phase 6 | K-Means + Hierarchical Clustering (from scratch) | 🔲 Pending |
| Phase 7 | Bayesian Probabilistic Reasoning module | 🔲 Pending |
| Phase 8 | CNN baseline (Keras) | 🔲 Pending |
| Phase 9 | Candidate ranking & final predictions | 🔲 Pending |
| Phase 10 | GUI for interactive visualization | 🔲 Pending |

---

## 📐 Evaluation Metrics

All classifiers are evaluated on the held-out test set using:

- **Accuracy** — overall correctness
- **Precision** — of predicted planets, how many are real
- **Recall** — of real planets, how many did we catch *(priority metric)*
- **F1-Score** — harmonic mean of precision and recall
- **AUC-ROC** — discrimination ability across thresholds
- **Confusion Matrix** — breakdown of TP, FP, TN, FN

---

*Last updated: April 17, 2026*
