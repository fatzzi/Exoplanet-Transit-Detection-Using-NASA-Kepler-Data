# Exoplanet Transit Detection Using NASA Kepler Data
### A Multi-Method AI Approach with Classification, Clustering, and Probabilistic Reasoning

**Eesha Fatima (31647) · Fatima Kaleem (31620) · Uroos Fatima (31094)**  
Institute of Business Administration, Karachi  
Spring 2026 — Introduction to Artificial Intelligence — Dr. Syed Ali Raza

---

## Project Overview

This project builds a multi-method AI pipeline to classify stellar light curves from the NASA Kepler Cumulative KOI (Kepler Objects of Interest) dataset. The goal is to identify which unconfirmed candidate signals are most likely to be genuine exoplanet transits, using Decision Trees, Naive Bayes, K-Means Clustering, and Bayesian Probabilistic Reasoning — all implemented from scratch.

---

## Progress Log

### ✅ Phase 1 — Dataset Acquisition & Understanding

**Dataset:** NASA Kepler Cumulative KOI Table (`cumulative_2026.04.12_06.34.10.csv`)  
**Source:** NASA Exoplanet Archive — `exoplanetarchive.ipac.caltech.edu`

The raw dataset contains approximately **9,564 KOI entries**, each representing a stellar signal flagged by the Kepler pipeline. Each entry carries over 40 engineered photometric and orbital features, including transit depth, orbital period, transit duration, stellar radius, signal-to-noise ratio, and centroid offset metrics.

**Class Distribution (original):**

| Label | Count | Role |
|---|---|---|
| FALSE POSITIVE | 4,839 | Training data (negative class) |
| CONFIRMED | 2,746 | Training data (positive class) |
| CANDIDATE | 1,979 | Inference targets (unknowns) |

**Key insight:** CONFIRMED and FALSE POSITIVE entries are ground truth — NASA has verified these through follow-up observations. The 1,979 CANDIDATE entries have never been confirmed or ruled out, primarily because:
- The Kepler mission ended in 2018 (ran out of fuel)
- Ground-based telescope time is limited and heavily oversubscribed
- Many candidates have weak signals or orbit faint stars, making them low priority for manual follow-up

These 1,979 candidates are the primary scientific target of this pipeline.

---

### ✅ Phase 2 — Data Cleaning (`cleaningdata.py`)

The raw CSV file contains comment lines (prefixed with `#`) and numerous columns that are either metadata, non-informative identifiers, or have excessive missing values.

**Steps performed:**

1. **Loaded raw CSV** using `pd.read_csv(..., comment='#')` to skip header comment rows
2. **Inspected missing values** — identified columns with >50% null rates
3. **Dropped high-missingness columns** — removed any column missing more than 50% of its values using a threshold filter
4. **Dropped non-feature columns** — removed identifiers, provenance fields, and administrative metadata that carry no predictive signal:

```
rowid, kepid, kepoi_name, koi_vet_stat, koi_vet_date,
koi_pdisposition, koi_disp_prov, koi_comment, koi_fittype,
koi_limbdark_mod, koi_parm_prov, koi_tce_delivname,
koi_quarters, koi_trans_mod, koi_datalink_dvr,
koi_datalink_dvs, koi_sparprov, koi_eccen
```

5. **Saved cleaned dataset** as `koi_clean.csv`

**Output shape:** 9,564 rows × 103 columns (102 features + 1 target)

**False positive flags retained** as features:

| Column | Meaning |
|---|---|
| `koi_fpflag_nt` | Not transit-like shape |
| `koi_fpflag_ss` | Secondary eclipse (eclipsing binary) |
| `koi_fpflag_co` | Centroid offset (background contamination) |
| `koi_fpflag_ec` | Ephemeris match to known false positive |

---

### ✅ Phase 3 — Preprocessing (`preprocessing.py`)

**Steps performed:**

#### 1. Train / Candidate Split
CANDIDATE rows were separated from CONFIRMED and FALSE POSITIVE rows **before** any fitting. This ensures no data leakage — candidate statistics never influence imputation or scaling parameters.

```
Training pool  →  7,585 rows  (CONFIRMED + FALSE POSITIVE)
Candidates     →  1,979 rows  (held out entirely)
```

#### 2. Median Imputation
Remaining missing values in numerical features were filled using **median imputation**, fitted exclusively on training data and then applied to candidates.

#### 3. Label Encoding
The target column was encoded numerically:
```
CONFIRMED      →  0
FALSE POSITIVE →  1
```

#### 4. Standard Scaling
All features were scaled using **Z-score standardization** (mean=0, std=1), fitted on training data only, then applied to both training splits and candidates.

#### 5. Train / Validation / Test Split
Stratified split maintaining class proportions across all three sets:

| Split | Rows | Proportion |
|---|---|---|
| Train | 5,309 | 70% |
| Validation | 1,138 | 15% |
| Test | 1,138 | 15% |

#### 6. SMOTE Oversampling
The training set was rebalanced using **SMOTE (Synthetic Minority Over-sampling Technique)** to correct for class imbalance between CONFIRMED and FALSE POSITIVE labels.

```
Before SMOTE:  imbalanced  (FALSE POSITIVE >> CONFIRMED)
After SMOTE:   CONFIRMED = 3,387 | FALSE POSITIVE = 3,387
```

#### 7. Saved Numpy Arrays

All splits saved as `.npy` files for use across model scripts:

```
X_train.npy        y_train.npy
X_val.npy          y_val.npy
X_test.npy         y_test.npy
X_candidates.npy
```

---

## Current Project Structure

```
Exoplanet-Transit-Detection-Using-NASA-Kepler-Data/
│
├── cumulative_2026.04.12_06.34.10.csv   ← Raw NASA dataset
├── koi_clean.csv                         ← Cleaned dataset (Phase 2 output)
│
├── cleaningdata.py                       ← Phase 2: cleaning script
├── preprocessing.py                      ← Phase 3: preprocessing script
│
├── X_train.npy                           ← Balanced training features
├── y_train.npy                           ← Balanced training labels
├── X_val.npy                             ← Validation features
├── y_val.npy                             ← Validation labels
├── X_test.npy                            ← Test features
├── y_test.npy                            ← Test labels
├── X_candidates.npy                      ← 1,979 candidate features (inference)
│
└── README.md                             ← This file
```

---

## Upcoming Phases

| Phase | Task | Status |
|---|---|---|
| Phase 4 | Decision Tree classifier (from scratch, entropy-based) | 🔲 Pending |
| Phase 5 | Naive Bayes classifier (from scratch, Gaussian) | 🔲 Pending |
| Phase 6 | K-Means + Hierarchical Clustering (from scratch) | 🔲 Pending |
| Phase 7 | Bayesian Probabilistic Reasoning module | 🔲 Pending |
| Phase 8 | CNN baseline (Keras) | 🔲 Pending |
| Phase 9 | Candidate ranking & final predictions | 🔲 Pending |
| Phase 10 | GUI for interactive visualization | 🔲 Pending |

---

## Evaluation Metrics (Planned)

All classifiers will be evaluated on the held-out test set using:

- **Accuracy** — overall correctness
- **Precision** — of predicted planets, how many are real
- **Recall** — of real planets, how many did we catch *(priority metric)*
- **F1-Score** — harmonic mean of precision and recall
- **AUC-ROC** — discrimination ability across thresholds
- **Confusion Matrix** — breakdown of TP, FP, TN, FN

> Recall is prioritized because a missed genuine planet (false negative) is more scientifically costly than a false alarm.

---
## 📊 Performance Summary
* **Validation Accuracy:** 95.61%
* **Test Accuracy:** 94.55%
* **AUC-ROC:** 0.9437
* **Key Finding:** The model correctly identifies 94% of confirmed planets (Recall) while maintaining a high precision of 91%.

---

## 🛠️ Logic and Functionality

The classifier is built from scratch using NumPy. Below is an explanation of the core logic:

### 1. Entropy Calculation (`entropy(y)`)
**How it works:** This function measures the "disorder" or "impurity" of a dataset. 
**Math:** $H(X) = -\sum p_i \log_2(p_i)$
If all labels in a group are the same (e.g., all confirmed planets), the entropy is **0**. If there is a 50/50 split, the entropy is **1**.

### 2. Information Gain (`information_gain(...)`)
**How it works:** It measures the reduction in entropy after splitting a dataset based on a specific feature threshold. 
**Logic:**
1. Calculate the entropy of the "parent" node.
2. Split the data into two "child" nodes (Left and Right).
3. Calculate the weighted average entropy of the children.
4. **Gain = Parent Entropy - Weighted Child Entropy.**
The model seeks the split that provides the highest Information Gain.

### 3. Best Split Selection (`best_split(X, y)`)
**How it works:** This is the "brain" of the training process. It iterates through every feature in the dataset and every unique value in those features to find the exact "threshold" that maximizes Information Gain.

### 4. Recursive Tree Building (`build_tree(...)`)
**How it works:** This function builds the tree structure.
* It stops if a node is "pure" (only one class remains).
* It stops if the `max_depth` (set to 10) is reached to prevent overfitting.
* Otherwise, it finds the best split, creates a `Node`, and calls itself recursively to build the left and right sub-branches.

---

## 📂 Data Structure and Nodes
The tree is composed of `Node` objects. Each node contains:
* **Feature/Threshold:** The "question" the node asks (e.g., *Is the transit depth < 0.05?*).
* **Left/Right:** Pointers to the next branches.
* **Value:** Only present in "Leaf" nodes, representing the final classification (0 for Confirmed, 1 for False Positive).

---

## 🔭 Results Analysis

### Confusion Matrix Breakdown
The model was tested on 1,138 unseen samples:
* **True Confirmed:** 386
* **True False Positives:** 690
* **False Negatives (Missed Planets):** 36
* **False Positives (False Alarms):** 26

### Candidate Evaluation
When applied to the 1,979 "Candidate" signals (unsolved by NASA), the model predicted:
* **743** CONFIRMED
* **1236** FALSE POSITIVE
 Data Pre-processing & Feature SelectionOriginally, the model was trained on the full Kepler dataset and achieved a near-perfect accuracy of 99.03%. However, analysis of the decision paths revealed that the tree was primarily using NASA-derived flags:koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_eckoi_score (NASA’s own confidence score)
 
 The Problem: These flags are "cheat codes"—they are results assigned by scientists after they already know the classification. A model using these isn't predicting new planets; it's just memorizing NASA’s notes.The Solution: We manually removed these columns to force the model to learn from raw physical observations, such as:Transit Depth: How much light the planet blocks.Orbital Period: How long it takes to orbit the star.Stellar Radius: The size of the parent star.This dropped the accuracy to 94.55%, but created a much more robust and scientifically honest model capable of identifying planets around new stars where these flags don't yet exist.Updated Functions ListFunctionPurposeLogicentropy(y)Measures ImpurityCalculates the randomness of classes using the $log_2$ formula.information_gain(...)Measures ImprovementSubtracts child entropy from parent entropy to see if a split was "useful."best_split(X, y)OptimizationIterates through every feature and every value to find the maximum possible gain.build_tree(...)Recursive LearningRepeatedly calls best_split to grow the tree until it hits max_depth or purity.predict(...)InferenceTraverses the finished tree for new data to reach a leaf node (0 or 1).📝 Final Evaluation ResultsWith the "leaked" columns removed, the model remains highly effective at distinguishing real signals from noise:Precision (0.91): When the model claims a planet is there, it is correct 91% of the time.Recall (0.94): The model successfully catches 94% of all actual confirmed planets in the test set.Total Test Errors: Only 62 mistakes out of 1,138 unseen samples.

 while Decision Trees are great because they are "explainable" (you can literally see the logic of the branches), they can be "greedy" and overfit, which is why we set max_depth=10

This provides a prioritized list for astronomers to focus their follow-up observations on the 743 high-probability candidates.
*Last updated: April 17, 2026*
