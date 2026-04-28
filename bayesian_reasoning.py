import numpy as np

# ── imports from your teammates' files ──────────────────────────────
from naive_bayes import GaussianNaiveBayes
from kmeans import KMeans

# Decision tree uses functions, not a class, so we import them directly.
# Because decision_tree.py now has the if __name__ == "__main__" guard,
# this import will NOT trigger the 35-minute training run.
from decision_tree import build_tree, predict as dt_predict


# ════════════════════════════════════════════════════════════════════
#  HELPER: extract real probability from Naive Bayes (not just label)
#
#  The GaussianNaiveBayes class only returns 0 or 1.
#  This function digs into its internals to get the actual
#  probability score for class 0 (Confirmed planet).
# ════════════════════════════════════════════════════════════════════
def nb_predict_proba(nb_model, X):
    proba_list = []
    for x in X:
        log_posteriors = []
        for idx in range(len(nb_model._classes)):
            prior = np.log(nb_model._priors[idx])
            pdf_vals = nb_model._pdf(idx, x)
            log_likelihood = np.sum(np.log(pdf_vals + 1e-9))
            log_posteriors.append(prior + log_likelihood)

        log_posteriors = np.array(log_posteriors)
        log_posteriors -= np.max(log_posteriors)   # numerical stability trick
        posteriors = np.exp(log_posteriors)
        posteriors /= posteriors.sum()             # normalize to sum to 1.0

        class_0_idx = np.where(nb_model._classes == 0)[0][0]
        proba_list.append(posteriors[class_0_idx])
    return np.array(proba_list)


# ════════════════════════════════════════════════════════════════════
#  BAYESIAN UPDATE ENGINE
#
#  Applies Bayes theorem once for a single piece of evidence.
#  Call this once per classifier, chaining the output as the next prior.
#
#  P(planet | evidence) = P(evidence | planet) * P(planet) / P(evidence)
# ════════════════════════════════════════════════════════════════════
def bayesian_update(prior, likelihood_given_planet, likelihood_given_fp, evidence_positive):
    """
    Args:
        prior                  : current P(planet) before this evidence
        likelihood_given_planet: P(classifier says planet | signal IS a planet)  = recall
        likelihood_given_fp    : P(classifier says planet | signal is NOT planet) = FPR
        evidence_positive      : True if this classifier voted "planet"

    Returns:
        Updated posterior P(planet | this evidence)
    """
    if evidence_positive:
        p_e_given_planet = likelihood_given_planet
        p_e_given_fp     = likelihood_given_fp
    else:
        # classifier said FP — use the complement probabilities
        p_e_given_planet = 1 - likelihood_given_planet
        p_e_given_fp     = 1 - likelihood_given_fp

    numerator   = p_e_given_planet * prior
    denominator = (p_e_given_planet * prior +
                   p_e_given_fp    * (1 - prior))

    return numerator / denominator if denominator != 0 else prior


# ════════════════════════════════════════════════════════════════════
#  STEP 1: Load all data splits
# ════════════════════════════════════════════════════════════════════
print("Loading data...")
X_train      = np.load("X_train.npy")
y_train      = np.load("y_train.npy")
X_val        = np.load("X_val.npy")        # used to measure honest likelihoods
y_val        = np.load("y_val.npy")
X_test       = np.load("X_test.npy")       # used for final evaluation
y_test       = np.load("y_test.npy")
X_candidates = np.load("X_candidates.npy") # 1,979 unresolved signals


# ════════════════════════════════════════════════════════════════════
#  STEP 2: Calculate the Prior
#
#  P(planet) = fraction of training signals that are genuine planets,
#  before looking at any classifier evidence.
#  After SMOTE balancing this will be exactly 0.500.
# ════════════════════════════════════════════════════════════════════
prior_planet = np.sum(y_train == 0) / len(y_train)
print(f"Prior P(planet)       = {prior_planet:.3f}")
print(f"Prior P(false pos)    = {1 - prior_planet:.3f}")


# ════════════════════════════════════════════════════════════════════
#  STEP 3: Train all three classifiers on the training set
#
#  Decision Tree will take ~15-18 minutes (one training only now,
#  because decision_tree.py has the if __name__ == '__main__' guard).
# ════════════════════════════════════════════════════════════════════
print("\nTraining Naive Bayes...")
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)

print("Training K-Means (k=2)...")
km = KMeans(k=2, max_iters=100)
km.fit(X_train)

print("Training Decision Tree (takes ~15 min, please wait)...")
tree = build_tree(X_train, y_train, max_depth=10)
print("All three classifiers trained.\n")


# ════════════════════════════════════════════════════════════════════
#  STEP 4: Identify which K-Means cluster corresponds to planets
#
#  K-Means assigns labels 0 and 1 randomly — we don't know which
#  one means "planet" until we check against the training labels.
# ════════════════════════════════════════════════════════════════════
train_clusters = km.predict(X_train)
cluster_planet_rate = {}
for cid in [0, 1]:
    mask = (train_clusters == cid)
    rate = np.mean(y_train[mask] == 0)   # fraction that are confirmed planets
    cluster_planet_rate[cid] = rate
    print(f"  Cluster {cid}: {rate:.2%} confirmed planets")

planet_cluster_id = max(cluster_planet_rate, key=cluster_planet_rate.get)
print(f"  → Planet cluster assigned to Cluster {planet_cluster_id}")


# ════════════════════════════════════════════════════════════════════
#  STEP 5: Measure likelihoods on the VALIDATION SET
#
#  We deliberately use the validation set (not training set) here.
#  Measuring on training data gives overfit numbers — the Decision Tree
#  memorises training data perfectly, giving recall=1.0 and FPR=0.0,
#  which would make it completely dominate the Bayesian update and
#  make NB and KM irrelevant.
#
#  The validation set was never seen during training, so it gives
#  honest estimates of how each classifier actually performs.
#
#  recall = P(classifier says planet | signal IS a planet)
#  FPR    = P(classifier says planet | signal is NOT a planet)
# ════════════════════════════════════════════════════════════════════

# --- Naive Bayes ---
nb_val_preds = nb.predict(X_val)
nb_recall    = np.mean(nb_val_preds[y_val == 0] == 0)
nb_fpr       = np.mean(nb_val_preds[y_val == 1] == 0)
print(f"\nNaive Bayes  recall = {nb_recall:.3f}  |  FPR = {nb_fpr:.3f}")

# --- K-Means ---
km_val_labels = km.predict(X_val)
km_recall     = np.mean(km_val_labels[y_val == 0] == planet_cluster_id)
km_fpr        = np.mean(km_val_labels[y_val == 1] == planet_cluster_id)
print(f"K-Means      recall = {km_recall:.3f}  |  FPR = {km_fpr:.3f}")

# --- Decision Tree ---
dt_val_preds = dt_predict(tree, X_val)
dt_recall    = np.mean(dt_val_preds[y_val == 0] == 0)
dt_fpr       = np.mean(dt_val_preds[y_val == 1] == 0)
print(f"Decision Tree recall = {dt_recall:.3f}  |  FPR = {dt_fpr:.3f}")


# ════════════════════════════════════════════════════════════════════
#  STEP 6: Run Bayesian inference on the 1,979 candidates
#
#  For each unresolved signal:
#    1. Start with the prior (0.5)
#    2. Update with NB vote
#    3. Update with KM vote
#    4. Update with DT vote
#    5. Final score = P(planet | all three classifiers)
# ════════════════════════════════════════════════════════════════════
print("\nRunning Bayesian inference on 1,979 candidates...")

nb_candidate_proba  = nb_predict_proba(nb, X_candidates)
km_candidate_labels = km.predict(X_candidates)
dt_candidate_preds  = dt_predict(tree, X_candidates)

final_scores = []
for i in range(len(X_candidates)):
    p = prior_planet

    # Evidence 1: Naive Bayes probability
    p = bayesian_update(p, nb_recall, nb_fpr,
                        evidence_positive=(nb_candidate_proba[i] >= 0.5))

    # Evidence 2: K-Means cluster assignment
    p = bayesian_update(p, km_recall, km_fpr,
                        evidence_positive=(km_candidate_labels[i] == planet_cluster_id))

    # Evidence 3: Decision Tree prediction
    p = bayesian_update(p, dt_recall, dt_fpr,
                        evidence_positive=(dt_candidate_preds[i] == 0))

    final_scores.append(p)

final_scores = np.array(final_scores)
threshold    = 0.5
final_labels = (final_scores >= threshold).astype(int)   # 1 = planet, 0 = false positive

n_planet_pred = np.sum(final_labels == 1)
n_fp_pred     = np.sum(final_labels == 0)

print(f"\n--- Bayesian Candidate Predictions ---")
print(f"Likely CONFIRMED (planet):  {n_planet_pred}")
print(f"Likely FALSE POSITIVE:      {n_fp_pred}")
print(f"Total candidates:           {len(final_labels)}")

sorted_idx = np.argsort(final_scores)[::-1]
print(f"\nTop 10 highest-probability planet candidates:")
print(f"{'Rank':<6} {'Candidate Index':<20} {'P(planet)':<12}")
print("-" * 40)
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"{rank:<6} {idx:<20} {final_scores[idx]:.4f}")


# ════════════════════════════════════════════════════════════════════
#  STEP 7: Evaluate on held-out test set
#
#  Candidates have no ground truth — we can't evaluate there.
#  The test set has known labels, so we run the same pipeline
#  here to get honest performance metrics.
# ════════════════════════════════════════════════════════════════════
print("\nEvaluating on held-out test set...")

nb_test_proba  = nb_predict_proba(nb, X_test)
km_test_labels = km.predict(X_test)
dt_test_preds  = dt_predict(tree, X_test)

test_scores = []
for i in range(len(X_test)):
    p = prior_planet
    p = bayesian_update(p, nb_recall, nb_fpr,
                        evidence_positive=(nb_test_proba[i] >= 0.5))
    p = bayesian_update(p, km_recall, km_fpr,
                        evidence_positive=(km_test_labels[i] == planet_cluster_id))
    p = bayesian_update(p, dt_recall, dt_fpr,
                        evidence_positive=(dt_test_preds[i] == 0))
    test_scores.append(p)

test_scores = np.array(test_scores)
test_preds  = (test_scores >= threshold).astype(int)   # 1 = planet, 0 = false positive

# y_test encoding: 0 = CONFIRMED (planet), 1 = FALSE POSITIVE
planet_pred = (test_preds == 1)
planet_true = (y_test == 0)

TP = np.sum( planet_pred &  planet_true)
FP = np.sum( planet_pred & ~planet_true)
FN = np.sum(~planet_pred &  planet_true)
TN = np.sum(~planet_pred & ~planet_true)

accuracy  = (TP + TN) / len(y_test)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n--- Test Set Performance (all three classifiers combined) ---")
print(f"Accuracy:  {accuracy  * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall    * 100:.2f}%")
print(f"F1-Score:  {f1        * 100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"                 Pred Planet  Pred FP")
print(f"Actual Planet:   [{TP}]        [{FN}]")
print(f"Actual FP:       [{FP}]        [{TN}]")


# ════════════════════════════════════════════════════════════════════
#  STEP 8: Sanity checks — verify the Bayesian logic is correct
#
#  We test all 8 possible vote combinations and verify that
#  the scores follow the expected ordering:
#    - "all 3 say planet" must give the highest score
#    - "all 3 say FP"     must give the lowest score
#    - more classifiers agreeing on planet = higher score
# ════════════════════════════════════════════════════════════════════
print("\n--- SANITY CHECKS ---")

def score_combo(nb_yes, km_yes, dt_yes):
    """Compute Bayesian posterior for a given set of votes."""
    p = prior_planet
    p = bayesian_update(p, nb_recall, nb_fpr, nb_yes)
    p = bayesian_update(p, km_recall, km_fpr, km_yes)
    p = bayesian_update(p, dt_recall, dt_fpr, dt_yes)
    return p

s_all_yes = score_combo(True,  True,  True)
s_nb_dt   = score_combo(True,  False, True)
s_nb_km   = score_combo(True,  True,  False)
s_km_dt   = score_combo(False, True,  True)
s_nb_only = score_combo(True,  False, False)
s_km_only = score_combo(False, True,  False)
s_dt_only = score_combo(False, False, True)
s_all_no  = score_combo(False, False, False)

print(f"All 3 say planet:       {s_all_yes:.4f}  ← should be highest")
print(f"NB + DT say planet:     {s_nb_dt:.4f}")
print(f"NB + KM say planet:     {s_nb_km:.4f}")
print(f"KM + DT say planet:     {s_km_dt:.4f}")
print(f"Only NB says planet:    {s_nb_only:.4f}")
print(f"Only KM says planet:    {s_km_only:.4f}")
print(f"Only DT says planet:    {s_dt_only:.4f}")
print(f"All 3 say FP:           {s_all_no:.4f}  ← should be lowest")

# Verify ordering is logically correct
checks_passed = True
if not (s_all_yes >= s_nb_dt):
    print(f"WARNING: all-yes ({s_all_yes:.4f}) should be >= NB+DT ({s_nb_dt:.4f})")
    checks_passed = False
if not (s_all_yes >= s_nb_km):
    print(f"WARNING: all-yes ({s_all_yes:.4f}) should be >= NB+KM ({s_nb_km:.4f})")
    checks_passed = False
if not (s_all_yes >= s_km_dt):
    print(f"WARNING: all-yes ({s_all_yes:.4f}) should be >= KM+DT ({s_km_dt:.4f})")
    checks_passed = False
if not (s_all_no <= s_nb_only):
    print(f"WARNING: all-no ({s_all_no:.4f}) should be <= NB-only ({s_nb_only:.4f})")
    checks_passed = False
if not (s_all_no <= s_km_only):
    print(f"WARNING: all-no ({s_all_no:.4f}) should be <= KM-only ({s_km_only:.4f})")
    checks_passed = False
if not (s_all_no <= s_dt_only):
    print(f"WARNING: all-no ({s_all_no:.4f}) should be <= DT-only ({s_dt_only:.4f})")
    checks_passed = False
if checks_passed:
    print("All ordering checks passed ✓")

# Show how many candidates fell into each vote combination
nb_votes = (nb_candidate_proba >= 0.5)
km_votes = (km_candidate_labels == planet_cluster_id)
dt_votes = (dt_candidate_preds == 0)

print(f"\nCandidate vote breakdown (all 8 combinations):")
for nv in [True, False]:
    for kv in [True, False]:
        for dv in [True, False]:
            count = np.sum((nb_votes == nv) & (km_votes == kv) & (dt_votes == dv))
            score = score_combo(nv, kv, dv)
            nb_s  = "planet" if nv else "FP    "
            km_s  = "planet" if kv else "FP    "
            dt_s  = "planet" if dv else "FP    "
            label = "CONFIRMED" if score >= threshold else "FALSE POSITIVE"
            print(f"  NB={nb_s} KM={km_s} DT={dt_s} → {count:4d} candidates  "
                  f"score={score:.4f}  label={label}")

# ════════════════════════════════════════════════════════════════════
#  STEP 9: Save outputs for Phase 9 (candidate ranking)
# ════════════════════════════════════════════════════════════════════
np.save("bayesian_candidate_scores.npy", final_scores)
np.save("bayesian_candidate_labels.npy", final_labels)
print("\nSaved: bayesian_candidate_scores.npy")
print("Saved: bayesian_candidate_labels.npy")
print("\nPhase 7 complete.")