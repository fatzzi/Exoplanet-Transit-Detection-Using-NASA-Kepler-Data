import numpy as np
import os

from naive_bayes import GaussianNaiveBayes
from kmeans import KMeans
from decision_tree import build_tree, predict as dt_predict


print("Loading data.")
X_train      = np.load("X_train.npy")
y_train      = np.load("y_train.npy")
X_val        = np.load("X_val.npy")
y_val        = np.load("y_val.npy")
X_candidates = np.load("X_candidates.npy")

n_candidates = len(X_candidates)
print(f"Candidates to rank: {n_candidates}")

bayesian_scores = np.load("bayesian_candidate_scores.npy")   # shape (1979,), range 0–1
print(f"Loaded Bayesian scores. Range: {bayesian_scores.min():.4f} – {bayesian_scores.max():.4f}")



cnn_available = os.path.exists("cnn_candidate_scores.npy")

if cnn_available:
    cnn_scores = np.load("cnn_candidate_scores.npy")         
    print(f"Loaded CNN scores.     Range: {cnn_scores.min():.4f} – {cnn_scores.max():.4f}")
else:
    print("WARNING: cnn_candidate_scores.npy not found.")
    print("         Run cnn_baseline.py first, then re-run this script.")
    print("         Continuing with Bayesian scores only for now.\n")


#  Re-train the three from-scratch classifiers and get per-candidate labels (0=planet, 1=false positive)

print("\nTraining Naive Bayes.")
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)
nb_labels = nb.predict(X_candidates)                  

print("Training K-Means (k=2).")
km = KMeans(k=2, max_iters=100)
km.fit(X_train)

# Identifing which cluster = planet cluster using validation labels
train_clusters = km.predict(X_train)
cluster_planet_rate = {
    cid: np.mean(y_train[train_clusters == cid] == 0) for cid in [0, 1]
}
planet_cluster_id = max(cluster_planet_rate, key=cluster_planet_rate.get)
km_raw = km.predict(X_candidates)
km_labels = (km_raw != planet_cluster_id).astype(int)         

print("Training Decision Tree (this takes arnd 15 min, please wait)")
tree = build_tree(X_train, y_train, max_depth=10)
dt_labels = dt_predict(tree, X_candidates)                    
print("All classifiers ready.\n")


# computing ensemble score through weighted average of probability scores.

dt_scores = (1 - dt_labels).astype(float)    
# dt_labels: 0=CONFIRMED(planet), 1=FP  -> planet prob = 1 - label
dt_planet_scores = (1 - dt_labels).astype(float)
nb_planet_scores = (1 - nb_labels).astype(float)
km_planet_scores = (1 - km_labels).astype(float)

if cnn_available:
    # Full ensemble: all five signals
    ensemble_scores = (
        0.40 * cnn_scores          +
        0.40 * bayesian_scores     +
        0.12 * dt_planet_scores    +
        0.05 * nb_planet_scores    +
        0.03 * km_planet_scores
    )
    print("Ensemble method: CNN (40%) + Bayesian (40%) + DT (12%) + NB (5%) + KM (3%)")
else:
   
    ensemble_scores = (
        0.60 * bayesian_scores     +
        0.25 * dt_planet_scores    +
        0.10 * nb_planet_scores    +
        0.05 * km_planet_scores
    )
    print("Ensemble method: Bayesian (60%) + DT (25%) + NB (10%) + KM (5%) [CNN missing]")


# Counting how many classifiers agree on each candidate

dt_vote  = (dt_labels == 0).astype(int)
nb_vote  = (nb_labels == 0).astype(int)
km_vote  = (km_labels == 0).astype(int)
bay_vote = (bayesian_scores >= 0.5).astype(int)

if cnn_available:
    cnn_vote      = (cnn_scores >= 0.5).astype(int)
    agreement     = dt_vote + nb_vote + km_vote + bay_vote + cnn_vote
    max_agreement = 5
else:
    agreement     = dt_vote + nb_vote + km_vote + bay_vote
    max_agreement = 4


# Assigning confidence tiers

tiers = np.where(ensemble_scores >= 0.80, "HIGH",
        np.where(ensemble_scores >= 0.50, "MEDIUM", "LOW"))

final_labels = (ensemble_scores >= 0.50).astype(int)  # 1=planet, 0=FP

n_high   = np.sum(tiers == "HIGH")
n_medium = np.sum(tiers == "MEDIUM")
n_low    = np.sum(tiers == "LOW")

print(f"\n--- Final Candidate Classification ---")
print(f"HIGH confidence planets (score ≥ 0.80):    {n_high}")
print(f"MEDIUM confidence planets (0.50–0.79):     {n_medium}")
print(f"LOW / False Positives (score < 0.50):      {n_low}")
print(f"Total:                                      {n_candidates}")


#cRanked list 
sorted_idx = np.argsort(ensemble_scores)[::-1]   # highest score first

print(f"\n{'─'*75}")
print(f"{'Rank':<6} {'Cand. Index':<14} {'Ensemble Score':<17} {'Tier':<10} {'Classifiers Agree'}")
print(f"{'─'*75}")
for rank, idx in enumerate(sorted_idx[:20], 1):
    score   = ensemble_scores[idx]
    tier    = tiers[idx]
    agree   = agreement[idx]
    print(f"{rank:<6} {idx:<14} {score:<17.4f} {tier:<10} {agree}/{max_agreement}")

print(f"{'─'*75}")
print(f"(Showing top 20 of {n_candidates} candidates)")


# Per-classifier breakdown for the top 10

print(f"\n--- Top 10 candidates — per-classifier breakdown")
header = f"{'Rank':<5} {'Idx':<6} {'Score':<8} {'Bay':>5} {'DT':>5} {'NB':>5} {'KM':>5}"
if cnn_available:
    header += f" {'CNN':>5}"
header += f"  {'Tier'}"
print(header)
print("─" * (len(header) + 2))

for rank, idx in enumerate(sorted_idx[:10], 1):
    score = ensemble_scores[idx]
    tier  = tiers[idx]
    bay_s = f"{bayesian_scores[idx]:.3f}"
    dt_s  = "✓" if dt_vote[idx] else "✗"
    nb_s  = "✓" if nb_vote[idx] else "✗"
    km_s  = "✓" if km_vote[idx] else "✗"
    row   = f"{rank:<5} {idx:<6} {score:<8.4f} {bay_s:>5} {dt_s:>5} {nb_s:>5} {km_s:>5}"
    if cnn_available:
        cnn_s = f"{cnn_scores[idx]:.3f}"
        row  += f" {cnn_s:>5}"
    row += f"  {tier}"
    print(row)


np.save("final_candidate_scores.npy",  ensemble_scores)
np.save("final_candidate_labels.npy",  final_labels)
np.save("final_candidate_tiers.npy",   tiers)
np.save("final_candidate_ranking.npy", sorted_idx)
np.save("final_agreement_counts.npy",  agreement)

# text file
with open("final_candidate_summary.txt", "w", encoding="utf-8") as f:
    f.write("EXOPLANET CANDIDATE RANKING — PHASE 9 FINAL OUTPUT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total candidates evaluated: {n_candidates}\n")
    f.write(f"HIGH confidence (score ≥ 0.80): {n_high}\n")
    f.write(f"MEDIUM confidence (0.50–0.79):  {n_medium}\n")
    f.write(f"LOW / False Positives:          {n_low}\n\n")
    f.write(f"{'Rank':<6} {'Cand. Index':<14} {'Score':<12} {'Tier':<10} {'Agreement'}\n")
    f.write("-" * 55 + "\n")
    for rank, idx in enumerate(sorted_idx, 1):
        f.write(f"{rank:<6} {idx:<14} {ensemble_scores[idx]:<12.4f} "
                f"{tiers[idx]:<10} {agreement[idx]}/{max_agreement}\n")

print("\nSaved outputs:")
print("  final_candidate_scores.npy   ← ensemble probability per candidate")
print("  final_candidate_labels.npy   ← 1=planet, 0=false positive")
print("  final_candidate_tiers.npy    ← HIGH / MEDIUM / LOW")
print("  final_candidate_ranking.npy  ← indices sorted best→worst")
print("  final_agreement_counts.npy   ← how many classifiers agreed")
print("  final_candidate_summary.txt  ← human-readable ranked list")
print("\nPhase 9 complete.")
