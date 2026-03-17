"""
Iteratively remove the top k features from ablation (best-to-remove first) and evaluate
10-year Brier each time. Find the k that minimizes Brier, with at least 20 features remaining.

Reads: feature_ablation_results.csv (from run_feature_ablation.py)
Writes: iterative_removal_results.csv, feature_exclusion_best.txt (used by tournament_model)
"""
import os
import sys
import pandas as pd
from tournament_model import prep_tournament_data, evaluate_brier_cv

MIN_FEATURES_REMAINING = 20


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ablation_path = os.path.join(base_dir, 'feature_ablation_results.csv')
    if not os.path.isfile(ablation_path):
        print(f"Missing {ablation_path}. Run run_feature_ablation.py first.", flush=True)
        sys.exit(1)

    ablation = pd.read_csv(ablation_path)
    # Sort by delta ascending: most beneficial to remove first (most negative delta)
    ablation = ablation.sort_values('delta', ascending=True).reset_index(drop=True)
    removal_order = ablation['feature_removed'].tolist()
    n_total = len(removal_order)
    k_max = n_total - MIN_FEATURES_REMAINING
    if k_max < 1:
        print("Not enough features to remove (need at least 20 remaining).", flush=True)
        sys.exit(1)

    print("Loading data (full feature set)...", flush=True)
    df, full_features = prep_tournament_data(base_dir, features_exclude=set())
    # Only consider features that exist in data and in ablation order
    removal_order = [f for f in removal_order if f in full_features]
    n_total = len(removal_order)
    k_max = min(k_max, n_total - MIN_FEATURES_REMAINING)
    if k_max < 1:
        print("After filtering, not enough features to remove.", flush=True)
        sys.exit(1)

    print(f"Iteratively removing top k=1..{k_max} (min {MIN_FEATURES_REMAINING} features remaining)...\n", flush=True)

    out_path = os.path.join(base_dir, "iterative_removal_results.csv")
    exclusion_path = os.path.join(base_dir, "feature_exclusion_best.txt")
    results = []
    best_brier = float('inf')
    best_k = 0

    for k in range(1, k_max + 1):
        exclude_k = set(removal_order[:k])
        features_k = [f for f in full_features if f not in exclude_k]
        n_remaining = len(features_k)
        brier = evaluate_brier_cv(df, features_k, verbose=False)
        results.append({"k_removed": k, "brier": brier, "n_features_remaining": n_remaining})
        print(f"k={k} (remove top {k}, {n_remaining} left) -> Brier {brier:.4f}", flush=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        if brier < best_brier:
            best_brier = brier
            best_k = k
            with open(exclusion_path, "w") as f:
                for feat in removal_order[:best_k]:
                    f.write(feat + "\n")
            print(f"  -> new best (k={best_k}, Brier={best_brier:.4f})", flush=True)

    out_df = pd.DataFrame(results)
    best_n_remaining = int(out_df.loc[out_df["brier"].idxmin(), "n_features_remaining"])
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Best: k={best_k} -> Brier {best_brier:.4f} ({best_n_remaining} features remaining)", flush=True)
    print(f"Exclusion list written to {exclusion_path}. tournament_model will use it for training and submission.", flush=True)
    return best_k, best_brier


if __name__ == "__main__":
    main()