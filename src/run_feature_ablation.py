"""
Run leave-one-feature-out ablation: for each feature, remove it and compute
10-year average Brier score. Report removals that improve (or don't hurt) the score.

Usage:
  python run_feature_ablation.py           # all features
  python run_feature_ablation.py --quick  # first 12 features only (faster sanity check)
"""
import os
import sys
import pandas as pd
from tournament_model import prep_tournament_data, evaluate_brier_cv

FEATURES = [
    'Diff_GamesPlayed', 'Diff_WinPct', 'Diff_PointsPG', 'Diff_OppPointsPG',
    'Diff_FGM_mean', 'Diff_FGA_mean', 'Diff_FGM3_mean', 'Diff_FGA3_mean',
    'Diff_FTM_mean', 'Diff_FTA_mean', 'Diff_OR_mean', 'Diff_DR_mean',
    'Diff_Ast_mean', 'Diff_TO_mean', 'Diff_Stl_mean', 'Diff_Blk_mean', 'Diff_PF_mean',
    'Diff_OppFGM_mean', 'Diff_OppFGA_mean', 'Diff_OppFGM3_mean', 'Diff_OppFGA3_mean',
    'Diff_OppFTM_mean', 'Diff_OppFTA_mean', 'Diff_OppOR_mean', 'Diff_OppDR_mean',
    'Diff_OppAst_mean', 'Diff_OppTO_mean', 'Diff_OppStl_mean', 'Diff_OppBlk_mean', 'Diff_OppPF_mean',
    'Diff_GamePossessions_mean', 'Diff_OffensiveRating_mean', 'Diff_DefensiveRating_mean',
    'Diff_NetRating_mean', 'Diff_SOS',
    'Diff_Recent_GamesPlayed', 'Diff_Recent_WinPct', 'Diff_Recent_PointsPG', 'Diff_Recent_OppPointsPG',
    'Diff_Recent_FGM_mean', 'Diff_Recent_FGA_mean', 'Diff_Recent_FGM3_mean', 'Diff_Recent_FGA3_mean',
    'Diff_Recent_FTM_mean', 'Diff_Recent_FTA_mean', 'Diff_Recent_OR_mean', 'Diff_Recent_DR_mean',
    'Diff_Recent_Ast_mean', 'Diff_Recent_TO_mean', 'Diff_Recent_Stl_mean', 'Diff_Recent_Blk_mean', 'Diff_Recent_PF_mean',
    'Diff_Recent_OppFGM_mean', 'Diff_Recent_OppFGA_mean', 'Diff_Recent_OppFGM3_mean', 'Diff_Recent_OppFGA3_mean',
    'Diff_Recent_OppFTM_mean', 'Diff_Recent_OppFTA_mean', 'Diff_Recent_OppOR_mean', 'Diff_Recent_OppDR_mean',
    'Diff_Recent_OppAst_mean', 'Diff_Recent_OppTO_mean', 'Diff_Recent_OppStl_mean', 'Diff_Recent_OppBlk_mean', 'Diff_Recent_OppPF_mean',
    'Diff_Recent_GamePossessions_mean', 'Diff_Recent_OffensiveRating_mean', 'Diff_Recent_DefensiveRating_mean', 'Diff_Recent_NetRating_mean',
    'Diff_PredictedCooperRating', 'Diff_SeedNum',
    'IsMen', 'A_PredictedCooperRating', 'B_PredictedCooperRating', 'A_SeedNum', 'B_SeedNum',
]


def main():
    quick = "--quick" in sys.argv
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("Loading data...", flush=True)
    df, data_features = prep_tournament_data(base_dir)
    # Use only features that exist in df
    features = [f for f in FEATURES if f in df.columns]
    if quick:
        features = features[:12]
        print("Quick mode: testing first 12 features only.", flush=True)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing and not quick:
        print(f"Warning: {len(missing)} features not in data: {missing[:5]}...", flush=True)
    print(f"Running ablation over {len(features)} features.\n", flush=True)

    print("Baseline (all features)...", flush=True)
    baseline_brier = evaluate_brier_cv(df, features, verbose=False)
    print(f"Baseline 10-year avg Brier: {baseline_brier:.4f}\n", flush=True)

    results = []
    out_path = os.path.join(base_dir, "feature_ablation_results.csv")
    for i, feat in enumerate(features):
        subset = [f for f in features if f != feat]
        brier = evaluate_brier_cv(df, subset, verbose=False)
        delta = brier - baseline_brier
        results.append({"feature_removed": feat, "brier": brier, "delta": delta})
        status = "IMPROVED" if delta < 0 else ("same" if abs(delta) < 1e-5 else "worse")
        print(f"[{i+1}/{len(features)}] Remove {feat} -> Brier {brier:.4f} (delta {delta:+.4f}) [{status}]", flush=True)
        # Save incrementally every 5 features in case of long run
        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("delta")
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}", flush=True)

    improved = out_df[out_df["delta"] < -1e-5]
    if len(improved) > 0:
        print("\n--- Removals that IMPROVED Brier (consider dropping these) ---", flush=True)
        for _, r in improved.iterrows():
            print(f"  {r['feature_removed']}: Brier {r['brier']:.4f} (delta {r['delta']:+.4f})", flush=True)
    else:
        print("\nNo single-feature removal improved the Brier score.", flush=True)


if __name__ == "__main__":
    main()
