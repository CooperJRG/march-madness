import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def prep_tournament_data(base_dir):
    """
    Creates historical pairwise matchups for the NCAA Tournament data.
    """
    print("Loading tournament results...")
    m_tourney_results = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MNCAATourneyCompactResults.csv'))
    w_tourney_results = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WNCAATourneyCompactResults.csv'))
    tourney_results = pd.concat([m_tourney_results, w_tourney_results], ignore_index=True)
    
    print("Loading regular season aggregated stats and predicted Cooper ratings...")
    m_stats_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MRegularSeasonAggregatedStats.csv'))
    w_stats_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WRegularSeasonAggregatedStats.csv'))
    stats_df = pd.concat([m_stats_df, w_stats_df], ignore_index=True)
    
    m_cooper_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MGeneratedCooperRatings.csv'))
    w_cooper_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WGeneratedCooperRatings.csv'))
    cooper_df = pd.concat([m_cooper_df, w_cooper_df], ignore_index=True)
    
    print("Loading seeds...")
    m_seeds = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MNCAATourneySeeds.csv'))
    w_seeds = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WNCAATourneySeeds.csv'))
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
    
    # Extract just the integer value of the seed (e.g., 'W01' -> 1)
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    
    # Merge Stats and Cooper ratings
    team_season_features = pd.merge(stats_df, cooper_df[['Season', 'TeamID', 'PredictedCooperRating']], on=['Season', 'TeamID'], how='left')
    # Merge Seeds
    team_season_features = pd.merge(team_season_features, seeds[['Season', 'TeamID', 'SeedNum']], on=['Season', 'TeamID'], how='left')
    
    # Optional Fill Missing Seeds with 17 (worst possible + 1)
    team_season_features['SeedNum'] = team_season_features['SeedNum'].fillna(17)
    
    # Create pairwise matchups for training
    # For each game, create two symmetric rows:
    # 1. Team A = Winner, Team B = Loser (Target = 1)
    # 2. Team A = Loser, Team B = Winner (Target = 0)
    print("Building pairwise matchups...")
    
    # Row 1 (A=W, B=L)
    win_row = tourney_results[['Season', 'WTeamID', 'LTeamID']].copy()
    win_row.columns = ['Season', 'TeamA', 'TeamB']
    win_row['Target'] = 1
    
    # Row 2 (A=L, B=W)
    lose_row = tourney_results[['Season', 'LTeamID', 'WTeamID']].copy()
    lose_row.columns = ['Season', 'TeamA', 'TeamB']
    lose_row['Target'] = 0
    
    matchups = pd.concat([win_row, lose_row], ignore_index=True)
    
    # Join features for Team A
    matchups = pd.merge(matchups, team_season_features, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], how='left')
    
    # Rename Team A features
    a_cols = {col: f'A_{col}' for col in team_season_features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TeamName_Silver']}
    matchups.rename(columns=a_cols, inplace=True)
    
    # Join features for Team B
    matchups = pd.merge(matchups, team_season_features, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], how='left')
    
    # Rename Team B features
    b_cols = {col: f'B_{col}' for col in team_season_features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TeamName_Silver', 'Gender']}
    matchups.rename(columns=b_cols, inplace=True)
    
    # Drop extra ids
    matchups.drop(['TeamID_x', 'TeamName_x', 'TeamName_Silver_x', 'Gender_x', 'TeamID_y', 'TeamName_y', 'TeamName_Silver_y', 'Gender_y'], axis=1, errors='ignore', inplace=True)
    
    # Calculate feature differences (A - B)
    print("Calculating feature differences...")
    metric_cols = [col for col in team_season_features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TeamName_Silver', 'Team_lower', 'Gender', 'team_lower']]
    
    diff_features = []
    for col in metric_cols:
        if f'A_{col}' in matchups.columns and f'B_{col}' in matchups.columns:
            diff_col = f'Diff_{col}'
            matchups[diff_col] = matchups[f'A_{col}'] - matchups[f'B_{col}']
            diff_features.append(diff_col)
    
    matchups['IsMen'] = matchups['TeamA'].apply(lambda t: 1 if str(t)[0] == '1' else 0)
    
    # In addition to differences, keep the absolute Cooper Ratings
    features = diff_features + ['IsMen', 'A_PredictedCooperRating', 'B_PredictedCooperRating', 'A_SeedNum', 'B_SeedNum']
    
    # Drop rows with NA (e.g. if we lack stats for a team in an early year)
    matchups.dropna(subset=features, inplace=True)
    
    return matchups, features

def post_process_predictions(preds, df_val):
    """
    Applies custom domain heuristics to ensemble outputs.
    1. Seed Overrides (1vs16, 2vs15)
    2. Prediction Boosting (boost picks under 85% by 10% towards certainty)
    """
    new_preds = preds.copy()
    diff_seeds = df_val['Diff_SeedNum'].values
    
    for i in range(len(new_preds)):
        p = new_preds[i]
        d_seed = diff_seeds[i]
        
        # 1. Seed Overrides (Averaged with raw probability to retain feature signal like Cooper Rating)
        # Team A is 1 seed, Team B is 16 seed (Diff = -15)
        if d_seed == -15:
            # Average raw prediction with a 98% historical expectation
            p = (p + 0.98) / 2.0
            p = max(p, 0.95) # Floor it at 95% certainty just in case
        # Team A is 16 seed, Team B is 1 seed (Diff = 15)
        elif d_seed == 15:
            p = (p + 0.02) / 2.0
            p = min(p, 0.05)
        # Team A is 2 seed, Team B is 15 seed (Diff = -13)
        elif d_seed == -13:
            # Average raw prediction with a 94% historical expectation
            p = (p + 0.94) / 2.0
            p = max(p, 0.85)
        # Team A is 15 seed, Team B is 2 seed (Diff = 13)
        elif d_seed == 13:
            p = (p + 0.06) / 2.0
            p = min(p, 0.15)
                
        new_preds[i] = p
        
    return new_preds

def train_tournament_model(df, features):
    """
    Trains XGBoost Classifier for probability prediction.
    """
    print("Training Tournament Prediction Model...")
    
    print("Executing 10-Year Rolling Cross-Validation...")
    
    # Identify the last 10 tournament seasons historically. (Note: 2020 was canceled)
    # The user is testing against the past 10 years for a < 0.17 target
    available_seasons = sorted(df['Season'].unique())
    # Exclude 2024 and 2025 if they are just samples, let's target 2013-2023 (10 tournaments, skipping 2020)
    test_seasons = [s for s in available_seasons if 2013 <= s <= 2023 and s != 2020]
    
    ensemble_briers = []
    
    for test_season in test_seasons:
        train_df = df[df['Season'] < test_season]
        val_df = df[df['Season'] == test_season]
        
        if len(train_df) == 0 or len(val_df) == 0:
            continue
            
        X_train = train_df[features]
        y_train = train_df['Target']
        
        X_val = val_df[features]
        y_val = val_df['Target']
        
        # 1. Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000, 
            learning_rate=0.02,
            max_depth=2,
            min_child_weight=1,
            subsample=0.6,
            colsample_bytree=0.6,
            gamma=0.3,
            reg_alpha=0.1,
            reg_lambda=1,
            eval_metric='logloss',
            early_stopping_rounds=50,
            random_state=42
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val_preds = xgb_model.predict_proba(X_val)[:, 1]
        
        # 2. Train LR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
        lr_model.fit(X_train_scaled, y_train)
        lr_val_preds = lr_model.predict_proba(X_val_scaled)[:, 1]
        
        # 3. Train MLP
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64), 
            activation='relu',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        mlp_model.fit(X_train_scaled, y_train)
        mlp_val_preds = mlp_model.predict_proba(X_val_scaled)[:, 1]
        
        # Aggregate Ensemble
        val_preds = (xgb_val_preds + lr_val_preds + mlp_val_preds) / 3.0
        
        # Phase 6: Post Processing
        val_preds_processed = post_process_predictions(val_preds, val_df)
        
        val_ll = log_loss(y_val, val_preds_processed)
        val_bs = brier_score_loss(y_val, val_preds_processed)
        
        print(f"Season {test_season} | Ensemble Log Loss: {val_ll:.4f} | Ensemble Brier: {val_bs:.4f}")
        ensemble_briers.append(val_bs)
        
    avg_brier = np.mean(ensemble_briers)
    print(f"\n======================================")
    print(f"10-YEAR AVERAGE BRIER SCORE: {avg_brier:.4f}")
    print(f"======================================\n")
    
    # Train final model on ALL data for submission
    print("Training final models on all historical data...")
    X_all = df[features]
    y_all = df['Target']
    
    final_model = xgb.XGBClassifier(
        n_estimators=300, # Hardcoded high robust number since no early_stopping
        learning_rate=0.02,
        max_depth=2,
        min_child_weight=1,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=0.3,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
    final_model.fit(X_all, y_all)
    
    # Train Final LR & MLP
    X_all_scaled = scaler.fit_transform(X_all) # Fit the scaler to EVERYTHING
    
    final_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    final_lr.fit(X_all_scaled, y_all)
    
    final_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    final_mlp.fit(X_all_scaled, y_all)
    
    return final_model, final_lr, final_mlp, scaler, features, df.iloc[0:1] # Just returns a dummy row to get column types

def create_2026_submission(xgb_model, lr_model, mlp_model, scaler, base_dir, features):
    """
    Produces the submission file for 2026 Phase 2.
    """
    print("\nGenerating 2026 predictions...")
    
    # Load Stage 2 Sample Submission
    sub = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'SampleSubmissionStage2.csv'))
    
    # Format of ID is YYYY_TeamA_TeamB
    sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
    sub['TeamA'] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
    sub['TeamB'] = sub['ID'].apply(lambda x: int(x.split('_')[2]))
    
    # We only care about 2026
    sub = sub[sub['Season'] == 2026].copy()
    
    # Load 2026 stats and predict Cooper ratings
    m_stats_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MRegularSeasonAggregatedStats.csv'))
    w_stats_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WRegularSeasonAggregatedStats.csv'))
    stats_df = pd.concat([m_stats_df, w_stats_df], ignore_index=True)
    
    m_cooper_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MGeneratedCooperRatings.csv'))
    w_cooper_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WGeneratedCooperRatings.csv'))
    cooper_df = pd.concat([m_cooper_df, w_cooper_df], ignore_index=True)
    
    m_seeds = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MNCAATourneySeeds.csv'))
    w_seeds = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WNCAATourneySeeds.csv'))
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)
    seeds['SeedNum'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    
    team_season_features = pd.merge(stats_df, cooper_df[['Season', 'TeamID', 'PredictedCooperRating']], on=['Season', 'TeamID'], how='left')
    team_season_features = pd.merge(team_season_features, seeds[['Season', 'TeamID', 'SeedNum']], on=['Season', 'TeamID'], how='left')
    team_season_features['SeedNum'] = team_season_features['SeedNum'].fillna(17)
    
    # Build dataframe for standardizing features
    pred_df = sub.copy()
    pred_df = pd.merge(pred_df, team_season_features, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], how='left')
    a_cols = {col: f'A_{col}' for col in team_season_features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TeamName_Silver']}
    pred_df.rename(columns=a_cols, inplace=True)
    
    pred_df = pd.merge(pred_df, team_season_features, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], how='left')
    b_cols = {col: f'B_{col}' for col in team_season_features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TeamName_Silver', 'Gender']}
    pred_df.rename(columns=b_cols, inplace=True)
    
    metric_cols = [col for col in team_season_features.columns if col not in ['Season', 'TeamID', 'TeamName', 'TeamName_Silver', 'Team_lower', 'Gender', 'team_lower']]
    
    for col in metric_cols:
        if f'A_{col}' in pred_df.columns and f'B_{col}' in pred_df.columns:
            diff_col = f'Diff_{col}'
            pred_df[diff_col] = pred_df[f'A_{col}'] - pred_df[f'B_{col}']
            
    pred_df['IsMen'] = pred_df['TeamA'].apply(lambda t: 1 if str(t)[0] == '1' else 0)
        
    X_test = pred_df[features]
    
    # Predict probabilities (class 1 is TeamA wins)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    
    X_test_scaled = scaler.transform(X_test)
    lr_preds = lr_model.predict_proba(X_test_scaled)[:, 1]
    mlp_preds = mlp_model.predict_proba(X_test_scaled)[:, 1]
    
    preds = (xgb_preds + lr_preds + mlp_preds) / 3.0
    
    # Apply post-processing heuristics
    preds = post_process_predictions(preds, pred_df)
    
    sub['Pred'] = preds
    
    # Save final submission
    out_path = os.path.join(base_dir, 'FINAL_MarchMadness_2026_Submission.csv')
    sub[['ID', 'Pred']].to_csv(out_path, index=False)
    print(f"DONE. 2026 Predictions saved to {out_path}")

if __name__ == "__main__":
    base_dir = "/Users/coopergilkey/Coding/March Madness"
    
    df, features = prep_tournament_data(base_dir)
    print("Features used:", features)
    xgb_model, lr_model, mlp_model, scaler, features, _ = train_tournament_model(df, features)
    create_2026_submission(xgb_model, lr_model, mlp_model, scaler, base_dir, features)
