import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

def load_and_merge_data(base_dir):
    """
    Loads aggregated stats and silver/gold standard cooper ratings to build the training set for both Men and Women.
    """
    print("Loading aggregated stats...")
    m_stats_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MRegularSeasonAggregatedStats.csv'))
    w_stats_df = pd.read_csv(os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WRegularSeasonAggregatedStats.csv'))
    stats_df = pd.concat([m_stats_df, w_stats_df], ignore_index=True)
    
    print("Loading Silver Standard Data for 2024 and 2025...")
    # Men 2024
    silver_m_2024 = pd.read_excel(os.path.join(base_dir, 'silver-standard', 'SilverMen2024.xlsx'))
    silver_m_2024['Season'] = 2024
    silver_m_2024['Gender'] = 'Men'
    
    # Women 2024 (Note the typo in filename 'SiverWomen2024.xlsx')
    silver_w_2024_raw = pd.read_excel(os.path.join(base_dir, 'silver-standard', 'SiverWomen2024.xlsx'), skiprows=1)
    silver_w_2024 = pd.DataFrame()
    silver_w_2024['team'] = silver_w_2024_raw['Team']
    silver_w_2024['elo'] = silver_w_2024_raw['Composite']
    silver_w_2024['Season'] = 2024
    silver_w_2024['Gender'] = 'Women'
    
    # Men 2025
    silver_m_2025_raw = pd.read_excel(os.path.join(base_dir, 'silver-standard', 'SilverMen2025.xlsx'), skiprows=20, header=None)
    silver_m_2025 = pd.DataFrame()
    silver_m_2025['team'] = silver_m_2025_raw.iloc[:, 0]
    silver_m_2025['elo'] = silver_m_2025_raw.iloc[:, -1]
    silver_m_2025['Season'] = 2025
    silver_m_2025['Gender'] = 'Men'
    
    print("Loading Gold Standard Data (as 2026 ground truth)...")
    # Men Gold
    try:
        gold_m_2026 = pd.read_csv(os.path.join(base_dir, 'gold-standard', 'MCooperRatings.csv'))
        gold_m_2026.rename(columns={'sb_name': 'team', 'b_xelo_n': 'elo'}, inplace=True)
        gold_m_2026['Season'] = 2026
        gold_m_2026['Gender'] = 'Men'
        gold_m_2026 = gold_m_2026[['team', 'elo', 'Season', 'Gender']]
    except Exception as e:
        print(f"Warning: Could not load Mens gold standard data: {e}")
        gold_m_2026 = pd.DataFrame()

    # Women Gold
    try:
        gold_w_2026 = pd.read_csv(os.path.join(base_dir, 'gold-standard', 'WCooperRatings.csv'))
        gold_w_2026.rename(columns={'sb_name': 'team', 'b_xelo_n': 'elo'}, inplace=True)
        gold_w_2026['Season'] = 2026
        gold_w_2026['Gender'] = 'Women'
        gold_w_2026 = gold_w_2026[['team', 'elo', 'Season', 'Gender']]
    except Exception as e:
        print(f"Warning: Could not load Womens gold standard data: {e}")
        gold_w_2026 = pd.DataFrame()

    # Combine the target data
    targets_df = pd.concat([
        silver_m_2024[['team', 'elo', 'Season', 'Gender']], 
        silver_w_2024[['team', 'elo', 'Season', 'Gender']], 
        silver_m_2025[['team', 'elo', 'Season', 'Gender']],
        gold_m_2026,
        gold_w_2026
    ], ignore_index=True)
    
    print("Merging targets with Kaggle stats...")
    
    targets_df['team_lower'] = targets_df['team'].astype(str).str.lower().str.strip()
    stats_df['team_lower'] = stats_df['TeamName_Silver'].astype(str).str.lower().str.strip()
    
    replace_dict = {
        "st. john's": "st john's",
        "saint mary's (ca)": "st mary's ca",
        "u miami (fl)": "miami fl",
        "uconn": "connecticut",
        "north carolina st": "nc state",
    }
    stats_df['team_lower'] = stats_df['team_lower'].replace(replace_dict)
    
    # Merge on Season, Gender, and Team Name
    df_merged = pd.merge(stats_df, targets_df, left_on=['Season', 'Gender', 'team_lower'], right_on=['Season', 'Gender', 'team_lower'], how='inner')
    print(f"Successfully joined {len(df_merged)} records for training (Men + Women).")
    
    return df_merged, m_stats_df, w_stats_df

def feature_engineering(df):
    """
    Creates derived features.
    """
    # A lot of these are already created in data_prep.py
    features = [
        'WinPct', 'PointsPG', 'OppPointsPG', 'FGM_mean', 'FGA_mean', 'FGM3_mean', 'FGA3_mean',
        'FTM_mean', 'FTA_mean', 'OR_mean', 'DR_mean', 'Ast_mean', 'TO_mean', 'Stl_mean', 'Blk_mean', 'PF_mean',
        'OppFGM_mean', 'OppFGA_mean', 'OppFGM3_mean', 'OppFGA3_mean', 'OppFTM_mean', 'OppFTA_mean',
        'OppOR_mean', 'OppDR_mean', 'OppAst_mean', 'OppTO_mean', 'OppStl_mean', 'OppBlk_mean', 'OppPF_mean',
        'GamePossessions_mean', 'OffensiveRating_mean', 'DefensiveRating_mean', 'NetRating_mean', 'SOS'
    ]
    return df, features

def train_and_eval_model(df_merged, features):
    print("Training Cooper Ratings regression model using XGBoost...")
    # Drop rows where target 'elo' is NaN
    df_merged = df_merged.dropna(subset=['elo'])
    
    X = df_merged[features]
    y = df_merged['elo']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        
    print(f"Average Out-of-Fold Validation MAE: {np.mean(maes):.2f}")
    
    # Train full model
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    final_model.fit(X, y)
    
    return final_model

def generate_historical_ratings(model, all_stats, features, out_path):
    print(f"Predicting Cooper Ratings for {len(all_stats)} historical team-seasons...")
    X_all = all_stats[features]
    
    # Predict
    all_stats['PredictedCooperRating'] = model.predict(X_all)
    
    # Select important columns to save
    out_df = all_stats[['Season', 'TeamID', 'TeamName', 'PredictedCooperRating']]
    out_df.to_csv(out_path, index=False)
    print(f"Saved generated cooper ratings to {out_path}")
    
    # Display the top 10 from 2021 as a sanity check
    print("\nSanity Check - Top 10 Teams in 2021:")
    print(out_df[out_df['Season'] == 2021].sort_values(by='PredictedCooperRating', ascending=False).head(10))

if __name__ == "__main__":
    base_dir = "/Users/coopergilkey/Coding/March Madness"
    
    # 1. Load data
    df_merged, m_stats_df, w_stats_df = load_and_merge_data(base_dir)
    
    # 2. Features
    df_merged, features = feature_engineering(df_merged)
    m_stats_df, _ = feature_engineering(m_stats_df)
    w_stats_df, _ = feature_engineering(w_stats_df)
    
    # 3. Train
    model = train_and_eval_model(df_merged, features)
    
    # 4. Predict
    out_path_m = os.path.join(base_dir, 'march-machine-learning-mania-2026', 'MGeneratedCooperRatings.csv')
    generate_historical_ratings(model, m_stats_df, features, out_path_m)
    
    out_path_w = os.path.join(base_dir, 'march-machine-learning-mania-2026', 'WGeneratedCooperRatings.csv')
    generate_historical_ratings(model, w_stats_df, features, out_path_w)
