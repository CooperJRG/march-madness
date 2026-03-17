import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss, brier_score_loss, make_scorer

# Import our data prep logic so we don't duplicate it
from tournament_model import prep_tournament_data

def brier_scorer(y_true, y_prob):
    # sklearn tries to maximize score, so return negative brier score
    return -brier_score_loss(y_true, y_prob)

if __name__ == "__main__":
    base_dir = "/Users/coopergilkey/Coding/March Madness"
    
    # 1. Load the prepped data exactly as the model does
    df, features = prep_tournament_data(base_dir)
    print(f"Loaded {len(df)} historical matchups with {len(features)} features.")
    
    # 2. Define train/val split (We will use pre-2023 for CV, validate on 2023)
    train_df = df[df['Season'] < 2023]
    val_df = df[df['Season'] == 2023]
    
    X_train = train_df[features]
    y_train = train_df['Target']
    
    X_val = val_df[features]
    y_val = val_df['Target']
    
    # 3. Define parameter grid for RandomizedSearchCV
    param_grid = {
        'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5, 6],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 5, 10, 20]
    }
    
    # Base classifier (using early stopping inside cross-val requires special setup,
    # so we'll use a fixed number of estimators for the search for speed, then manually test the best on val)
    clf = xgb.XGBClassifier(
        n_estimators=300, # Kept moderate for CV speed
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    # Custom scorer for Brier
    brier = make_scorer(brier_scorer, needs_proba=True)
    
    print("Starting RandomizedSearchCV...")
    # Perform search
    random_search = RandomizedSearchCV(
        clf, 
        param_distributions=param_grid, 
        n_iter=40, # Number of parameter settings that are sampled
        scoring=brier, 
        cv=3, # 3-fold CV on pre-2023 data
        verbose=1, 
        n_jobs=-1, # Use all cores
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nBest Parameters from CV:")
    best_params = random_search.best_params_
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    # Evaluate best model on 2023 holdout
    print("\nTraining best model on 2023 Holdout Set to find optimal estimators...")
    best_clf = xgb.XGBClassifier(
        **best_params,
        n_estimators=2000,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42
    )

    best_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    
    val_preds = best_clf.predict_proba(X_val)[:, 1]
    best_ll = log_loss(y_val, val_preds)
    best_bs = brier_score_loss(y_val, val_preds)
    
    print(f"\n2023 Validation Log Loss: {best_ll:.4f}")
    print(f"2023 Validation Brier Score: {best_bs:.4f}")
    print(f"Optimal n_estimators: {best_clf.best_iteration}")
    print("\nDONE.")
