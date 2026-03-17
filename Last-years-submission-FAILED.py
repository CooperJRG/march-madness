#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# Cell 1: Imports
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

BASE_DIR_KAGGLE = "march-machine-learning-mania-2026"
BASE_DIR_SILVER = "silver-standard"

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
from scipy.special import expit

plt.style.use('seaborn-whitegrid')


# In[ ]:


# ============================================================
# Cell 2: Utility Functions & Data Processing Helpers
# ============================================================
def seed_to_int(seed_str):
    """Convert a seed string like 'W01' or '16a' to an integer seed."""
    match = re.search(r'(\d+)', seed_str)
    return int(match.group(1)) if match else np.nan

def standardize_team_name(name):
    """
    Convert a team name like 'Saint Mary's (CA)' -> 'saint-marys-ca'
    """
    name = name.lower()
    for ch in ["'", "(", ")", ".", ","]:
        name = name.replace(ch, "")
    name = name.replace(" ", "-")
    return name

def load_silver_elo_for_year(year):
    """
    Loads Silver(COOPER) ratings for the given year from an Excel file.
    For 2024: uses 'elo'; for 2025: uses 'SBCB (Bayesian)' (renamed to 'elo').
    Returns a dict mapping standardized team names -> rating.
    """
    filepath = os.path.join(BASE_DIR_SILVER, f"SilverMen{year}.xlsx")
    df = pd.read_excel(filepath)
    df.dropna(axis=1, how='all', inplace=True)
    team_index = 'team'
    if year == 2025:
        df = df.iloc[14:]
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        columns_to_keep = ['Team', "SBCB (Bayesian)"]
        df = df.drop(df.columns.difference(columns_to_keep), axis=1)
        team_index = 'Team'
    df.columns = [col.strip() for col in df.columns]
    if "SBCB (Bayesian)" in df.columns:
        df.rename(columns={"SBCB (Bayesian)": "elo"}, inplace=True)
    df["team_std"] = df[team_index].apply(standardize_team_name)
    elo_dict = dict(zip(df["team_std"], df["elo"]))
    return elo_dict

def create_symmetric_dataset(df):
    """Create a symmetric dataset (two perspectives) from tournament results."""
    records = []
    for _, row in df.iterrows():
        records.append({
            'Season': row['Season'],
            'Team1': row['WTeamID'],
            'Team2': row['LTeamID'],
            'Team1Seed': row['WSeed'],
            'Team2Seed': row['LSeed'],
            'SeedDiff': row['WSeed'] - row['LSeed'],
            'Label': 1,  # Team1 wins
            'Margin': row['WScore'] - row['LScore']
        })
        records.append({
            'Season': row['Season'],
            'Team1': row['LTeamID'],
            'Team2': row['WTeamID'],
            'Team1Seed': row['LSeed'],
            'Team2Seed': row['WSeed'],
            'SeedDiff': row['LSeed'] - row['WSeed'],
            'Label': 0,  # Team1 loses
            'Margin': row['LScore'] - row['WScore']
        })
    return pd.DataFrame(records)

def compute_team_stats(df):
    """Compute per-team averages and variability from detailed regular season results."""
    records = []
    for _, row in df.iterrows():
        season = row['Season']
        records.append({
            'Season': season,
            'TeamID': row['WTeamID'],
            'PointsScored': row['WScore'],
            'PointsAllowed': row['LScore'],
            'PointDiff': row['WScore'] - row['LScore']
        })
        records.append({
            'Season': season,
            'TeamID': row['LTeamID'],
            'PointsScored': row['LScore'],
            'PointsAllowed': row['WScore'],
            'PointDiff': row['LScore'] - row['WScore']
        })
    df_records = pd.DataFrame(records)
    
    # Compute basic averages
    agg_funcs = {
        'PointsScored': 'mean',
        'PointsAllowed': 'mean',
        'PointDiff': ['mean', 'std']  # mean and standard deviation
    }
    stats = df_records.groupby(['Season', 'TeamID']).agg(agg_funcs).reset_index()
    stats.columns = ['Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'AvgPointDiff', 'StdPointDiff']
    
    # Optionally compute a ratio of points scored to allowed
    stats['PointsRatio'] = stats['AvgPointsScored'] / stats['AvgPointsAllowed']
    
    return stats


def score_diff_to_prob(score_diff, k=0.4):
    """Convert a predicted margin to win probability using a sigmoid."""
    return expit(k * score_diff)

def msilver_wpct(pwr1, pwr2):
    """Men's win probability using older Silver formula (optional)."""
    tscore = (pwr1 - pwr2) / 11
    return norm.cdf(tscore)

def wsilver_wpct(pwr1, pwr2, home=0):
    """Women's win probability with home advantage (optional)."""
    hfa = 2.73 * home
    tscore = (pwr1 - pwr2 + hfa) / 11.5
    return norm.cdf(tscore)



# In[3]:


# ============================================================
# Cell 3: Load and Process Men's Data
# ============================================================
# Load tournament results and seeds
regular_season = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'MRegularSeasonCompactResults.csv'))
tourney_results = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'MNCAATourneyCompactResults.csv'))
tourney_seeds = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'MNCAATourneySeeds.csv'))
tourney_seeds['SeedInt'] = tourney_seeds['Seed'].apply(seed_to_int)

w_seeds = tourney_seeds[['Season', 'TeamID', 'SeedInt']].rename(
    columns={'TeamID': 'Wteam', 'SeedInt': 'WSeed'}
)
l_seeds = tourney_seeds[['Season', 'TeamID', 'SeedInt']].rename(
    columns={'TeamID': 'Lteam', 'SeedInt': 'LSeed'}
)
men_data = tourney_results.merge(
    w_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'Wteam'], how='left'
).merge(
    l_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'Lteam'], how='left'
)
men_dataset = create_symmetric_dataset(men_data)

# Detailed season results for stats
try:
    detailed_men = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'MRegularSeasonDetailedResults.csv'))
except FileNotFoundError:
    print("MRegularSeasonDetailedResults.csv not found.")
    detailed_men = pd.DataFrame()

# Compute extended team stats (ensure your compute_team_stats returns the extended columns)
team_stats_men = compute_team_stats(detailed_men)

# Merge extended team stats for Team1
men_dataset = men_dataset.merge(
    team_stats_men[['Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'AvgPointDiff', 'StdPointDiff', 'PointsRatio']],
    left_on=['Season', 'Team1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'AvgPointsScored': 'Team1_AvgPointsScored',
    'AvgPointsAllowed': 'Team1_AvgPointsAllowed',
    'AvgPointDiff': 'Team1_AvgPointDiff',
    'StdPointDiff': 'Team1_StdPointDiff',
    'PointsRatio': 'Team1_PointsRatio'
}).drop(columns=['TeamID'])

# Merge extended team stats for Team2
men_dataset = men_dataset.merge(
    team_stats_men[['Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'AvgPointDiff', 'StdPointDiff', 'PointsRatio']],
    left_on=['Season', 'Team2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'AvgPointsScored': 'Team2_AvgPointsScored',
    'AvgPointsAllowed': 'Team2_AvgPointsAllowed',
    'AvgPointDiff': 'Team2_AvgPointDiff',
    'StdPointDiff': 'Team2_StdPointDiff',
    'PointsRatio': 'Team2_PointsRatio'
}).drop(columns=['TeamID'])

# Create matchup difference features
men_dataset['Diff_AvgPointDiff'] = men_dataset['Team1_AvgPointDiff'] - men_dataset['Team2_AvgPointDiff']
men_dataset['Diff_AvgPointsScored'] = men_dataset['Team1_AvgPointsScored'] - men_dataset['Team2_AvgPointsScored']
men_dataset['Diff_AvgPointsAllowed'] = men_dataset['Team1_AvgPointsAllowed'] - men_dataset['Team2_AvgPointsAllowed']
men_dataset['Diff_StdPointDiff'] = men_dataset['Team1_StdPointDiff'] - men_dataset['Team2_StdPointDiff']
men_dataset['Diff_PointsRatio'] = men_dataset['Team1_PointsRatio'] - men_dataset['Team2_PointsRatio']

men_dataset['Gender'] = 'Men'


# In[4]:


# ============================================================
# Cell 4: Load and Process Women's Data (with Extended Stats)
# ============================================================
try:
    detailed_women = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'WRegularSeasonDetailedResults.csv'))
except FileNotFoundError:
    print("WRegularSeasonDetailedResults.csv not found.")
    detailed_women = pd.DataFrame()

# Compute extended team stats for women
team_stats_women = compute_team_stats(detailed_women)

# Load tournament results and seeds for women
w_tourney_results = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'WNCAATourneyCompactResults.csv'))
w_tourney_seeds = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'WNCAATourneySeeds.csv'))
w_tourney_seeds['SeedInt'] = w_tourney_seeds['Seed'].apply(seed_to_int)

w_w_seeds = w_tourney_seeds[['Season', 'TeamID', 'SeedInt']].rename(
    columns={'TeamID': 'Wteam', 'SeedInt': 'WSeed'}
)
w_l_seeds = w_tourney_seeds[['Season', 'TeamID', 'SeedInt']].rename(
    columns={'TeamID': 'Lteam', 'SeedInt': 'LSeed'}
)
women_data = w_tourney_results.merge(
    w_w_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'Wteam'], how='left'
).merge(
    w_l_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'Lteam'], how='left'
)
women_dataset = create_symmetric_dataset(women_data)

# Merge extended team stats for Team1
women_dataset = women_dataset.merge(
    team_stats_women[['Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'AvgPointDiff', 'StdPointDiff', 'PointsRatio']],
    left_on=['Season', 'Team1'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'AvgPointsScored': 'Team1_AvgPointsScored',
    'AvgPointsAllowed': 'Team1_AvgPointsAllowed',
    'AvgPointDiff': 'Team1_AvgPointDiff',
    'StdPointDiff': 'Team1_StdPointDiff',
    'PointsRatio': 'Team1_PointsRatio'
}).drop(columns=['TeamID'])

# Merge extended team stats for Team2
women_dataset = women_dataset.merge(
    team_stats_women[['Season', 'TeamID', 'AvgPointsScored', 'AvgPointsAllowed', 'AvgPointDiff', 'StdPointDiff', 'PointsRatio']],
    left_on=['Season', 'Team2'],
    right_on=['Season', 'TeamID'],
    how='left'
).rename(columns={
    'AvgPointsScored': 'Team2_AvgPointsScored',
    'AvgPointsAllowed': 'Team2_AvgPointsAllowed',
    'AvgPointDiff': 'Team2_AvgPointDiff',
    'StdPointDiff': 'Team2_StdPointDiff',
    'PointsRatio': 'Team2_PointsRatio'
}).drop(columns=['TeamID'])

# Create matchup difference features
women_dataset['Diff_AvgPointDiff'] = women_dataset['Team1_AvgPointDiff'] - women_dataset['Team2_AvgPointDiff']
women_dataset['Diff_AvgPointsScored'] = women_dataset['Team1_AvgPointsScored'] - women_dataset['Team2_AvgPointsScored']
women_dataset['Diff_AvgPointsAllowed'] = women_dataset['Team1_AvgPointsAllowed'] - women_dataset['Team2_AvgPointsAllowed']
women_dataset['Diff_StdPointDiff'] = women_dataset['Team1_StdPointDiff'] - women_dataset['Team2_StdPointDiff']
women_dataset['Diff_PointsRatio'] = women_dataset['Team1_PointsRatio'] - women_dataset['Team2_PointsRatio']

women_dataset['Gender'] = 'Women'


# In[5]:


# ============================================================
# Cell 5: Combine Datasets & Train Combined Model using MAE Evaluation
# ============================================================
# Combine men's and women's datasets
combined_dataset = pd.concat([men_dataset, women_dataset], ignore_index=True)

# Define a comprehensive feature set including extended stats:
# - SeedDiff: Difference in tournament seeds.
# - Diff_AvgPointDiff: Difference in average point differential.
# - Diff_AvgPointsScored: Difference in average points scored.
# - Diff_AvgPointsAllowed: Difference in average points allowed.
# - Diff_StdPointDiff: Difference in the standard deviation of point differentials (measures consistency).
# - Diff_PointsRatio: Difference in points ratio (efficiency metric: avg points scored / avg points allowed).
features_combined = [
    'SeedDiff', 
    'Diff_AvgPointDiff', 
    'Diff_AvgPointsScored', 
    'Diff_AvgPointsAllowed', 
    'Diff_StdPointDiff', 
    'Diff_PointsRatio'
]

# Feature matrix and target variable
X_combined = combined_dataset[features_combined]
y_combined = combined_dataset['Margin']

# Initialize and train the XGBoost regressor with MAE as the evaluation metric
combined_xgb_reg = XGBRegressor(random_state=42, eval_metric='mae')
combined_xgb_reg.fit(X_combined, y_combined)

# Compute training predictions and evaluate using MAE
train_preds = combined_xgb_reg.predict(X_combined)
train_mae = mean_absolute_error(y_combined, train_preds)
print(f"Training MAE: {train_mae:.4f}")


# In[6]:


k_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
brier_scores = {}

for k in k_values:
    probs = [score_diff_to_prob(m, k=k) for m in y_combined]
    brier_scores[k] = brier_score_loss((y_combined > 0).astype(int), probs)  # assuming label 1 if margin > 0
    print(f"k={k} -> Brier Score: {brier_scores[k]:.4f}")


# In[7]:


# ============================================================
# Cell 6: Create Lookup Dictionaries
# ============================================================
def create_stats_lookup(stats_df):
    lookup = {}
    for _, row in stats_df.iterrows():
        lookup[(row['Season'], row['TeamID'])] = {
            'AvgPointsScored': row['AvgPointsScored'],
            'AvgPointsAllowed': row['AvgPointsAllowed'],
            'AvgPointDiff': row['AvgPointDiff']
        }
    return lookup

men_stats_lookup = create_stats_lookup(team_stats_men)
women_stats_lookup = create_stats_lookup(team_stats_women)


# In[8]:


# ============================================================
# Cell 7: Load Nate Silver Ratings
# ============================================================
silver_elo_lookup = {}
for yr in [2024, 2025]:
    try:
        silver_elo_lookup[yr] = load_silver_elo_for_year(yr)
        print(f"Loaded {len(silver_elo_lookup[yr])} team ratings for {yr}")
    except FileNotFoundError:
        silver_elo_lookup[yr] = {}
        print(f"No file found for {yr}; using empty dict.")
    except ValueError as e:
        silver_elo_lookup[yr] = {}
        print(f"Error loading {yr} data: {e}")


# In[9]:


# ============================================================
# Cell 8: Additional Utility Functions for Team Name Standardization
# ============================================================
def silver_wpct(rating1, rating2, scale=11):
    """
    Convert rating difference to win probability using a normal CDF.
    """
    diff = rating1 - rating2
    tscore = diff / scale
    return norm.cdf(tscore)

TEAM_NAME_OVERRIDES = {
    "St John's": "st-johns-ny",
    "Houston Chr": "houston-christian",
    "Texas A&M": "texas-am",
    "TAM C. Christi": "texas-am-corpus-christi",
    "Ole Miss": "ole-miss",
    "UNC Wilmington": "north-carolina-wilmington",
    "UNC Asheville": "north-carolina-asheville",
    "NC State": "north-carolina-state",
    "G Washington": "george-washington",
    "S Carolina St": "south-carolina-state",
    "S Dakota St": "south-dakota-state",
    "Miss Valley St": "mississippi-valley-state",
    "PFW": "purdue-fort-wayne",
    "MA Lowell": "massachusetts-lowell",
    "McNeese St": "mcneese-state",
    "SIUE": "southern-illinois-edwardsville",
    "Abilene Chr": "abilene-christian",
    "Ark Pine Bluff": "arkansas-pine-bluff",
    "UTRGV": "texas-rio-grande-valley",
    "E Michigan": "eastern-michigan",
    "Gonzaga": "gonzaga",
    "Connecticut": "connecticut",
}

def standardize_name_approx(kaggle_name: str) -> str:
    """Convert a Kaggle team name into a Silver-like standardized name."""
    name = kaggle_name.lower()
    for ch in ["'", "(", ")", ".", ",", "&"]:
        name = name.replace(ch, "")
    return name.replace(" ", "-")

def unify_team_name(kaggle_name: str) -> str:
    """Apply manual overrides then standardize the team name."""
    if kaggle_name in TEAM_NAME_OVERRIDES:
        return TEAM_NAME_OVERRIDES[kaggle_name]
    return standardize_name_approx(kaggle_name)


# In[24]:


# ============================================================
# Cell 9: Ensemble Prediction & Evaluation Functions
def predict_matchup_ensemble(team1, team2, seed1, seed2, season, gender,
                             model, stats_lookup, silver_lookup_dict, xgb_weight=0.5):
    """
    Compute win probability using an ensemble approach that combines the model's 
    margin-to-probability prediction and a baseline.
    
    Uses the complete feature set:
      - SeedDiff
      - Diff_AvgPointDiff
      - Diff_AvgPointsScored
      - Diff_AvgPointsAllowed
      - Diff_StdPointDiff
      - Diff_PointsRatio
    """
    # Default stats if team info is missing
    default_stats = {'AvgPointsScored': 0, 'AvgPointsAllowed': 0, 
                     'AvgPointDiff': 0, 'StdPointDiff': 0, 'PointsRatio': 0}
    s1 = stats_lookup.get((season, team1), default_stats)
    s2 = stats_lookup.get((season, team2), default_stats)
    
    # Compute differences for all features
    seed_diff = seed1 - seed2
    diff_avg_point_diff = s1['AvgPointDiff'] - s2['AvgPointDiff']
    diff_avg_points_scored = s1['AvgPointsScored'] - s2['AvgPointsScored']
    diff_avg_points_allowed = s1['AvgPointsAllowed'] - s2['AvgPointsAllowed']
    diff_std_point_diff = s1.get('StdPointDiff', 0) - s2.get('StdPointDiff', 0)
    diff_points_ratio = s1.get('PointsRatio', 0) - s2.get('PointsRatio', 0)
    
    # Create feature array with 6 features (matching training)
    features = np.array([[seed_diff, diff_avg_point_diff, diff_avg_points_scored,
                           diff_avg_points_allowed, diff_std_point_diff, diff_points_ratio]])
    predicted_margin = model.predict(features)[0]
    p_xgb = score_diff_to_prob(predicted_margin)
    
    # Silver baseline fixed at 0.5
    p_silver = 0.5  
    final_prob = xgb_weight * p_xgb + (1 - xgb_weight) * p_silver
    return final_prob


def evaluate_brier_ensemble(dataset, model, season, gender, silver_lookup_dict, xgb_weight=0.5):
    """
    Evaluate Brier score using the ensemble prediction (deterministic probability).
    """
    subset = dataset[(dataset['Season'] == season) & (dataset['Gender'] == gender)].copy()
    if subset.empty:
        print(f"No tournament data for {gender} {season}.")
        return None
    preds = []
    for _, row in subset.iterrows():
        p = predict_matchup_ensemble(
            row['Team1'], row['Team2'],
            row['Team1Seed'], row['Team2Seed'],
            season, gender, model, men_stats_lookup,  # assuming men_stats_lookup for men games
            silver_lookup_dict, xgb_weight
        )
        preds.append(p)
    brier_val = brier_score_loss(subset['Label'], preds)
    print(f"{gender} {season} Brier Score (Ensemble, xgb_weight={xgb_weight}): {brier_val:.4f}")
    return brier_val

# --- New: Simulation-based matchup probability ---
def simulate_matchup_prob(team1, team2, seed1, seed2, season, gender,
                          model, stats_lookup, silver_lookup_dict, xgb_weight=0.5, n_sim=100):
    """Simulate a matchup many times to approximate win probability."""
    wins = 0
    for _ in range(n_sim):
        p = predict_matchup_ensemble(team1, team2, seed1, seed2,
                                     season, gender, model, stats_lookup, silver_lookup_dict, xgb_weight)
        if np.random.rand() < p:
            wins += 1
    return wins / n_sim

def evaluate_brier_from_round_probs(dataset, round_probs, season, gender, round_key="Round of 32"):
    """
    Evaluate the Brier score by determining the most-likely winner of each game 
    based on precomputed round probabilities.
    
    For each game (row) in the dataset for the specified season and gender,
    the function looks up the team probabilities from round_probs for the given round_key.
    It then predicts the outcome deterministically: if Team1's probability is greater than 
    Team2's, predict win probability 1.0 for Team1 (and 0.0 for a loss); if lower, vice versa.
    In the rare case they are equal, a prediction of 0.5 is used.
    
    The predictions are compared to the actual outcomes (Label column) to compute the Brier score.
    
    Parameters:
      dataset: DataFrame containing tournament game data (must include columns 'Season', 'Gender', 
               'Team1', 'Team2', and 'Label').
      round_probs: dict mapping round names (e.g., "Round of 32") to dictionaries of {teamID: probability}
      season: season to evaluate (e.g., 2024)
      gender: tournament gender ("Men" or "Women")
      round_key: key in round_probs to use for predictions (default "Round of 32")
      
    Returns:
      brier_val: computed Brier score
    """
    subset = dataset[(dataset['Season'] == season) & (dataset['Gender'] == gender)].copy()
    if subset.empty:
        print(f"No tournament data available for {gender} {season}.")
        return None

    preds = []
    for _, row in subset.iterrows():
        team1 = row['Team1']
        team2 = row['Team2']
        # Look up the precomputed probability for each team in the specified round.
        p_team1 = round_probs.get(round_key, {}).get(team1, 0)
        p_team2 = round_probs.get(round_key, {}).get(team2, 0)
        
        # Determine the predicted outcome deterministically:
        # If Team1's probability is higher, predict win (1); if lower, predict loss (0).
        if p_team1 > p_team2:
            pred = 1.0
        elif p_team1 < p_team2:
            pred = 0.0
        else:
            pred = 0.5  # fallback in case of a tie
        preds.append(pred)
    
    brier_val = brier_score_loss(subset['Label'], preds)
    print(f"{gender} {season} Brier Score (Using '{round_key}' round probabilities): {brier_val:.4f}")
    return brier_val



def evaluate_brier_xgb_forced_champion(dataset, model, round_probs, season, gender, features):
    """
    Evaluate the Brier score using XGB margin predictions for each game,
    but with a forced outcome if a game features the "declared champion" (i.e., 
    the team with the highest championship round probability).
    
    The procedure is:
      1. Determine the forced champion from round_probs in the "Championship" round.
      2. For each game (row) in the dataset for the given season and gender:
            - If forced champion is one of the teams, force that outcome:
                * If Team1 == forced champion, predict win probability = 1.0.
                * If Team2 == forced champion, predict win probability = 0.0.
            - Otherwise, use the XGB margin prediction converted to win probability.
      3. Compute the Brier score between these predicted probabilities and the actual outcomes.
    
    Parameters:
      dataset: DataFrame containing tournament game data (must include columns 
               'Season', 'Gender', 'Team1', 'Team2', 'Label', and the feature columns).
      model: XGB model used for margin predictions.
      round_probs: dict mapping round names to dicts of {teamID: probability}; must
                   contain a "Championship" key.
      season: season to evaluate (e.g., 2024)
      gender: tournament gender ("Men" or "Women")
      features: list of column names used as features for the XGB margin model.
      
    Returns:
      brier_val: computed Brier score.
    """
    # Determine the forced champion from the "Championship" round probabilities.
    champ_probs = round_probs.get("Championship", {})
    if not champ_probs:
        print("No championship round probabilities provided.")
        return None
    forced_champion = max(champ_probs, key=champ_probs.get)
    
    subset = dataset[(dataset['Season'] == season) & (dataset['Gender'] == gender)].copy()
    if subset.empty:
        print(f"No tournament data available for {gender} {season}.")
        return None

    preds = []
    for _, row in subset.iterrows():
        team1 = row['Team1']
        team2 = row['Team2']
        # If forced champion is present in the matchup, override the prediction.
        if team1 == forced_champion:
            pred = 1.0
        elif team2 == forced_champion:
            pred = 0.0
        else:
            # Otherwise, compute prediction solely using the XGB margin.
            input_features = row[features].values.reshape(1, -1)
            predicted_margin = model.predict(input_features)[0]
            pred = score_diff_to_prob(predicted_margin)
        preds.append(pred)
    
    brier_val = brier_score_loss(subset['Label'], preds)
    print(f"{gender} {season} Brier Score (XGB Margin with forced champion {forced_champion}): {brier_val:.4f}")
    return brier_val


def evaluate_brier(dataset, model, season, gender, features):
    subset = dataset[(dataset['Season'] == season) & (dataset['Gender'] == gender)]
    if subset.empty:
        print(f"No tournament data available for {gender} {season}.")
        return None
    subset = subset.copy()
    subset['PredictedMargin'] = model.predict(subset[features])
    subset['PredictedWinProb'] = subset['PredictedMargin'].apply(score_diff_to_prob)
    brier_val = brier_score_loss(subset['Label'], subset['PredictedWinProb'])
    print(f"{gender} {season} Brier Score (XGB Margin->Prob): {brier_val:.4f}")
    return brier_val


# In[30]:


# ============================================================
# Cell 10: Simulation Functions with Progress (for Round Probabilities)
# ============================================================
from tqdm import tqdm

def simulate_region_bracket_ensemble_with_progress(season, region, gender, model, stats_lookup,
                                                   seeds_df, silver_lookup_dict, xgb_weight=1):
    """
    Simulate a regional bracket using ensemble prediction.
    Returns:
      - bracket: detailed game results
      - region_champion: (team, seed)
      - round_progress: dict mapping rounds ("Round of 32", "Sweet 16", "Elite 8", "Region Champion")
    """
    region_seeds = seeds_df[(seeds_df['Season'] == season) & (seeds_df['Seed'].str.startswith(region))]
    teams_in_region = region_seeds.drop_duplicates(subset=['SeedInt']).set_index('SeedInt')[['TeamID']].to_dict('index')
    
    round_progress = {"Round of 32": [], "Sweet 16": [], "Elite 8": [], "Region Champion": []}
    round1_pairings = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
    next_round = []
    bracket = {}
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
    round_results = []
    
    # Round 1: Round of 64 -> 32
    for pairing in round1_pairings:
        seedA, seedB = pairing
        if seedA not in teams_in_region or seedB not in teams_in_region:
            continue
        teamA = teams_in_region[seedA]['TeamID']
        teamB = teams_in_region[seedB]['TeamID']
        prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                        model, stats_lookup, silver_lookup_dict, xgb_weight)
        winner = teamA if np.random.rand() < prob else teamB
        round_results.append((teamA, teamB, winner, seedA, seedB, prob))
        adv_seed = seedA if winner == teamA else seedB
        next_round.append((winner, adv_seed))
        round_progress["Round of 32"].append(winner)
    bracket[round_names[1]] = round_results
    
    # Round 2: 32 -> Sweet 16
    current_round = next_round
    next_round = []
    round_results = []
    for i in range(0, len(current_round), 2):
        if i+1 < len(current_round):
            teamA, seedA = current_round[i]
            teamB, seedB = current_round[i+1]
            prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                            model, stats_lookup, silver_lookup_dict, xgb_weight)
            winner = teamA if np.random.rand() < prob else teamB
            round_results.append((teamA, teamB, winner, seedA, seedB, prob))
            adv_seed = seedA if winner == teamA else seedB
            next_round.append((winner, adv_seed))
            round_progress["Sweet 16"].append(winner)
    bracket[round_names[2]] = round_results
    
    # Round 3: Sweet 16 -> Elite 8
    current_round = next_round
    next_round = []
    round_results = []
    for i in range(0, len(current_round), 2):
        if i+1 < len(current_round):
            teamA, seedA = current_round[i]
            teamB, seedB = current_round[i+1]
            prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                            model, stats_lookup, silver_lookup_dict, xgb_weight)
            winner = teamA if np.random.rand() < prob else teamB
            round_results.append((teamA, teamB, winner, seedA, seedB, prob))
            adv_seed = seedA if winner == teamA else seedB
            next_round.append((winner, adv_seed))
            round_progress["Elite 8"].append(winner)
    bracket[round_names[3]] = round_results
    
    # Final round in region: Elite 8 -> Region Champion
    region_champion = None
    if next_round:
        region_champion = next_round[0]
        round_progress["Region Champion"].append(region_champion[0])
    
    return bracket, region_champion, round_progress

def simulate_final_four_ensemble_with_progress(region_champions, season, gender, model, stats_lookup,
                                               silver_lookup_dict, xgb_weight=1):
    """
    Simulate the Final Four and championship using ensemble prediction.
    Returns a dict with keys "Final Four" and "Championship".
    """
    final_progress = {"Final Four": [], "Championship": []}
    regions = sorted(region_champions.keys())
    
    # Semifinal 1
    teamA, seedA = region_champions[regions[0]]
    teamB, seedB = region_champions[regions[1]]
    prob1 = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                     model, stats_lookup, silver_lookup_dict, xgb_weight)
    winner1 = teamA if np.random.rand() < prob1 else teamB
    final_progress["Final Four"].append(winner1)
    
    # Semifinal 2
    teamC, seedC = region_champions[regions[2]]
    teamD, seedD = region_champions[regions[3]]
    prob2 = predict_matchup_ensemble(teamC, teamD, seedC, seedD, season, gender,
                                     model, stats_lookup, silver_lookup_dict, xgb_weight)
    winner2 = teamC if np.random.rand() < prob2 else teamD
    final_progress["Final Four"].append(winner2)
    
    # Championship
    seed_final1 = seedA if winner1 == teamA else seedB
    seed_final2 = seedC if winner2 == teamC else seedD
    prob_final = predict_matchup_ensemble(winner1, winner2, seed_final1, seed_final2,
                                          season, gender, model, stats_lookup,
                                          silver_lookup_dict, xgb_weight)
    champion = winner1 if np.random.rand() < prob_final else winner2
    final_progress["Championship"].append(champion)
    
    return final_progress

def simulate_tournament_ensemble_probabilities(season, gender, model, stats_lookup, seeds_df, silver_lookup_dict, n_sim=5000, xgb_weight=1):
    """
    Run tournament simulation n_sim times for a given season/gender and aggregate
    the probability for each team to reach each round:
      "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"
    """
    rounds = ["Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    team_round_counts = {rnd: {} for rnd in rounds}
    
    regions = sorted(seeds_df[seeds_df['Season'] == season]['Seed'].str[0].unique())
    if len(regions) < 4:
        print("Not enough regions to simulate tournament.")
        return {}
    
    # Wrap the simulation loop with tqdm for a progress bar.
    for _ in tqdm(range(n_sim), desc="Simulating tournament ensemble"):
        region_champs = {}
        region_round_progress = {}
        for region in regions:
            _, champ, progress = simulate_region_bracket_ensemble_with_progress(
                season, region, gender, model, stats_lookup, seeds_df, silver_lookup_dict, xgb_weight
            )
            region_round_progress[region] = progress
            if champ is not None:
                region_champs[region] = champ
        for progress in region_round_progress.values():
            for rnd in ["Round of 32", "Sweet 16", "Elite 8"]:
                for team in progress.get(rnd, []):
                    team_round_counts[rnd][team] = team_round_counts[rnd].get(team, 0) + 1
        if len(region_champs) >= 4:
            final_progress = simulate_final_four_ensemble_with_progress(
                region_champs, season, gender, model, stats_lookup, silver_lookup_dict, xgb_weight
            )
            for rnd in ["Final Four", "Championship"]:
                for team in final_progress.get(rnd, []):
                    team_round_counts[rnd][team] = team_round_counts[rnd].get(team, 0) + 1
    team_round_prob = {rnd: {team: count / n_sim for team, count in team_round_counts[rnd].items()} for rnd in rounds}
    return team_round_prob

# ============================================================
# Cell 12 (Moved Up): Load Team Names
# ============================================================
teams = pd.read_csv(os.path.join(BASE_DIR_KAGGLE, 'MTeams.csv'))
team_names = dict(zip(teams['TeamID'], teams['TeamName']))

# Example usage for round probabilities:
men_round_probs_2024 = simulate_tournament_ensemble_probabilities(
    2024, 'Men', combined_xgb_reg, men_stats_lookup, tourney_seeds, silver_elo_lookup, n_sim=1, xgb_weight=1
)
print("\nAggregated Round Probabilities for Men 2024:")
print(men_round_probs_2024)

men_round_probs_2025 = simulate_tournament_ensemble_probabilities(
    2025, 'Men', combined_xgb_reg, men_stats_lookup, tourney_seeds, silver_elo_lookup, n_sim=1000, xgb_weight=1
)
print("\nAggregated Round Probabilities for Men 2025:")
named_probs = {}
for round_name, team_probs in men_round_probs_2025.items():
    # Convert each team ID to its name (or string if not found)
    named_team_probs = { team_names.get(team, str(team)): prob for team, prob in team_probs.items() }
    named_probs[round_name] = named_team_probs

print(named_probs)


# In[17]:


# ============================================================
# Cell 11: Simulation Functions Without Progress & Printing Helpers
# (Used for bracket prediction)
# ============================================================
def simulate_region_bracket_ensemble(season, region, gender, model, stats_lookup,
                                     seeds_df, silver_lookup_dict, xgb_weight=0.5, forced_champion=None):
    """
    Simulate a regional bracket (without detailed progress) using XGB margin predictions,
    but forcing the outcome for any game that involves the forced champion.
    
    Parameters:
      forced_champion: if not None, a teamID that will always win when present in a matchup.
      
    Returns:
      - bracket: dict of rounds with game details
      - region_champion: tuple (team, seed)
    """
    region_seeds = seeds_df[(seeds_df['Season'] == season) & (seeds_df['Seed'].str.startswith(region))]
    teams_in_region = region_seeds.drop_duplicates(subset=['SeedInt']).set_index('SeedInt')[['TeamID']].to_dict('index')
    
    round1_pairings = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]
    bracket = {}
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]
    round_results = []
    next_round = []
    
    # Round 1
    for pairing in round1_pairings:
        seedA, seedB = pairing
        if seedA not in teams_in_region or seedB not in teams_in_region:
            continue
        teamA = teams_in_region[seedA]['TeamID']
        teamB = teams_in_region[seedB]['TeamID']
        
        # Forced champion override: if either team is the forced champion, that team wins.
        if forced_champion is not None:
            if teamA == forced_champion:
                winner = teamA
                prob = 1.0
            elif teamB == forced_champion:
                winner = teamB
                prob = 0.0
            else:
                prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                                model, stats_lookup, silver_lookup_dict, xgb_weight)
                winner = teamA if prob >= 0.5 else teamB
        else:
            prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                            model, stats_lookup, silver_lookup_dict, xgb_weight)
            winner = teamA if prob >= 0.5 else teamB
        
        round_results.append((teamA, teamB, winner, seedA, seedB, prob))
        adv_seed = seedA if winner == teamA else seedB
        next_round.append((winner, adv_seed))
    bracket[round_names[0]] = round_results
    
    # Subsequent rounds (Rounds of 32, Sweet 16, Elite 8)
    for idx in range(1, len(round_names)):
        current_round_results = []
        new_round = []
        for i in range(0, len(next_round), 2):
            if i+1 < len(next_round):
                teamA, seedA = next_round[i]
                teamB, seedB = next_round[i+1]
                if forced_champion is not None:
                    if teamA == forced_champion:
                        winner = teamA
                        prob = 1.0
                    elif teamB == forced_champion:
                        winner = teamB
                        prob = 0.0
                    else:
                        prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                                        model, stats_lookup, silver_lookup_dict, xgb_weight)
                        winner = teamA if prob >= 0.5 else teamB
                else:
                    prob = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                                    model, stats_lookup, silver_lookup_dict, xgb_weight)
                    winner = teamA if prob >= 0.5 else teamB
                current_round_results.append((teamA, teamB, winner, seedA, seedB, prob))
                adv_seed = seedA if winner == teamA else seedB
                new_round.append((winner, adv_seed))
        bracket[round_names[idx]] = current_round_results
        next_round = new_round
    
    region_champion = next_round[0] if next_round else (None, None)
    return bracket, region_champion

def simulate_final_four_ensemble(region_champions, season, gender, model, stats_lookup,
                                 silver_lookup_dict, xgb_weight=0.5, forced_champion=None):
    """
    Simulate the Final Four and championship using XGB margin predictions,
    forcing any game that involves the forced champion.
    
    Parameters:
      forced_champion: if not None, a teamID that will always win when present.
      
    Returns:
      final_four: dict with keys 'Semifinal' and 'Championship'
    """
    regions = sorted(region_champions.keys())
    semifinal_results = []
    
    # Semifinal 1
    teamA, seedA = region_champions[regions[0]]
    teamB, seedB = region_champions[regions[1]]
    if forced_champion is not None:
        if teamA == forced_champion:
            winner1 = teamA
            prob1 = 1.0
        elif teamB == forced_champion:
            winner1 = teamB
            prob1 = 0.0
        else:
            prob1 = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                             model, stats_lookup, silver_lookup_dict, xgb_weight)
            winner1 = teamA if prob1 >= 0.5 else teamB
    else:
        prob1 = predict_matchup_ensemble(teamA, teamB, seedA, seedB, season, gender,
                                         model, stats_lookup, silver_lookup_dict, xgb_weight)
        winner1 = teamA if prob1 >= 0.5 else teamB
    semifinal_results.append((teamA, teamB, winner1, seedA, seedB, prob1))
    
    # Semifinal 2
    teamC, seedC = region_champions[regions[2]]
    teamD, seedD = region_champions[regions[3]]
    if forced_champion is not None:
        if teamC == forced_champion:
            winner2 = teamC
            prob2 = 1.0
        elif teamD == forced_champion:
            winner2 = teamD
            prob2 = 0.0
        else:
            prob2 = predict_matchup_ensemble(teamC, teamD, seedC, seedD, season, gender,
                                             model, stats_lookup, silver_lookup_dict, xgb_weight)
            winner2 = teamC if prob2 >= 0.5 else teamD
    else:
        prob2 = predict_matchup_ensemble(teamC, teamD, seedC, seedD, season, gender,
                                         model, stats_lookup, silver_lookup_dict, xgb_weight)
        winner2 = teamC if prob2 >= 0.5 else teamD
    semifinal_results.append((teamC, teamD, winner2, seedC, seedD, prob2))
    
    final_four = {'Semifinal': semifinal_results}
    
    # Championship matchup
    seed_final1 = seedA if winner1 == teamA else seedB
    seed_final2 = seedC if winner2 == teamC else seedD
    if forced_champion is not None:
        if winner1 == forced_champion:
            champion = winner1
            prob_final = 1.0
        elif winner2 == forced_champion:
            champion = winner2
            prob_final = 0.0
        else:
            prob_final = predict_matchup_ensemble(winner1, winner2, seed_final1, seed_final2,
                                                  season, gender, model, stats_lookup,
                                                  silver_lookup_dict, xgb_weight)
            champion = winner1 if prob_final >= 0.5 else winner2
    else:
        prob_final = predict_matchup_ensemble(winner1, winner2, seed_final1, seed_final2,
                                              season, gender, model, stats_lookup,
                                              silver_lookup_dict, xgb_weight)
        champion = winner1 if prob_final >= 0.5 else winner2
    final_four['Championship'] = (winner1, winner2, champion, seed_final1, seed_final2, prob_final)
    
    return final_four


def print_region_bracket(region, bracket, team_names):
    print(f"\nRegion: {region}")
    for round_name, games in bracket.items():
        print(f"\n{round_name}:")
        for game in games:
            teamA, teamB, winner, seedA, seedB, prob = game
            nameA = team_names.get(teamA, str(teamA))
            nameB = team_names.get(teamB, str(teamB))
            winner_name = team_names.get(winner, str(winner))
            print(f"  {nameA} (Seed {seedA}) vs {nameB} (Seed {seedB}) -> Winner: {winner_name} (Prob: {prob:.2f})")

def print_final_four(final_four, team_names):
    print("\nFinal Four:")
    for i, game in enumerate(final_four['Semifinal'], 1):
        teamA, teamB, winner, seedA, seedB, prob = game
        nameA = team_names.get(teamA, str(teamA))
        nameB = team_names.get(teamB, str(teamB))
        winner_name = team_names.get(winner, str(winner))
        print(f"  Semifinal {i}: {nameA} (Seed {seedA}) vs {nameB} (Seed {seedB}) -> Winner: {winner_name} (Prob: {prob:.2f})")
    teamA, teamB, champion, seedA, seedB, prob = final_four['Championship']
    nameA = team_names.get(teamA, str(teamA))
    nameB = team_names.get(teamB, str(teamB))
    champion_name = team_names.get(champion, str(champion))
    print("\nChampionship:")
    print(f"  {nameA} (Seed {seedA}) vs {nameB} (Seed {seedB}) -> Champion: {champion_name} (Prob: {prob:.2f})")


# In[18]:


# Removed cell 12 since it was moved up


# ============================================================
# Cell 13 (Penultimate): Evaluate Brier Scores for 2024 Season
# ============================================================
print("\nEvaluating Brier Scores for 2024 Season:")

# For Women (XGB-based margin->prob)
brier_women_xgb = evaluate_brier(combined_dataset, combined_xgb_reg, 2024, 'Women', features_combined)

# For Men using XGB predictions directly
brier_men_xgb = evaluate_brier(combined_dataset, combined_xgb_reg, 2024, 'Men', features_combined)

# For Men using Ensemble predictions
brier_men_ensemble = evaluate_brier_ensemble(combined_dataset, combined_xgb_reg, 2024, 'Men', silver_elo_lookup, xgb_weight=1)

# For Men using Simulation-based predictions
brier_men_simulation = evaluate_brier_from_round_probs(
    combined_dataset,
    men_round_probs_2024,
    2024,
    'Men'
)

brier_men_xgb_forced = evaluate_brier_xgb_forced_champion(
    dataset=combined_dataset,
    model=combined_xgb_reg,
    round_probs=men_round_probs_2024,
    season=2024,
    gender='Men',
    features=features_combined
)


# In[27]:


# ============================================================
# Cell 14 (Final): Simulated Predicted Bracket with Forced Champion for 2025
# ============================================================
men_seasons = sorted(tourney_seeds['Season'].unique())
women_seasons = sorted(w_tourney_seeds['Season'].unique())
predicted_season_men = 2025 if 2025 in men_seasons else men_seasons[-1]
predicted_season_women = 2025 if 2025 in women_seasons else women_seasons[-1]

# Determine forced champion for Men from the championship round probabilities
if "Championship" in men_round_probs_2025 and men_round_probs_2025["Championship"]:
    forced_champion_men = max(men_round_probs_2025["Championship"], key=men_round_probs_2025["Championship"].get)
else:
    forced_champion_men = None

print("\n" + "="*50)
print("Simulated Predicted Bracket (XGB Margin with Forced Champion)")

for gender, season, seeds_df, stats_lookup, forced_champion in [
        ('Men', predicted_season_men, tourney_seeds, men_stats_lookup, forced_champion_men),
        ('Women', predicted_season_women, w_tourney_seeds, women_stats_lookup, None)]:
    
    print(f"\nSimulated Bracket for {gender} in Season {season}:")
    region_champs = {}
    regions = sorted(seeds_df[seeds_df['Season'] == season]['Seed'].str[0].unique())
    for region in regions:
        bracket, champ = simulate_region_bracket_ensemble(
            season, region, gender, combined_xgb_reg,
            stats_lookup, seeds_df, silver_elo_lookup, xgb_weight=0.5,
            forced_champion=forced_champion
        )
        region_champs[region] = champ
        print_region_bracket(region, bracket, team_names)
    if len(region_champs) >= 4:
        final_four = simulate_final_four_ensemble(
            region_champs, season, gender, combined_xgb_reg,
            stats_lookup, silver_elo_lookup, xgb_weight=0.5,
            forced_champion=forced_champion
        )
        print_final_four(final_four, team_names)
    else:
        print(f"Not enough regions to simulate a Final Four for {gender}")


# In[ ]:




