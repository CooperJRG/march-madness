import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def calculate_regular_season_stats(season_results_file):
    """
    Reads MRegularSeasonDetailedResults.csv or WRegularSeasonDetailedResults.csv
    and computes aggregated statistics per team per season.
    """
    print(f"Reading {season_results_file}...")
    df = pd.read_csv(season_results_file)
    
    # We want stats per Team per Season. Let's stack winners and losers.
    
    # Winners perspective
    w_df = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 
               'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
               'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
               'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 
               'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']].copy()
    w_df['Win'] = 1
    w_df.rename(columns={
        'WTeamID': 'TeamID', 'WScore': 'Points', 'LTeamID': 'OppID', 'LScore': 'OppPoints',
        'WLoc': 'Loc',
        'WFGM': 'FGM', 'WFGA': 'FGA', 'WFGM3': 'FGM3', 'WFGA3': 'FGA3', 'WFTM': 'FTM', 'WFTA': 'FTA',
        'WOR': 'OR', 'WDR': 'DR', 'WAst': 'Ast', 'WTO': 'TO', 'WStl': 'Stl', 'WBlk': 'Blk', 'WPF': 'PF',
        'LFGM': 'OppFGM', 'LFGA': 'OppFGA', 'LFGM3': 'OppFGM3', 'LFGA3': 'OppFGA3', 'LFTM': 'OppFTM', 'LFTA': 'OppFTA',
        'LOR': 'OppOR', 'LDR': 'OppDR', 'LAst': 'OppAst', 'LTO': 'OppTO', 'LStl': 'OppStl', 'LBlk': 'OppBlk', 'LPF': 'OppPF'
    }, inplace=True)
    
    # Losers perspective
    l_df = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc',
               'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 
               'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
               'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 
               'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].copy()
    l_df['Win'] = 0
    def flip_loc(loc):
        if loc == 'H': return 'A'
        if loc == 'A': return 'H'
        return 'N'
    l_df['WLoc'] = l_df['WLoc'].apply(flip_loc)
    
    l_df.rename(columns={
        'LTeamID': 'TeamID', 'LScore': 'Points', 'WTeamID': 'OppID', 'WScore': 'OppPoints',
        'WLoc': 'Loc',
        'LFGM': 'FGM', 'LFGA': 'FGA', 'LFGM3': 'FGM3', 'LFGA3': 'FGA3', 'LFTM': 'FTM', 'LFTA': 'FTA',
        'LOR': 'OR', 'LDR': 'DR', 'LAst': 'Ast', 'LTO': 'TO', 'LStl': 'Stl', 'LBlk': 'Blk', 'LPF': 'PF',
        'WFGM': 'OppFGM', 'WFGA': 'OppFGA', 'WFGM3': 'OppFGM3', 'WFGA3': 'OppFGA3', 'WFTM': 'OppFTM', 'WFTA': 'OppFTA',
        'WOR': 'OppOR', 'WDR': 'OppDR', 'WAst': 'OppAst', 'WTO': 'OppTO', 'WStl': 'OppStl', 'WBlk': 'OppBlk', 'WPF': 'OppPF'
    }, inplace=True)
    
    # Combine
    games = pd.concat([w_df, l_df], ignore_index=True)
    
    # Calculate Possessions for each game (approximate formula)
    # Possessions = FGA - OR + TO + 0.475 * FTA
    games['Possessions'] = games['FGA'] - games['OR'] + games['TO'] + 0.475 * games['FTA']
    games['OppPossessions'] = games['OppFGA'] - games['OppOR'] + games['OppTO'] + 0.475 * games['OppFTA']
    # Use average possessions for the game
    games['GamePossessions'] = (games['Possessions'] + games['OppPossessions']) / 2.0
    
    games['OffensiveRating'] = 100.0 * games['Points'] / games['GamePossessions']
    games['DefensiveRating'] = 100.0 * games['OppPoints'] / games['GamePossessions']
    games['NetRating'] = games['OffensiveRating'] - games['DefensiveRating']
    
    # Group by Season and TeamID
    print("Aggregating team stats...")
    agg_funcs = {
        'Win': ['count', 'mean'],
        'Points': 'mean',
        'OppPoints': 'mean',
        'FGM': 'mean',
        'FGA': 'mean',
        'FGM3': 'mean',
        'FGA3': 'mean',
        'FTM': 'mean',
        'FTA': 'mean',
        'OR': 'mean',
        'DR': 'mean',
        'Ast': 'mean',
        'TO': 'mean',
        'Stl': 'mean',
        'Blk': 'mean',
        'PF': 'mean',
        'OppFGM': 'mean',
        'OppFGA': 'mean',
        'OppFGM3': 'mean',
        'OppFGA3': 'mean',
        'OppFTM': 'mean',
        'OppFTA': 'mean',
        'OppOR': 'mean',
        'OppDR': 'mean',
        'OppAst': 'mean',
        'OppTO': 'mean',
        'OppStl': 'mean',
        'OppBlk': 'mean',
        'OppPF': 'mean',
        'GamePossessions': 'mean',
        'OffensiveRating': 'mean',
        'DefensiveRating': 'mean',
        'NetRating': 'mean'
    }
    
    team_season_stats = games.groupby(['Season', 'TeamID']).agg(agg_funcs).reset_index()
    
    # Flatten multi-level columns
    team_season_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in team_season_stats.columns.values]
    
    # Rename some columns for clarity
    team_season_stats.rename(columns={
        'Win_count': 'GamesPlayed',
        'Win_mean': 'WinPct',
        'Points_mean': 'PointsPG',
        'OppPoints_mean': 'OppPointsPG',
    }, inplace=True)
    
    # Calculate Recent (Last 30 Days) Stats
    print("Aggregating RECENT team stats (DayNum >= 100)...")
    recent_games = games[games['DayNum'] >= 100]
    recent_stats = recent_games.groupby(['Season', 'TeamID']).agg(agg_funcs).reset_index()
    recent_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in recent_stats.columns.values]
    
    recent_stats.rename(columns={
        'Win_count': 'Recent_GamesPlayed',
        'Win_mean': 'Recent_WinPct',
        'Points_mean': 'Recent_PointsPG',
        'OppPoints_mean': 'Recent_OppPointsPG',
    }, inplace=True)
    
    recent_rename_dict = {}
    for col in recent_stats.columns:
        if col not in ['Season', 'TeamID', 'Recent_GamesPlayed', 'Recent_WinPct', 'Recent_PointsPG', 'Recent_OppPointsPG']:
            recent_rename_dict[col] = f'Recent_{col}'
    recent_stats.rename(columns=recent_rename_dict, inplace=True)
    
    # Calculate simple strength of schedule (average opp win pct)
    print("Calculating Strength of Schedule...")
    # Create dictionary of (Season, TeamID) -> WinPct
    win_pct_dict = team_season_stats.set_index(['Season', 'TeamID'])['WinPct'].to_dict()
    
    def get_opp_win_pct(row):
        opp_id = row['OppID']
        season = row['Season']
        return win_pct_dict.get((season, opp_id), 0.5) # default to 0.5 if opp not found
    
    games['OppWinPct'] = games.apply(get_opp_win_pct, axis=1)
    
    # Average OppWinPct per team per season
    sos = games.groupby(['Season', 'TeamID'])['OppWinPct'].mean().reset_index()
    sos.rename(columns={'OppWinPct': 'SOS'}, inplace=True)
    
    # Merge back to team stats
    team_season_stats = pd.merge(team_season_stats, sos, on=['Season', 'TeamID'], how='left')
    
    # Merge Recent Stats
    team_season_stats = pd.merge(team_season_stats, recent_stats, on=['Season', 'TeamID'], how='left')

    # Fill NaNs specifically in recent stats if a team somehow played 0 games in last 30 days
    # (Optional: might be rare, but handles edge cases)
    recent_cols = [c for c in team_season_stats.columns if c.startswith('Recent_')]
    team_season_stats[recent_cols] = team_season_stats[recent_cols].fillna(0.0)

    return team_season_stats

def map_team_names(stats_df, teams_csv_path, format="Men"):
    """
    Add TeamName to the stats. Format determines some cleaning
    """
    print(f"Reading {teams_csv_path}...")
    teams_df = pd.read_csv(teams_csv_path)
    
    merged = pd.merge(stats_df, teams_df[['TeamID', 'TeamName']], on='TeamID', how='left')
    
    # Clean TeamName for merging with Cooper Ratings (Silver/Gold standard files)
    # The gold standard uses names like 'Duke', 'Michigan', 'UConn'
    # Kaggle uses 'Connecticut'. Let's do some common replacements.
    replace_dict = {
        'Connecticut': 'UConn',
        'Miami FL': 'U Miami (FL)',
        'N Carolina': 'North Carolina',
        'S Carolina': 'South Carolina',
        'St John\'s': 'St. John\'s',
        'Saint Joseph\'s': 'Saint Joseph\'s',
        'Saint Louis': 'Saint Louis',
        'Saint Mary\'s CA': 'Saint Mary\'s (CA)',
        'Rutgers': 'Rutgers',
        # Add more as needed based on analysis
    }
    
    merged['TeamName_Silver'] = merged['TeamName'].replace(replace_dict)
    
    # Add gender column for future ease of use
    merged['Gender'] = format
    
    return merged

if __name__ == "__main__":
    # Project root; data lives in march-machine-learning-mania-2026/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'march-machine-learning-mania-2026'))
    
    configurations = [
        {
            "gender": "Men",
            "reg_season_file": "MRegularSeasonDetailedResults.csv",
            "teams_file": "MTeams.csv",
            "out_file": "MRegularSeasonAggregatedStats.csv"
        },
        {
            "gender": "Women",
            "reg_season_file": "WRegularSeasonDetailedResults.csv",
            "teams_file": "WTeams.csv",
            "out_file": "WRegularSeasonAggregatedStats.csv"
        }
    ]
    
    for config in configurations:
        season_file = os.path.join(base_dir, config["reg_season_file"])
        teams_file = os.path.join(base_dir, config["teams_file"])
        out_path = os.path.join(base_dir, config["out_file"])
        
        if os.path.exists(season_file) and os.path.exists(teams_file):
            print(f"--- Processing {config['gender']}'s Data ---")
            stats = calculate_regular_season_stats(season_file)
            stats = map_team_names(stats, teams_file, config['gender'])
            stats.to_csv(out_path, index=False)
            print(f"Saved {config['gender']}'s aggregated stats to {out_path}\n")
        else:
            print(f"Missing required files for {config['gender']}.")
