import pandas as pd
import scipy.signal
try:
    import scipy.signal.signaltools
except AttributeError:
    pass
if not hasattr(scipy.signal, '_centered') and not hasattr(scipy.signal, 'signaltools'):
    import scipy.signal as sigt
else:
    sigt = scipy.signal.signaltools if hasattr(scipy.signal, 'signaltools') else scipy.signal
if not hasattr(sigt, '_centered'):
    def _centered(arr, newsize):
        return arr
    sigt._centered = _centered
import statsmodels.api as sm

df = pd.read_csv('/Users/coopergilkey/Coding/March Madness/march-machine-learning-mania-2026/MRegularSeasonDetailedResults.csv')
df = df[df['Season'] == 2024]
df1 = df[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']].copy()
df1.columns = ['Season', 'T1_TeamID', 'T2_TeamID', 'T1_Score', 'T2_Score']

df2 = df[['Season', 'LTeamID', 'WTeamID', 'LScore', 'WScore']].copy()
df2.columns = ['Season', 'T1_TeamID', 'T2_TeamID', 'T1_Score', 'T2_Score']

games = pd.concat([df1, df2], ignore_index=True)
games['PointDiff'] = games['T1_Score'] - games['T2_Score']

games['T1_TeamID'] = 'T1_' + games['T1_TeamID'].astype(str)
games['T2_TeamID'] = 'T2_' + games['T2_TeamID'].astype(str)

formula = "PointDiff ~ -1 + T1_TeamID + T2_TeamID"
glm = sm.GLM.from_formula(formula=formula, data=games, family=sm.families.Gaussian()).fit()

quality = pd.DataFrame(glm.params).reset_index()
quality.columns = ["TeamID", "Quality"]
print(quality.head())
quality = quality[quality["TeamID"].str.contains("T1_")].reset_index(drop=True)
quality["TeamID"] = quality["TeamID"].str.extract(r'T1_(\d+)').astype(int)
print(quality.head())
