# 2.0 K-Means and PCA

# Required packages
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



## Prepare data

"""
## Requirements

# Extract NBA Data
from nba_api.stats import endpoints
from nba_api.stats.endpoints import LeagueDashPlayerStats

# Dataframe, processing, stats and viz
import os 
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
from itertools import combinations
import unicodedata
import re


# Local Dir
os.chdir("C:/Data Projects/NBA")

##############################
# Player data extract pipeline
##############################

season = "2021-22"

def fetch_player_stats():
    player_stats = LeagueDashPlayerStats(season=season)
    df = player_stats.get_data_frames()[0]  # Convert API response to DataFrame
    return df

nba_raw = fetch_player_stats()
nba_raw.to_csv(f"data/season data/nba_{season}.csv", index=False)

# read from local dir
nba_raw = pd.read_csv(f"data/season data/nba_{season}.csv")


#####################################
## Preprocess / Subset viable players
#####################################

# Define GP and MPG subset thresholds 
minimum_games_quantile = 0.15
min_games = nba_raw['GP'].quantile(minimum_games_quantile)


minutes_per_game_threshold = 15
nba_raw['mpg'] = nba_raw['MIN'] / nba_raw['GP']


# Subset viable players
nba_subset = nba_raw[
    (nba_raw['GP'] >= min_games) &
    (nba_raw['mpg']>= minutes_per_game_threshold)
].copy()

# rename player name for parsimony 
nba_subset.rename(columns={'PLAYER_NAME': 'Player'}, inplace=True)

##########################################
# Calculate Per Game stats and concatenate
##########################################

# All categories used in scoring and/or analysis

raw_categories = ['GP', 'FGA', 'FGM', 'FTA', 'FTM', 'FG3M', 'PTS',
             'REB', 'AST', 'STL', 'BLK', 'TOV']

per_game_cats = [cat for cat in raw_categories if cat != 'GP']

meta_cols = ['Player', 'GP', 'FG_PCT', 'FT_PCT']

# Df includes: players, raw percentages, per-game stats
pg_stats = pd.concat([
    nba_subset[meta_cols],
    nba_subset[per_game_cats].div(nba_subset['GP'], axis=0),
], axis=1)


##########################################################
# Calculate Percentage Impact Scores 
###########################################################

# Get league averages to calculate deficits
FT_avg = pg_stats["FTM"].sum()/pg_stats["FTA"].sum()
FG_avg = pg_stats["FGM"].sum()/pg_stats["FGA"].sum()

# Calculate deficits
pg_stats["FT_def"] = pg_stats["FT_PCT"] - FT_avg
pg_stats["FG_def"] = pg_stats["FG_PCT"] - FG_avg

# Impact Method (Deficit x attempts standardized)
pg_stats["FT_impact"] = pg_stats["FT_def"] * pg_stats["FTA"]
pg_stats["FG_impact"] = pg_stats["FG_def"] * pg_stats["FGA"]


# Slope weighted Attempts Method (Mu + s(A-Mu))*%_def
FT_slope = .89
FG_slope = .93

FTA_mu = pg_stats['FTA'].mean()
FGA_mu = pg_stats['FGA'].mean()

pg_stats["FT_EffI"] = (FTA_mu + FT_slope*(pg_stats["FTA"] - FTA_mu)) * pg_stats['FT_def']
pg_stats["FG_EffI"] = (FGA_mu + FG_slope*(pg_stats["FGA"] - FGA_mu)) * pg_stats['FG_def']


##############################################
# Turnover Analysis
##############################################

model = smf.ols("TOV ~ FT_impact + FG_impact + FG3M + AST + REB + PTS + STL + BLK", data=pg_stats).fit()
print(model.summary())

pg_stats['TOV_resid'] = model.resid

pg_stats["TOV_effect"] = pg_stats['TOV'] + .2*pg_stats['TOV_resid']

#############################################################################
## Reverse code per-game Turnovers for different metrics
#############################################################################

pg_stats['tov'] = -pg_stats['TOV']
pg_stats['tov_effect'] = -pg_stats['TOV_effect']
"""



##############################
# Principle Component Analysis
##############################

# Category correlation matrix
R = pg_stats[scoring_cats].corr()

# 2. PCA on the correlation matrix (variables = categories)
pca_vars = PCA()
pca_vars.fit(R.values)

explained = pd.Series(
    pca_vars.explained_variance_ratio_,
    name="Variance Explained",
    index=[f"PC{i+1}" for i in range(len(scoring_cats))]
)

loadings = pd.DataFrame(
    pca_vars.components_.T,             # rows = categories
    index=R.index,                      # category names
    columns=[f"PC{i+1}" for i in range(len(scoring_cats))]
)

print("\n🔹 Variance Explained by Each Principal Component:")
print(explained.round(4))

print("\n🔹 Category Loadings (how each category loads onto each PC):")
print(loadings.round(4))


# Player level projection

X = Z(pg_stats[scoring_cats])   # rows = players, cols = z-scored stats

# Keep first 2 components for visualization / clustering
pca_players = PCA(n_components=2)
player_pcs = pca_players.fit_transform(X)

# Put PCs back into pg_stats
pg_stats['PC1_player'] = player_pcs[:, 0]
pg_stats['PC2_player'] = player_pcs[:, 1]

print("\n🔹 Player-level PCA (first 2 components):")
print(pca_players.explained_variance_ratio_.round(4))


#######################
# K Means Clustering
#######################

# 4. KMeans on players in PC1–PC2 space
k = 3  # guards / bigs / hybrids is a reasonable starting point
kmeans = KMeans(n_clusters=k, random_state=42)

pc_df = pg_stats[['PC1_player', 'PC2_player']]
pg_stats['cluster'] = kmeans.fit_predict(pc_df)

print("\n🔹 Cluster counts:")
print(pg_stats['cluster'].value_counts().sort_index())

cluster_means = pg_stats.groupby('cluster')[scoring_cats].mean().round(2)
print("\n🔹 Cluster mean stats (rough archetypes):")
print(cluster_means)







