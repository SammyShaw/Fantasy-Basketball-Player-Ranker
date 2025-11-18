### NBA Player Ranker 2.0

## Requirements

# Extract NBA Data
from nba_api.stats import endpoints
from nba_api.stats.endpoints import LeagueDashPlayerStats

# Dataframe, processing, stats and viz
import os 
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

season = "2020-21"

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


#############################################################################
# Calculate Traditional Z-Score using Impact Scores
############################################################################

# Z score Function
def Z(stat):
    return (stat - stat.mean())/stat.std()

def Z_rank(df, categories, metric_label):
    df[metric_label] = Z(df[categories]).sum(axis=1) # rank and add the categories 
    df[metric_label + "_rank"] = df[metric_label].rank(ascending=False, method='min').reindex(df.index) # return rank order
    return df

######################
# Traditional Z Scores
######################
scoring_cats = ['FT_impact', 'FG_impact', 'PTS', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'tov']
pg_stats = Z_rank(pg_stats, scoring_cats, 'Traditional_Z')

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
"""

#############################################################################
### Shaw Rankings
#############################################################################

### Define Weights

shaw_weights = {
    'PTS':       1.15,
    'FG3M':      1.15,
    'REB':       0.55,
    'AST':       1.00,
    'STL':       1.35,
    'BLK':       0.65,
    'FT_impact': 1.00,
    'FG_impact': 1.00,
    'tov':       0.85,
}


shaw_z = Z(pg_stats[scoring_cats])

for cat in scoring_cats:
    shaw_z[cat] = shaw_z[cat] * shaw_weights[cat]

pg_stats[ [f"{c}_shaw_z" for c in scoring_cats] ] = shaw_z

# Clipped Z scores
def clippedZ(stat, lower=-3.8, upper=None):
    return stat.clip(lower=lower, upper=upper)

def clippedZ_rank(df, categories, metric_label):
    df[metric_label] = clippedZ(df[categories]).sum(axis=1)
    df[metric_label + "_rank"] = df[metric_label].rank(ascending=False, method='min').reindex(df.index)
    return df

pg_stats = clippedZ_rank(pg_stats, [f"{c}_shaw_z" for c in scoring_cats], "SHAW")



#######################
# Merge BBM table
#######################

# assume you already have: season = "2023-24" (or "2024-25", etc.)

def season_to_bbm_suffix(season: str) -> str:
    """Convert '2023-24' -> '23_24', '2024-25' -> '24_25'."""
    start2 = season[2:4]
    end2 = season[5:7]
    return f"{start2}_{end2}"

# Build path dynamically
bbm_suffix = season_to_bbm_suffix(season)
bbm_path = f"data/BBM rankings/BBM_PlayerRankings{bbm_suffix}.xls"

# Read BBM table for this season
bbm = pd.read_excel(bbm_path)

# select just name and rank
bbm = bbm[["Name", "Rank"]].copy()

# rename BBM_rank to distinguish from ranks in pg_stats
bbm.rename(columns={'Rank': 'BBM_rank'}, inplace=True)

# Clean names function
def remove_accents(text):
    if isinstance(text, str):
        return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    return text

# Clean names in BBM
bbm["Name"] = bbm["Name"].str.strip().apply(remove_accents)
bbm["Name"] = (
    bbm["Name"]
    .str.strip()
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.lower()
)

bbm["Name"] = bbm["Name"].replace({
    "alexandre sarr": "alex sarr",
    "cam johnson": "cameron johnson", 
    "derrick jones": "derrick jones jr",
    "gary trent": "gary trent jr",
    "herb jones": "herbert jones",
    "jimmy butler": "jimmy butler iii",
    "kelly oubre": "kelly oubre jr", 
    "lonnie walker": "lonnie walker iv",
    "nicolas claxton": "nic claxton", 
    "robert williams": "robert williams iii",
    "ron holland": "ronald holland ii",
    "trey murphy": "trey murphy iii",
    "cam payne": "cameron payne",
    "kevin knox": "kevin knox ii",
    "marcus morris": "marcus morris sr",
    "trey jemison": "trey jemison iii",
    "danuel house": "danuel house jr",
    "james ennis": "james ennis iii",
    "juan hernangomez": "juancho hernangomez", 
    "kira lewis": "kira lewis jr",
    "mo harkless": "maurice harkless",
    "david duke": "david duke jr",
    "duane washington": "duane washington jr",
    "josh primo": "joshua primo",
    "moe harkless": "maurice harkless"
})

# Clean names in original 
pg_stats['Player'] = pg_stats['Player'].str.strip().apply(remove_accents)
pg_stats['Player'] = (
    pg_stats['Player']
    .str.strip()
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.lower()
)

# Merge
pg_stats = pg_stats.merge(
    bbm[['BBM_rank', 'Name']],
    left_on='Player',
    right_on='Name',
    how='left'
)


# Check unmatched names
unmatched = pg_stats[pg_stats["BBM_rank"].isna()]
print(unmatched["Player"].sort_values().unique())

########################
# TOP N Teams Comparison
#########################



h2h = ['Traditional_Z_rank', 'SHAW_rank']


# top_n_list = [10, 20, 35, 50, 75, 100, 130]
top_n_list = list(range(1,151))

comp_stats = ['FG_PCT', 'FT_PCT', 'PTS', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'tov']


def generate_summary_dfs(df, rank_metrics, top_n_list, categories):

    summary_dfs = {}

    for n in top_n_list:
        summary_stats = {}

        for metric in rank_metrics:
            top_players = df.sort_values(by=metric).head(n)

            # Sum makes and attempts
            FGM = top_players['FGM'].sum()
            FGA = top_players['FGA'].sum()
            FTM = top_players['FTM'].sum()
            FTA = top_players['FTA'].sum()

            # Calculate percentages
            FG_PCT = FGM / FGA if FGA > 0 else 0
            FT_PCT = FTM / FTA if FTA > 0 else 0

            # Sum counting stats
            total_stats = top_players[['PTS', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'tov']].sum()

            # Add derived stats
            total_stats['FG_PCT'] = FG_PCT
            total_stats['FT_PCT'] = FT_PCT

            summary_stats[metric] = total_stats

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_stats).T
        summary_dfs[f'top_{n}'] = summary_df

    return summary_dfs



summary_dfs = generate_summary_dfs(
    df=pg_stats,
    rank_metrics=h2h,
    top_n_list=top_n_list,
    categories=comp_stats
)



def compare_summary_dfs(summary_dfs, categories):
    """
    Takes in summary_dfs and performs head-to-head matchups.
    Returns matchup results as a dictionary of Series showing win counts per metric.
    """
    matchup_results_by_top_n = {}

    for label, summary_df in summary_dfs.items():
        metrics = summary_df.index.tolist()
        
        # Add columns for storing results
        summary_df['Total_Category_Wins'] = 0
        summary_df['Total_Matchup_Wins'] = 0

        # Compare each metric against all others for Total_Category_Wins and Total_Matchup_Wins
        for i, m1 in enumerate(metrics):
            for m2 in metrics[i+1:]:
                team1 = summary_df.loc[m1]
                team2 = summary_df.loc[m2]

                m1_wins = 0
                m2_wins = 0

                for cat in categories:
                    if cat in summary_df.columns:
                        if team1[cat] > team2[cat]:
                            m1_wins += 1
                        elif team1[cat] < team2[cat]:
                            m2_wins += 1

                # Update total category wins
                summary_df.loc[m1, 'Total_Category_Wins'] += m1_wins
                summary_df.loc[m2, 'Total_Category_Wins'] += m2_wins

                # Update total matchup wins
                if m1_wins > m2_wins:
                    summary_df.loc[m1, 'Total_Matchup_Wins'] += 1
                elif m2_wins > m1_wins:
                    summary_df.loc[m2, 'Total_Matchup_Wins'] += 1
                else:
                    summary_df.loc[m1, 'Total_Matchup_Wins'] += 0.5
                    summary_df.loc[m2, 'Total_Matchup_Wins'] += 0.5

        matchup_results_by_top_n[label] = summary_df[['Total_Matchup_Wins', 'Total_Category_Wins']]

    return matchup_results_by_top_n


matchups = compare_summary_dfs(
    summary_dfs=summary_dfs,
    categories=comp_stats
)

for label, result in matchups.items():
    print(f"\n🏀 Head-to-head wins among metrics ({label}):")
    print(result)


import pandas as pd
import matplotlib.pyplot as plt

combined = []
for label, df in matchups.items():
    temp = df.copy()
    
    # keep track of which tier (top_10, top_20, etc.)
    temp["top_n"] = int(label.split("_")[-1])   # convert "top_10" → 10 (int)
    
    combined.append(temp)

combined_df = pd.concat(combined)

# Move the metric name (index) into a column
combined_df["metric"] = combined_df.index

# ---------------------------
# 1. Cumulative totals across ALL top-N tiers (what you already had)
# ---------------------------
cumulative_table = (
    combined_df
    .groupby("metric")[["Total_Matchup_Wins", "Total_Category_Wins"]]
    .sum()
    .sort_values("Total_Matchup_Wins", ascending=False)
)

print("\n🏀 Cumulative results across all top-N tiers:")
print(cumulative_table)

# ---------------------------
# 2. Running cumulative totals BY top N (for trendlines)
# ---------------------------

# Sort so cumsum happens in the correct order of top_n
combined_df = combined_df.sort_values(["metric", "top_n"])

combined_df[["Cum_Matchup_Wins", "Cum_Category_Wins"]] = (
    combined_df
    .groupby("metric")[["Total_Matchup_Wins", "Total_Category_Wins"]]
    .cumsum()
)


# ---------------------------
# 3. Plot cumulative matchup wins vs top N
# ---------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
ax1, ax2 = axes

color_map = {
    "Traditional_Z_rank": "tab:blue",
    "BBM_rank":           "tab:green",
    "SHAW_rank":          "tab:orange",
}


# -----------------------------
# Plot 1: Matchup Wins
# -----------------------------
for metric, group in combined_df.groupby("metric"):
    color = color_map.get(metric, None)  # None = fallback to default if missing
    ax1.plot(
        group["top_n"],
        group["Cum_Matchup_Wins"],
        marker="o",
        label=metric,
        color=color,
    )

    final_total = cumulative_table.loc[metric, "Total_Matchup_Wins"]
    last_x = group["top_n"].iloc[-1]
    last_y = group["Cum_Matchup_Wins"].iloc[-1]

    ax1.text(last_x + 1.5, last_y, f"{final_total:.0f}", va="center", fontsize=12)

ax1.set_title(f"{season} Cumulative Head-to-Head Matchup Wins")
ax1.set_xlabel("Top N")
ax1.set_ylabel("Cumulative Matchup Wins")
ax1.grid(True, alpha=0.3)

# -----------------------------
# Plot 2: Category Wins
# -----------------------------
for metric, group in combined_df.groupby("metric"):
    color = color_map.get(metric, None)
    ax2.plot(
        group["top_n"],
        group["Cum_Category_Wins"],
        marker="o",
        label=metric,
        color=color,
    )

    final_total = cumulative_table.loc[metric, "Total_Category_Wins"]
    last_x = group["top_n"].iloc[-1]
    last_y = group["Cum_Category_Wins"].iloc[-1]

    ax2.text(last_x + 1.5, last_y, f"{final_total:.0f}", va="center", fontsize=12)

ax2.set_title(f"{season} Cumulative Category Wins")
ax2.set_xlabel("Top N")
ax2.set_ylabel("Cumulative Category Wins")
ax2.grid(True, alpha=0.3)

# Combined legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, title="Metric", loc="lower center", ncol=3)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leaves space for legend
plt.show()

