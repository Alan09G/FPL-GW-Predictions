import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)

# URLs

gw_url = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/gameweek_summaries.csv"
players_url = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/players.csv"
player_stats_url = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/playerstats.csv"
team_url = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/teams.csv"

# get current gameweek
gw_df = pd.read_csv(gw_url)
current_gw = gw_df.loc[gw_df['is_current'] == True, 'id'].values[0]

#ATTRIBUTE MASKS

attributes = [
    'id', 
    'web_name',
    'event_points', 
    'target_next_points', 
    'bps_per_90', 
    'form',
    'now_cost',
    'points_per_game',
    'ict_index', 
    'gw',
    'minutes', 
    'rolling_minutes', 
    'rolling_total_points',
    'team_elo', 
    'opponent_elo', 
    'selected_by_percent'
]

attributes_outfield = attributes + [
    'goals_scored',
    'assists',
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements',
    'defensive_contribution_per_90',
    'rolling_expected_goals',
    'rolling_expected_assists'
]

attributes_def_mid = attributes_outfield + [
    'clean_sheets_per_90',
    'goals_conceded_per_90',
    'expected_goals_conceded',
    'rolling_clean_sheets',
    'rolling_goals_conceded',
]

attributes_keepers = attributes + [
    'rolling_saves',
    'rolling_clean_sheets',
    'rolling_goals_conceded',
    'clean_sheets_per_90', 
    'goals_conceded_per_90', 
    'penalties_saved',
    'saves_per_90'
]

attributes_teams = [
    "home_team_name",
    "away_team_name",
    "home_team_elo",
    "away_team_elo",
    "finished",
    "home_expected_goals_xg",
    "away_expected_goals_xg",
    "gameweek"
]

# PLAYER DATA  

def get_player_dataset():
    gw_df = pd.read_csv(gw_url)
    players_df = pd.read_csv(players_url)
    player_stats_df = pd.read_csv(player_stats_url)
    team_df = pd.read_csv(team_url)

    #print(player_stats_df.info())
    #print(players_df.info())
    #print(team_df.info())

    #add player position and team
    merged_df = player_stats_df.merge(
        players_df[['player_id', 'position', 'team_code']],
        left_on='id',
        right_on='player_id',
        how='left'
    )

    #Get team and opponent ELOs
    elo_dfs = []

    for gw in merged_df['gw'].unique():
        gw_fixtures = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/By%20Tournament/Premier%20League/GW{gw}/fixtures.csv"
        fixtures_df = pd.read_csv(gw_fixtures)
        fixtures_df['gw'] = gw

        # Home team perspective
        home_df = fixtures_df[[
            'gw',
            'home_team',
            'home_team_elo',
            'away_team_elo'
        ]].copy()

        home_df.rename(columns={
            'home_team': 'team_code',
            'home_team_elo': 'team_elo',
            'away_team_elo': 'opponent_elo'
        }, inplace=True)

        # Away team perspective
        away_df = fixtures_df[[
            'gw',
            'away_team',
            'away_team_elo',
            'home_team_elo'
        ]].copy()

        away_df.rename(columns={
            'away_team': 'team_code',
            'away_team_elo': 'team_elo',
            'home_team_elo': 'opponent_elo'
        }, inplace=True)

        fixture_elo_df = pd.concat([home_df, away_df], ignore_index=True)

        elo_dfs.append(fixture_elo_df)

    elo_df = pd.concat(elo_dfs, ignore_index=True)

    # Aggregate double gameweeks
    elo_df = elo_df.groupby(
        ['gw', 'team_code'],
        as_index=False
    ).agg(
        team_elo=('team_elo', 'mean'),
        opponent_elo=('opponent_elo', 'mean'),
        fixture_count=('opponent_elo', 'size')
    )

    merged_df = merged_df.merge(elo_df, on=['gw','team_code'], how='left')

    merged_df['bps_per_90'] = np.where(
        merged_df['minutes'] > 0,
        round(merged_df['bps'] / (merged_df['minutes']/90), 2),
        0
    )

    merged_df.dropna(subset=['team_elo'], inplace=True)

    return merged_df

def get_rolling_attributes():
    if current_gw < 1 or current_gw > 38:
        raise ValueError('GW must be between 1 and 38')
    features = ['id', 'minutes', 'total_points', 'goals_conceded', 'clean_sheets', 'saves', 'expected_goals', 'expected_assists', ]
    rolling_features = []
    for i in range(1, current_gw + 1):
        gw_info = f'https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2025-2026/By%20Tournament/Premier%20League/GW{i}/player_gameweek_stats.csv'
        stats_df = pd.read_csv(gw_info)
        stats_df = stats_df[features].copy()
        stats_df['gw'] = i
        rolling_features.append(stats_df)

    gw_features = pd.concat(rolling_features, ignore_index=True)
    gw_features = gw_features.sort_values(['id', 'gw'])

    rolling_features = ['minutes', 'total_points', 'goals_conceded', 'clean_sheets', 'saves', 'expected_goals', 'expected_assists']

    for col in rolling_features:
        gw_features[f'rolling_{col}'] = (
            gw_features.groupby('id')[col]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    rolling_feature_cols = [
        'id',
        'gw'
    ] + [f'rolling_{col}' for col in rolling_features]

    return gw_features[rolling_feature_cols]

# GET MATCH DATA

def get_match_dataset():
    match_df = []

    for gw in range(1, current_gw + 1):
        match_info = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/By%20Tournament/Premier%20League/GW{gw}/matches.csv"
        team_info = f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/refs/heads/main/data/2026-2027/By%20Tournament/Premier%20League/GW{gw}/teams.csv"
        gw_match_df = pd.read_csv(match_info)
        gw_team_df = pd.read_csv(team_info)[['code', 'name']]

        # add team names   
        gw_match_df['home_team_name'] = gw_match_df['home_team'].map(gw_team_df.set_index('code')['name'])
        gw_match_df['away_team_name'] = gw_match_df['away_team'].map(gw_team_df.set_index('code')['name'])
        match_df.append(gw_match_df)

    return pd.concat(match_df, ignore_index=True)

# TEAM DATA

def get_team_dataset(match_df: pd.DataFrame) -> pd.DataFrame:
    # create masks  
    home_team = ["home_team_name", "home_team_elo", "gameweek", "finished", "home_expected_goals_xg"]
    away_team = ["away_team_name", "away_team_elo", "gameweek", "finished", "away_expected_goals_xg"]

    #split the match dataset into home and away team dataframes, then rename columns to match
    # and concatenate them into a single team dataframe
    home_df = match_df[home_team].copy()
    home_df.rename(columns={
        "home_team_name": "team_name",
        "home_team_elo": "team_elo",
        "home_expected_goals_xg": "expected_goals"
    }, inplace=True)

    away_df = match_df[away_team].copy()
    away_df.rename(columns={
        "away_team_name": "team_name",
        "away_team_elo": "team_elo",
        "away_expected_goals_xg": "expected_goals"
    }, inplace=True)

    team_df = pd.concat([home_df, away_df], ignore_index=True)

    #fill in missing elo values  
    team_df['team_elo'] = team_df['team_elo'].fillna(0)
    
    #Account for double gameweeks   
    team_df = team_df.groupby(['team_name', 'team_elo', 'gameweek'], as_index=False).mean()

    return team_df




