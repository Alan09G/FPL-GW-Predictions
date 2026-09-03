import DataHandling as dh
import CreateVisualData as create

#GET PLAYER DATASET
 
player_df = dh.get_player_dataset()

# sort the dataset by id and gw
player_df = player_df.sort_values(['id', 'gw'])
# create target attribute for next gameweek points for each gw
player_df['target_next_points'] = (
    player_df.groupby('id')['event_points'].shift(-1)
)

# Get rolling attributes for each player up to the current gameweek
rolling_features = dh.get_rolling_attributes()
player_df = player_df.merge(rolling_features, on=['id', 'gw'], how='left')

# drop players with less than minutes played threshold 
player_df = player_df[player_df['minutes'] >= (dh.current_gw- 1) * 45] # 45 minutes per gameweek

# drop rows with missing target_next_points values
player_df.dropna(subset=['target_next_points'], inplace=True)

#fill na values for rolling features with 0
null_columns = ['clean_sheets_per_90', 'goals_conceded_per_90', 'saves_per_90', 'defensive_contribution_per_90']
player_df[null_columns] = player_df[null_columns].fillna(0)

#print(*list(merged_df.columns), sep='\n')
#print(merged_df.info())

# seperate the dataset by position
keepers_df = player_df.loc[player_df['position'] == 'Goalkeeper', dh.attributes_keepers]
defenders_df = player_df.loc[player_df['position'] == 'Defender', dh.attributes_def_mid]
midfield_df = player_df.loc[player_df['position'] == 'Midfielder', dh.attributes_def_mid]
forward_df = player_df.loc[player_df['position'] == 'Forward', dh.attributes_outfield] 

#create.expected_goals_assists_plot(forward_df)   
#create.expected_goals_assists_plot(midfield_df)
#create.expected_goals_assists_plot(defenders_df)

#GET MATCH AND TEAM DATASET

match_df = dh.get_match_dataset()
match_df = match_df[dh.attributes_teams]
team_df = dh.get_team_dataset(match_df)
create.xg_xa_team_plot(team_df)
    

# TESTING PREDICTION MODELS
"""print('Keeper Predictions changing: \n')
print('Random Forest: \n', dh.random_forest(keepers_df))
print('XGBoost: \n', dh.xgboost(keepers_df))

print('============================================')

print('Defender Predictions: \n')
print("Random Forest: \n", dh.random_forest(defenders_df))
print('XGBoost: \n', dh.xgboost(defenders_df))

print('=============================================')
print('Mid Field Predictions: \n')
print('Random Forest: \n', dh.random_forest(midfield_df))
print('XGBoost: \n', dh.xgboost(midfield_df))

print('=============================================')
print('Attacker Predictions: \n')
print('Random Forest: \n', dh.random_forest(forward_df))
print('XGBoost: \n', dh.xgboost(forward_df))
"""
