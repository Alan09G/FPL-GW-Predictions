import DataHandling as dh

CURRENT_GW = 38

# gets unfiltered dataset  
merged_df = dh.get_dataset()

# sort the dataset by id and gw
merged_df = merged_df.sort_values(['id', 'gw'])
# create target attribute for next gameweek points for each gw
merged_df['target_next_points'] = (
    merged_df.groupby('id')['event_points'].shift(-1)
)

# Get rolling attributes for each player up to the current gameweek
rolling_features = dh.get_rolling_attributes(CURRENT_GW)
merged_df = merged_df.merge(rolling_features, on=['id', 'gw'], how='left')

# drop rows with missing target_next_points values
merged_df.dropna(subset=['target_next_points'], inplace=True)

#fill na values for rolling features with 0
null_columns = ['clean_sheets_per_90', 'goals_conceded_per_90', 'saves_per_90', 'defensive_contribution_per_90']
merged_df[null_columns] = merged_df[null_columns].fillna(0)

print(*list(merged_df.columns), sep='\n')

# seperate the dataset by position
keepers_df = merged_df.loc[merged_df['position'] == 'Goalkeeper', dh.attributes_keepers]
defenders_df = merged_df.loc[merged_df['position'] == 'Defender', dh.attributes_def_mid]
midfield_df = merged_df.loc[merged_df['position'] == 'Midfielder', dh.attributes_def_mid]
forward_df = merged_df.loc[merged_df['position'] == 'Forward', dh.attributes_outfield] 


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
