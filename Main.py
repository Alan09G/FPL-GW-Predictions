import numpy as np
import pandas as pd
import DataHandling as dh
import matplotlib.pyplot as plt
import seaborn as sns

merged_df = dh.get_dataset()

merged_df = merged_df.sort_values(['id', 'gw'])
merged_df['target_next_points'] = (
    merged_df.groupby('id')['event_points'].shift(-1)
)

rolling_features = dh.get_rolling_attributes(38)
merged_df = merged_df.merge(rolling_features, on=['id', 'gw'], how='left')

merged_df.dropna(subset=['target_next_points'], inplace=True)

null_columns = ['clean_sheets_per_90', 'goals_conceded_per_90', 'saves_per_90', 'defensive_contribution_per_90']
merged_df[null_columns] = merged_df[null_columns].fillna(0)

keepers_df = merged_df.loc[merged_df['position'] == 'Goalkeeper', dh.attributes_keepers]
defenders_df = merged_df.loc[merged_df['position'] == 'Defender', dh.attributes]
midfield_df = merged_df.loc[merged_df['position'] == 'Midfielder', dh.attributes]
forward_df = merged_df.loc[merged_df['position'] == 'Forward', dh.attributes_attackers]

print('Keeper Predictions changing: \n')
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
