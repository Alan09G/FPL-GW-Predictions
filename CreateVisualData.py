import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def xg_xa_player_plot(df):
    print(df.info())
    df = df[['web_name', 'expected_goals', 'expected_assists']].copy()
    # Group the data by player name and calculate the mean of expected goals and assists
    df =df.groupby('web_name', as_index=False).mean()

    #CREATE SCATTER PLOT
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='expected_goals', y='expected_assists')
    plt.title('Expected Goals vs Expected Assists')
    plt.xlabel('Expected Goals')
    plt.ylabel('Expected Assists')

    # Add mean lines for expected goals and assists
    plt.axvline(df['expected_goals'].mean(), color='red', linestyle='--', label='Mean Expected Goals')
    plt.axhline(df['expected_assists'].mean(), color='blue', linestyle='--', label='Mean Expected Assists')
    plt.legend()

    # Add labels (player names) to each data point
    for i, row in df.iterrows():
        plt.text(row['expected_goals'], row['expected_assists'], row['web_name'], fontsize=9, ha='right')
    
    plt.show()

def show_elo_rankings(team_df, gw):
    # Sort by elo   
    team_df = team_df[team_df['gameweek'] == gw].sort_values(by='team_elo', ascending=False)

    print("Elo Rankings:")

    for team in team_df['team_name']:
        elo = team_df.loc[team_df['team_name'] == team, 'team_elo'].values[0]
        print(f"{team}: {elo}")

def stats_per_gw(team_df, gw):

    fig, ax = plt.subplots(4, 5, figsize=(20, 15))

    for i, team in enumerate(team_df['team_name'].unique()):
        team_data = team_df[team_df['team_name'] == team]
        row, col = divmod(i, 5)
        ax[row, col].plot(team_data['gameweek'], team_data['expected_goals'], marker='o', label='Expected Goals')
        ax[row, col].set_title(team)
        ax[row, col].set_xlabel('Gameweek')
        ax[row, col].set_ylabel('Expected Goals')
        ax[row, col].legend()
        
        ax[row, col].set_xlim(1, (gw + 1))
        ax[row, col].set_ylim(0, team_data['expected_goals'].max() + 1)

    plt.tight_layout()
    plt.show()