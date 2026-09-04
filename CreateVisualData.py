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

def xg_xa_team_plot(df):

    df = df[['team_name', 'expected_goals', 'expected_assists']].copy()
    # Group the data by team name and calculate the mean of expected goals and assists
    df =df.groupby('team_name', as_index=False).mean()

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

    # Add labels (team names) to each data point
    for i, row in df.iterrows():
        plt.text(row['expected_goals'], row['expected_assists'], row['name'], fontsize=9, ha='right')
    
    plt.show()

def show_elo_rankings(team_df, gw):
    # Sort by elo   
    team_df = team_df[team_df['gameweek'] == gw].sort_values(by='team_elo', ascending=False)

    print("Elo Rankings:")

    for team in team_df['team_name']:
        elo = team_df.loc[team_df['team_name'] == team, 'team_elo'].values[0]
        print(f"{team}: {elo}")
