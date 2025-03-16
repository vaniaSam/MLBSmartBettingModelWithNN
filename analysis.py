"""
analysis.py

Analyzes MLB team performance using statistics data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data

def analyze_team_performance():
    """
    Analyzes team statistics to identify top-performing teams.
    """
    df = load_data()

    # Check available columns
    print("Available columns:", df.columns)

    # Select relevant stats (Updated with correct column names)
    relevant_columns = ["team", "b_r", "b_h", "b_hr", "win", "loss"]
    df = df[relevant_columns]

    # Calculate win percentage
    df["win_percentage"] = df["win"] / (df["win"] + df["loss"])

    # Sort teams by runs scored
    top_scoring_teams = df.sort_values(by="b_r", ascending=False).head(10)
    print("\nTop 10 teams by runs scored:")
    print(top_scoring_teams)

    # Plot top scoring teams
    plt.figure(figsize=(10, 5))
    plt.bar(top_scoring_teams["team"], top_scoring_teams["b_r"], color="blue")
    plt.xlabel("Team")
    plt.ylabel("Total Runs")
    plt.title("Top 10 MLB Teams by Runs Scored (2019)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    analyze_team_performance()
