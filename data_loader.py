"""

Loads and preprocesses MLB betting odds and team performance data.
"""

import pandas as pd

def load_data():
    """
    Loads and merges betting odds with MLB game results and team stats.

    Returns:
        final_df (pd.DataFrame): Merged dataset.
    """
    # Load MLB betting odds
    odds_df = pd.read_csv("oddsData.csv")

    # Load Retrosheet data
    gameinfo_2019 = pd.read_csv("2019gameinfo.csv")
    teamstats_2019 = pd.read_csv("2019teamstats.csv")
    gameinfo_2020 = pd.read_csv("2020gameinfo.csv")
    teamstats_2020 = pd.read_csv("2020teamstats.csv")
    gameinfo_2021 = pd.read_csv("2021gameinfo.csv")
    teamstats_2021 = pd.read_csv("2021teamstats.csv")

    # Convert game dates (YYYYMMDD → YYYY-MM-DD)
    for df in [gameinfo_2019, gameinfo_2020, gameinfo_2021]:
        df["date"] = pd.to_datetime(df["date"], format='%Y%m%d').dt.strftime('%Y-%m-%d')

    # Standardize Team Names
    team_name_mapping = {"NYA": "NYY", "NYN": "NYM", "LAN": "LAD", "SFN": "SF", "CHA": "CHW", "CHN": "CHC"}
    for df in [gameinfo_2019, gameinfo_2020, gameinfo_2021, teamstats_2019, teamstats_2020, teamstats_2021]:
        df.replace({"team": team_name_mapping, "hometeam": team_name_mapping, "visteam": team_name_mapping}, inplace=True)

    # Merge datasets
    def merge_data(odds_df, gameinfo_df, teamstats_df):
        merged_df = odds_df.merge(gameinfo_df, left_on=['date', 'team'], right_on=['date', 'hometeam'], how='inner')
        merged_df = merged_df.merge(teamstats_df, on=['date', 'team'], how='inner')
        return merged_df

    merged_2019 = merge_data(odds_df, gameinfo_2019, teamstats_2019)
    merged_2020 = merge_data(odds_df, gameinfo_2020, teamstats_2020)
    merged_2021 = merge_data(odds_df, gameinfo_2021, teamstats_2021)

    # Combine all years
    final_df = pd.concat([merged_2019, merged_2020, merged_2021], ignore_index=True)

    return final_df
