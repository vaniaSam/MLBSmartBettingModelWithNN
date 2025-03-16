"""

Loads and preprocesses MLB team statistics and exports for Power BI.
"""

import pandas as pd

def load_data():
    """
    Loads MLB 2019 team statistics data and exports it for visualization.

    Returns:
        df (pd.DataFrame): Processed dataset.
    """
    # Load dataset
    df = pd.read_csv("2019teamstats.csv")

    # Select relevant stats
    df = df[["team", "b_r", "b_h", "b_hr", "p_er", "win", "loss"]]

    # Calculate win percentage
    df["win_percentage"] = df["win"] / (df["win"] + df["loss"])

    # Save cleaned data for Power BI
    df.to_csv("mlb_analysis_powerbi.csv", index=False)

    print("✅ Data exported to 'mlb_analysis_powerbi.csv' for Power BI.")
    return df

if __name__ == "__main__":
    load_data()
