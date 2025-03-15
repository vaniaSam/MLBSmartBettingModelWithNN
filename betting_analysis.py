"""
Finds +EV bets using either the Random Forest model or the Neural Network.
"""

from model import train_random_forest, train_neural_network


def find_profitable_bets(model_type="random_forest"):
    """
    Identifies profitable (+EV) bets by comparing model predictions with sportsbook odds.

    Parameters:
        model_type (str): Choose between "random_forest" and "neural_network".
    """
    if model_type == "random_forest":
        model, X_test, y_test, y_pred = train_random_forest()
    elif model_type == "neural_network":
        model, X_test, y_test, y_pred = train_neural_network()
    else:
        raise ValueError("Invalid model type. Choose 'random_forest' or 'neural_network'.")

    # Predict win probabilities
    if model_type == "random_forest":
        y_prob = model.predict_proba(X_test)[:, 1]
    else:  # Neural Network
        y_prob = model.predict(X_test)[:, 0]

    # Compare with sportsbook odds
    comparison_df = X_test.copy()
    comparison_df["Actual_Win"] = y_test
    comparison_df["Model_Prob"] = y_prob
    comparison_df["EV_Bet"] = comparison_df["Model_Prob"] > comparison_df["implied_prob"]

    # Display Profitable Bets
    profitable_bets = comparison_df[comparison_df["EV_Bet"] == True]
    print(profitable_bets[["Model_Prob", "implied_prob", "EV_Bet"]])
