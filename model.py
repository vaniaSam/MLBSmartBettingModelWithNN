"""
model.py

Trains an ML model to predict win percentages based on team stats.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from data_loader import load_data

def train_model():
    """
    Trains a Random Forest model to predict win percentage.

    Returns:
        model (RandomForestRegressor): Trained model.
        X_test, y_test, y_pred: Test set and predictions.
    """
    df = load_data()

    # Select features and target (Updated with correct column names)
    feature_columns = ["b_r", "b_h", "b_hr", "p_er"]
    df["win_percentage"] = df["win"] / (df["win"] + df["loss"])
    df = df.dropna(subset=feature_columns + ["win_percentage"])

    X = df[feature_columns]
    y = df["win_percentage"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Model Performance
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    train_model()
