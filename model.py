"""
Trains both a Random Forest model and a Neural Network to predict MLB game winners.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from data_loader import load_data


def preprocess_data():
    """
    Loads and preprocesses data for modeling.

    Returns:
        X_train, X_test, y_train, y_test (tuple): Train and test sets for features and target variable.
    """
    final_df = load_data()

    # Convert moneyline odds to implied probability
    def moneyline_to_probability(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    final_df["implied_prob"] = final_df["line_x"].apply(moneyline_to_probability)

    # Select Features
    feature_columns = ["implied_prob", "runLine", "runLineOdds", "total", "overOdds", "underOdds", "innings", "p_er",
                       "b_hr", "b_h", "b_w", "b_k"]
    final_df = final_df.dropna(subset=feature_columns)

    # Define Features & Target Variable
    X = final_df[feature_columns]
    y = final_df["win"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_random_forest():
    """
    Trains a Random Forest model.

    Returns:
        model (RandomForestClassifier): Trained Random Forest model.
        X_test, y_test, y_pred: Test set and predictions.
    """
    X_train, X_test, y_train, y_test = preprocess_data()

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Model Performance
    print(f"Random Forest Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred


def train_neural_network():
    """
    Trains a Neural Network model.

    Returns:
        model (Sequential): Trained neural network model.
        X_test, y_test, y_pred: Test set and predictions.
    """
    X_train, X_test, y_train, y_test = preprocess_data()

    # Scale Features for Neural Network
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build Neural Network Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (win or loss)
    ])

    # Compile Model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train Model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Predict
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Model Performance
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Neural Network Model Accuracy: {test_acc * 100:.2f}%")

    return model, X_test, y_test, y_pred
