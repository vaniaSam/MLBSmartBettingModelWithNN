# MLBSmartBettingModelWithNN
MLB Team Performance & Win Prediction

A data-driven approach to predicting MLB team win percentages using Machine Learning and Neural Networks.

Overview

This project analyzes MLB team performance statistics and predicts win percentages using Machine Learning (Random Forest) and Neural Networks (TensorFlow, Keras). We leverage 4,900+ game records, feature engineering, and sports analytics to identify key factors influencing team success.

Key Objectives:
✔ Predict MLB team win percentages using historical game data
✔ Analyze key performance indicators (offensive & defensive stats) affecting win rates
✔ Compare Random Forest and Neural Network performance for prediction accuracy
✔ Visualize findings through Power BI dashboards

Dataset

We used MLB team statistics containing 4,900+ records from past seasons, including:

Game Information: Team names, season details, and game results
Offensive Metrics: Runs scored, home runs, batting average
Defensive Metrics: ERA (Earned Run Average), strikeouts, walks
Win/Loss Records: Overall team performance per season
Approach & Methodology

1️ Data Preprocessing & Feature Engineering
✔ Removed missing values and structured JSON & tabular data into clean training datasets
✔ Engineered features like run differentials, ERA impact, and batting efficiency
✔ Normalized numerical features for better model performance

2️ Model Development
We built two predictive models:

 Random Forest Regression

✔ Achieved 93% accuracy (R² = 0.93) in predicting team win percentages
✔ Identified key features impacting team success, such as runs scored and ERA

Neural Network (TensorFlow & Keras)

✔ Built a deep learning model with 3 hidden layers
✔ Performed similarly to Random Forest but required longer training time
✔ Showed strong performance in capturing non-linear relationships

3️ Model Comparison & Insights
✔ Runs scored had a 0.87 correlation with team win percentage
✔ Teams with lower ERA (strong pitching) showed consistently higher win rates
✔ Random Forest provided slightly better interpretability, making it more useful for sports analysts
✔ Neural Networks handled complex feature interactions well but required careful hyperparameter tuning

Findings & Insights

 Key Takeaways from the Analysis:
✔ Scoring runs is the strongest predictor of wins (Teams with 5+ runs/game had a 70%+ win rate)
✔ Strong pitching is critical for consistent success (ERA below 3.50 resulted in higher playoff chances)
✔ Home-field advantage has a marginal impact (Win rates were ~3% higher for home teams)
✔ Neural Networks and Random Forest both performed well, but Random Forest provided better explainability
