import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/linear_regression.pkl"


def main():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model file not found.")
        print("Run this first: python3 main.py")
        return

    print("\n🏀 NBA TOTAL POINTS PREDICTOR")
    print("--------------------------")
    print("Enter the average statistics from each team's LAST 3 games.")
    print("All values are points per game.\n")

    home_scored = float(input("Home team - average points scored: "))
    home_allowed = float(input("Home team - average points allowed: "))
    away_scored = float(input("Away team - average points scored: "))
    away_allowed = float(input("Away team - average points allowed: "))

    X = pd.DataFrame(
    [[
        home_scored,
        home_allowed,
        away_scored,
        away_allowed
    ]],
    columns=[
        "home_avg_scored",
        "home_avg_allowed",
        "away_avg_scored",
        "away_avg_allowed"
    ]
)
    prediction = model.predict(X)[0]

    print("\n--------------------------")
    print(f"Predicted total points: {prediction:.1f}")
    print("--------------------------")
    print("Academic ML prediction only; not betting advice.")


if __name__ == "__main__":
    main()
