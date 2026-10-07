import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://huggingface.co/datasets/davdsdfd/nba-games/resolve/main/games.csv"
DATA_PATH = "data/games.csv"


def load_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading NBA game data...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

    # Keep only completed games with the fields needed for this simple project.
    needed = [
        "GAME_DATE_EST", "GAME_ID", "TEAM_ID_home", "TEAM_ID_away",
        "PTS_home", "PTS_away"
    ]
    df = df[needed].dropna().copy()
    return df


def add_pregame_features(df, window=3):
    """
    For each game, use only information from games that happened BEFORE it.
    We calculate each team's average points scored and conceded in its last
    `window` games.
    """
    history = {}
    rows = []

    for _, game in df.iterrows():
        home = int(game["TEAM_ID_home"])
        away = int(game["TEAM_ID_away"])

        home_hist = history.get(home, [])
        away_hist = history.get(away, [])

        if len(home_hist) >= window and len(away_hist) >= window:
            home_recent = home_hist[-window:]
            away_recent = away_hist[-window:]

            rows.append({
                "date": game["GAME_DATE_EST"],
                "game_id": int(game["GAME_ID"]),
                "home_team": home,
                "away_team": away,
                "home_avg_scored": np.mean([x["scored"] for x in home_recent]),
                "home_avg_allowed": np.mean([x["allowed"] for x in home_recent]),
                "away_avg_scored": np.mean([x["scored"] for x in away_recent]),
                "away_avg_allowed": np.mean([x["allowed"] for x in away_recent]),
                "target_total_points": float(game["PTS_home"] + game["PTS_away"])
            })

        # IMPORTANT: update history only after creating the features.
        history.setdefault(home, []).append({
            "scored": float(game["PTS_home"]),
            "allowed": float(game["PTS_away"])
        })
        history.setdefault(away, []).append({
            "scored": float(game["PTS_away"]),
            "allowed": float(game["PTS_home"])
        })

    return pd.DataFrame(rows)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    return {
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
    }, model, pred


def main():
    os.makedirs("results", exist_ok=True)

    games = load_data()
    data = add_pregame_features(games, window=3)

    if len(data) < 100:
        raise ValueError("Not enough usable games after feature engineering.")

    # Chronological split: older games for training, newest 20% for testing.
    split = int(len(data) * 0.80)
    train = data.iloc[:split]
    test = data.iloc[split:]

    features = [
        "home_avg_scored",
        "home_avg_allowed",
        "away_avg_scored",
        "away_avg_allowed",
    ]

    X_train = train[features]
    y_train = train["target_total_points"]
    X_test = test[features]
    y_test = test["target_total_points"]

    models = [
        ("Linear Regression", LinearRegression()),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=10,
                n_jobs=-1,
            ),
        ),
    ]

    results = []
    predictions = test[["date", "game_id", "home_team", "away_team", "target_total_points"]].copy()

    for name, model in models:
        result, fitted_model, pred = evaluate_model(
            name, model, X_train, y_train, X_test, y_test
        )
        results.append(result)
        predictions[name.replace(" ", "_") + "_prediction"] = pred

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_results.csv", index=False)
    predictions.to_csv("results/predictions.csv", index=False)

    # Plot model comparison.
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.ylabel("RMSE (lower is better)")
    plt.title("NBA Total Points Prediction - Model Comparison")
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150)
    plt.close()

    # Plot actual vs Random Forest prediction.
    rf_pred = predictions["Random_Forest_prediction"]

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, rf_pred, alpha=0.45)
    min_v = min(y_test.min(), rf_pred.min())
    max_v = max(y_test.max(), rf_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Actual Total Points")
    plt.ylabel("Predicted Total Points")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted.png", dpi=150)
    plt.close()

    # Save a small summary text file.
    best = results_df.sort_values("RMSE").iloc[0]
    with open("results/summary.txt", "w", encoding="utf-8") as f:
        f.write("NBA TOTAL POINTS PREDICTION\n")
        f.write("===========================\n\n")
        f.write(f"Games used: {len(data)}\n")
        f.write(f"Training games: {len(train)}\n")
        f.write(f"Testing games: {len(test)}\n\n")
        f.write("Model results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nBest model by RMSE: ")
        f.write(f"{best['Model']} (RMSE = {best['RMSE']})\n")

    print("\nProject completed successfully!")
    print(results_df.to_string(index=False))
    print("\nFiles created inside results/:")
    print("- model_results.csv")
    print("- predictions.csv")
    print("- model_comparison.png")
    print("- actual_vs_predicted.png")
    print("- summary.txt")


if __name__ == "__main__":
    main()
