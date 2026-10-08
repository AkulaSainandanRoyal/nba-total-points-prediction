import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_URL = "https://huggingface.co/datasets/davdsdfd/nba-games/resolve/main/games.csv"
DATA_PATH = "data/games.csv"


def load_data():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Downloading NBA dataset...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    else:
        print("Loading existing NBA dataset...")

    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    # Select the required columns
    df = df[
        [
            "GAME_DATE_EST",
            "TEAM_ID_home",
            "TEAM_ID_away",
            "PTS_home",
            "PTS_away",
        ]
    ].copy()

    # Remove games with missing values
    df = df.dropna(
        subset=[
            "GAME_DATE_EST",
            "TEAM_ID_home",
            "TEAM_ID_away",
            "PTS_home",
            "PTS_away",
        ]
    )

    # Convert date to datetime
    df["GAME_DATE_EST"] = pd.to_datetime(
        df["GAME_DATE_EST"]
    )

    # Sort games chronologically
    df = df.sort_values(
        "GAME_DATE_EST"
    ).reset_index(drop=True)

    # Total points scored in each game
    df["total_points"] = (
        df["PTS_home"] + df["PTS_away"]
    )

    # Store previous games for every team
    team_history = {}

    features = []
    targets = []

    for _, row in df.iterrows():

        home_team = row["TEAM_ID_home"]
        away_team = row["TEAM_ID_away"]

        # Previous games for both teams
        home_history = team_history.get(
            home_team,
            []
        )

        away_history = team_history.get(
            away_team,
            []
        )

        # Need at least 3 previous games
        # for both teams
        if (
            len(home_history) >= 3
            and len(away_history) >= 3
        ):

            home_last3 = home_history[-3:]
            away_last3 = away_history[-3:]

            # Home team's average points scored
            home_scored = np.mean(
                [
                    game["scored"]
                    for game in home_last3
                ]
            )

            # Home team's average points allowed
            home_allowed = np.mean(
                [
                    game["allowed"]
                    for game in home_last3
                ]
            )

            # Away team's average points scored
            away_scored = np.mean(
                [
                    game["scored"]
                    for game in away_last3
                ]
            )

            # Away team's average points allowed
            away_allowed = np.mean(
                [
                    game["allowed"]
                    for game in away_last3
                ]
            )

            features.append(
                [
                    home_scored,
                    home_allowed,
                    away_scored,
                    away_allowed,
                ]
            )

            targets.append(
                row["total_points"]
            )

        # Update home team's history
        team_history.setdefault(
            home_team,
            []
        ).append(
            {
                "scored": row["PTS_home"],
                "allowed": row["PTS_away"],
            }
        )

        # Update away team's history
        team_history.setdefault(
            away_team,
            []
        ).append(
            {
                "scored": row["PTS_away"],
                "allowed": row["PTS_home"],
            }
        )

    # Create feature DataFrame
    X = pd.DataFrame(
        features,
        columns=[
            "home_avg_scored",
            "home_avg_allowed",
            "away_avg_scored",
            "away_avg_allowed",
        ],
    )

    # Target variable
    y = pd.Series(
        targets,
        name="total_points"
    )

    # Final safety check
    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    return X, y

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Make predictions
    predictions = model.predict(
        X_test
    )

    # Calculate metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    result = {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return (
        result,
        model,
        predictions
    )


def main():

    print(
        "\n🏀 NBA TOTAL POINTS PREDICTION"
    )
    print(
        "==============================\n"
    )

    # -----------------------------------------
    # 1. Load dataset
    # -----------------------------------------

    df = load_data()

    print(
        f"Total games in dataset: {len(df)}"
    )

    # -----------------------------------------
    # 2. Prepare data
    # -----------------------------------------

    X, y = prepare_data(df)

    print(
        f"Games usable for ML: {len(X)}"
    )

    # -----------------------------------------
    # 3. Chronological train-test split
    # -----------------------------------------

    split_index = int(
        len(X) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print(
        f"Training games: {len(X_train)}"
    )

    print(
        f"Testing games: {len(X_test)}"
    )

    # -----------------------------------------
    # 4. Define models
    # -----------------------------------------

    models = [
        (
            "Linear Regression",
            LinearRegression()
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]

    # -----------------------------------------
    # 5. Train and evaluate models
    # -----------------------------------------

    results = []
    predictions = {}
    fitted_models = {}

    for name, model in models:

        result, fitted_model, pred = (
            evaluate_model(
                name,
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )
        )

        results.append(result)

        fitted_models[name] = (
            fitted_model
        )

        predictions[
            name.replace(
                " ",
                "_"
            ) + "_prediction"
        ] = pred

    # -----------------------------------------
    # 6. Save Linear Regression model
    # -----------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        fitted_models[
            "Linear Regression"
        ],
        "models/linear_regression.pkl"
    )

    print(
        "\nSaved trained model:"
    )

    print(
        "models/linear_regression.pkl"
    )

    # -----------------------------------------
    # 7. Save model results
    # -----------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "results/model_results.csv",
        index=False
    )

    # -----------------------------------------
    # 8. Save predictions
    # -----------------------------------------

    prediction_df = pd.DataFrame(
        {
            "Actual_Total_Points":
                y_test.values,
            **predictions
        }
    )

    prediction_df.to_csv(
        "results/predictions.csv",
        index=False
    )

    # -----------------------------------------
    # 9. Model comparison graph
    # -----------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        results_df["Model"],
        results_df["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")

    plt.title(
        "Model Comparison - RMSE"
    )

    plt.tight_layout()

    plt.savefig(
        "results/model_comparison.png"
    )

    plt.close()

    # -----------------------------------------
    # 10. Actual vs Predicted graph
    # -----------------------------------------

    best_model_name = (
        results_df.loc[
            results_df["RMSE"].idxmin(),
            "Model"
        ]
    )

    best_prediction_column = (
        best_model_name.replace(
            " ",
            "_"
        )
        + "_prediction"
    )

    best_predictions = predictions[
        best_prediction_column
    ]

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        y_test,
        best_predictions,
        alpha=0.3
    )

    min_value = min(
        y_test.min(),
        best_predictions.min()
    )

    max_value = max(
        y_test.max(),
        best_predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel(
        "Actual Total Points"
    )

    plt.ylabel(
        "Predicted Total Points"
    )

    plt.title(
        f"Actual vs Predicted - "
        f"{best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        "results/actual_vs_predicted.png"
    )

    plt.close()

    # -----------------------------------------
    # 11. Save summary
    # -----------------------------------------

    with open(
        "results/summary.txt",
        "w"
    ) as file:

        file.write(
            "NBA Total Points Prediction\n"
        )

        file.write(
            "============================\n\n"
        )

        file.write(
            f"Games used: {len(X)}\n"
        )

        file.write(
            f"Training games: "
            f"{len(X_train)}\n"
        )

        file.write(
            f"Testing games: "
            f"{len(X_test)}\n\n"
        )

        file.write(
            results_df.to_string(
                index=False
            )
        )

        file.write(
            f"\n\nBest model by RMSE: "
            f"{best_model_name}\n"
        )

    # -----------------------------------------
    # 12. Display results
    # -----------------------------------------

    print("\nMODEL RESULTS")
    print("=============")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nBest model by RMSE: "
        f"{best_model_name}"
    )

    print("\nFiles created:")

    print(
        "  results/model_results.csv"
    )

    print(
        "  results/predictions.csv"
    )

    print(
        "  results/model_comparison.png"
    )

    print(
        "  results/actual_vs_predicted.png"
    )

    print(
        "  results/summary.txt"
    )

    print(
        "  models/linear_regression.pkl"
    )

    print(
        "\n✅ Project completed successfully!"
    )


if __name__ == "__main__":
    main()
