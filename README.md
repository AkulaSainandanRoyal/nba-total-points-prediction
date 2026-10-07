# 🏀 NBA Total Points Prediction

A simple Machine Learning project based on the paper **"The Bank is Open: AI in Sports Gambling"**.

## 🎯 Objective

Predict the **total number of points scored by both NBA teams in a game**.

Example:

> Lakers 112 + Celtics 108 = 220 total points

The model tries to predict this total **before the game**, using only information from the teams' previous games.

## 💡 Simple Approach

For every upcoming game, we calculate:

- Home team's average points scored in its last 3 games
- Home team's average points allowed in its last 3 games
- Away team's average points scored in its last 3 games
- Away team's average points allowed in its last 3 games

We then train two regression models:

1. **Linear Regression**
2. **Random Forest Regressor**

The data is split chronologically:

```text
Older 80% of games  → Training
Newest 20% of games → Testing
```

This avoids using future games to predict past games.

## 📁 Project Structure

```text
nba-total-points-project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── games.csv              # downloaded automatically
│
└── results/
    ├── model_results.csv
    ├── predictions.csv
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    └── summary.txt
```

## ▶️ How to Run

### 1. Install Python packages

```bash
pip install -r requirements.txt
```

### 2. Run the project

```bash
python main.py
```

The program automatically downloads the NBA games dataset the first time it is run.

## 📊 Output

After execution, the `results/` folder contains:

- `model_results.csv` — MAE, RMSE and R² for each model
- `predictions.csv` — actual and predicted game totals
- `model_comparison.png` — comparison of model RMSE
- `actual_vs_predicted.png` — Random Forest prediction plot
- `summary.txt` — short experiment summary

## 📏 Evaluation Metrics

### MAE
Average absolute prediction error in points.

### RMSE
Penalizes larger prediction errors more strongly. Lower is better.

### R²
Shows how much of the variation in total points is explained by the model. Higher is better.

## 📚 Reference

The project idea is based on:

**Alexandre Bucquet and Vishnu Sarukkai, "The Bank is Open: AI in Sports Gambling."**

The original paper studied NBA Over-Under prediction using Random Forest, Collaborative Filtering, Neural Networks and LSTM models.

## 🗃️ Dataset

The program uses the public `NBA Games Data` dataset available through Hugging Face. The dataset contains game dates, teams, scores and team-level box-score information.

The code downloads the dataset automatically, so the large raw dataset does not need to be manually uploaded to GitHub.

## ⚠️ Note

This is an academic ML project. Predictions are for demonstration and learning purposes and are **not financial or betting advice**.
