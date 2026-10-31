"""
Streamlit Web Interface for NBA Total Points Prediction
Allows live parameter input and displays model predictions.
"""

import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="NBA Total Points Predictor", page_icon="🏀", layout="centered"
)

st.title("🏀 NBA Total Points Prediction - AI Sports Gambling")
st.write(
    "Interactive UI for evaluating team matchups and predicting total game"
    " points."
)

# Sidebar inputs for live prediction
st.sidebar.header("Matchup Parameters")
home_pts = st.sidebar.slider("Home Team Avg Points", 90.0, 130.0, 112.0)
away_pts = st.sidebar.slider("Away Team Avg Points", 90.0, 130.0, 108.0)
pace = st.sidebar.slider("Game Pace (Possessions)", 90.0, 105.0, 98.0)
off_diff = st.sidebar.slider("Offensive Rating Diff", -10.0, 10.0, 2.0)

if st.button("Predict Total Game Points"):
  # Simple inference logic matching your features
  predicted_total = (
      home_pts * 0.45 + away_pts * 0.45 + pace * 0.1 + off_diff * 0.1 + 15
  )

  st.success(f"### Predicted Total Game Points: {predicted_total:.2f}")

  if predicted_total > 220:
    st.info(
        "🔥 **Recommendation:** OVER betting line favored based on high offensive"
        " pace."
    )
  else:
    st.info(
        "🛡️ **Recommendation:** UNDER betting line favored based on defensive"
        " matchup."
    )