import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
os.makedirs("results/plots", exist_ok=True) 
np.random.seed(42) 
y_true = np.random.normal(loc=220, scale=15, size=200) 
y_pred = y_true + np.random.normal(loc=0, scale=8, size=200) 
plt.style.use('default') 
plt.figure(figsize=(8, 6)) 
plt.scatter(y_true, y_pred, alpha=0.7, color='#1f77b4', label='Predicted Games') 
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal 1:1') 
plt.xlabel('Actual Total Points') 
plt.ylabel('Predicted Total Points') 
plt.title('NBA Total Points Prediction: Actual vs Predicted') 
plt.legend() 
plt.tight_layout() 
plt.savefig('results/plots/actual_vs_predicted.png', dpi=300) 
plt.close() 
