import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv("all_results.csv")

# Compute Pearson correlation coefficient
r, _ = pearsonr(df["Vina_Score_H3"], df["Vina_Score_SAM"])

plt.figure(figsize=(9, 4.5))
plt.scatter(df["Vina_Score_H3"], df["Vina_Score_SAM"], c="black", alpha = 0.7)
x_range = [df["Vina_Score_H3"].min(), df["Vina_Score_H3"].max()]
plt.plot(x_range, x_range, 'r--', alpha=0.7, label='y = x')
plt.xlabel("In Place Vina Affinity (Whole Complex) (kcal/mol)")
plt.ylabel("In Place Vina Affinity (Candidate Only) (kcal/mol)")
plt.title("Candidate Only vs. Whole Complex Vina Affinity Estimates")
plt.grid(True)

# Set axis limits to start y-axis at 0 and maintain square aspect ratio
x_min, x_max = df["Vina_Score_H3"].min(), df["Vina_Score_H3"].max()
y_min, y_max = df["Redocking_Vina_Score_H3"].min(), +1

# Add padding to prevent data points from being cut off
x_padding = (x_max - x_min) * 0.05
y_padding = abs(y_min) * 0.05

plt.xlim(x_min - x_padding, x_max + x_padding)
plt.ylim(y_min - y_padding, y_max + y_padding)
plt.gca().set_aspect('equal')

plt.legend(loc="upper right")
plt.text(0.05, 0.87, f'Pearson r = {r:.3f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.savefig("H3_vs_SAM_inplace.png")