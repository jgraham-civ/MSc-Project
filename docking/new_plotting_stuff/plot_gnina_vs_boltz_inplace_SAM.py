import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv("all_results.csv")

# Calculate Pearson r for all data
r, p = pearsonr(df["Vina_Score_SAM"], df["Boltz_Affinity_SAM"])

r2, p2 = pearsonr(df["Vina_Score_SAM"], df["Boltz_Probability_SAM"])
print(f"Pearson r for Vina vs Boltz Probability: {r2:.3f}")

# Plot Vina vs Boltz Affinity
plt.figure(figsize=(10, 5))

# Use diverging colormap with Boltz_Probability_H3
scatter = plt.scatter(df["Vina_Score_SAM"], df["Boltz_Affinity_SAM"], 
                     c=df["Boltz_Probability_SAM"], cmap='coolwarm_r', alpha=0.9, s=30,
                     vmin=0, vmax=1)

plt.xlabel("In Place Vina Affinity (kcal/mol)")
plt.ylabel("Boltz-2 Affinity (kcal/mol)")
plt.title("Boltz-2 vs. In PlaceVina Affinity Estimates [C + S]")
plt.grid(True)

# Identity line
x_range = [df["Vina_Score_SAM"].min(), df["Vina_Score_SAM"].max()]
plt.plot(x_range, x_range, 'r--', alpha=0.7, label='y = x')
plt.legend(loc="upper right")

# Set axis limits to start y-axis at 0 and maintain square aspect ratio
x_min, x_max = df["Vina_Score_SAM"].min(), df["Vina_Score_SAM"].max()
y_min, y_max = df["Boltz_Affinity_SAM"].min(), -4

# Add padding to prevent data points from being cut off
x_padding = (x_max - x_min) * 0.05
y_padding = abs(y_min) * 0.05

plt.xlim(x_min - x_padding, x_max + x_padding)
plt.ylim(y_min - y_padding, y_max + y_padding)
plt.gca().set_aspect('equal')

# Add colorbar
cbar = plt.colorbar(scatter, ax=plt.gca(), shrink=0.8, aspect=10)
cbar.set_label('Boltz-2 Binding Probability', rotation=270, labelpad=15)

# Annotate with Pearson r
plt.text(0.05, 0.87, f'Pearson r = {r:.3f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.savefig("gnina_vs_boltz_scoreonly.png")