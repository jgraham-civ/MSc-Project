import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr

# Read the data files
df = pd.read_csv("all_results.csv")

# Create figure with gridspec for more control over subplot sizes
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# First subplot: Score-only vs Redocking
r1, _ = pearsonr(df["Vina_Score_SAM"], df["Redocking_Vina_Score_SAM"])
print(f"Pearson r (Re-docking vs. In Place): {r1:.3f}")

ax1.scatter(df["Vina_Score_SAM"], df["Redocking_Vina_Score_SAM"], c="black", alpha=0.7)
x_range1 = [df["Redocking_Vina_Score_SAM"].min(), df["Vina_Score_SAM"].max()]
ax1.plot(x_range1, x_range1, 'r--', alpha=0.7, label='y = x')
ax1.set_xlabel("In Place Vina Affinity (kcal/mol)")
ax1.set_ylabel("Re-docking Vina Affinity (kcal/mol)")
ax1.set_title("Re-docking vs. In Place Vina Affinity Estimates [C + S]")
ax1.grid(True, alpha=0.3)

# Set axis limits for first subplot
x_min1, x_max1 = df["Vina_Score_SAM"].min(), df["Vina_Score_SAM"].max()
y_min1, y_max1 = df["Redocking_Vina_Score_SAM"].min(), -4

# Add padding to prevent data points from being cut off
x_padding1 = (x_max1 - x_min1) * 0.05
y_padding1 = abs(y_min1) * 0.05

ax1.set_xlim(-12, 1)
ax1.set_ylim(-12, 1)

ax1.legend(loc="upper left")

# Second subplot: SAM vs H3 in place (swapped axes)
r2, _ = pearsonr(df["Redocking_Vina_Score_SAM"], df["Redocking_Vina_Score_H3"])
print(f"Pearson r (SAM vs. SAM + H3): {r2:.3f}")

ax2.scatter(df["Redocking_Vina_Score_SAM"], df["Redocking_Vina_Score_H3"], c="black", alpha=0.7)
x_range2 = [df["Redocking_Vina_Score_SAM"].min(), df["Redocking_Vina_Score_SAM"].max()]
ax2.plot(x_range2, x_range2, 'r--', alpha=0.7, label='y = x')
ax2.set_xlabel("Re-docking Vina Affinity (kcal/mol) [C + S]")
ax2.set_ylabel("Re-docking Vina Affinity (kcal/mol) [C + S + H3]")
ax2.set_title("[C + S] vs. [C + S + H3] Re-docking Vina Affinity")
ax2.grid(True, alpha=0.3)

ax2.legend(loc="upper left")

# Set aspect ratio to 1:1 for first plot only
ax1.set_aspect('equal')

# Adjust layout and save
plt.tight_layout()
plt.savefig("docking_conditions.png", dpi=300, bbox_inches='tight')
plt.show()
