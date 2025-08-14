import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data file
df = pd.read_csv("all_results_updated.csv")

# Create the horizontal violin plot
plt.figure(figsize=(10, 3))
violin_plot = sns.violinplot(x=df["Redocking_Vina_Score_SAM"], orient='h', color='lightgray', inner='box')
sns.stripplot(x=df["Redocking_Vina_Score_SAM"], orient='h', color='black', alpha=0.5, size=4, jitter=0.05)

# Customize the plot
plt.title("Distribution of Re-docking Vina Scores (SAM + candidate)", fontsize=10)
plt.xlabel("Vina Score (kcal/mol)", fontsize=10)

# Add grid for better readability
plt.grid(True, alpha=0.3, axis='x')

plt.axvline(x=-6, color='#E67066', linestyle='--', alpha=1)

# Adjust layout and save
plt.tight_layout()
plt.savefig("redocking_vina_scores_violin.png", dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Number of data points: {len(df['Redocking_Vina_Score_SAM'])}")
print(f"Mean: {df['Redocking_Vina_Score_SAM'].mean():.3f}")
print(f"Median: {df['Redocking_Vina_Score_SAM'].median():.3f}")
print(f"Standard deviation: {df['Redocking_Vina_Score_SAM'].std():.3f}")
print(f"Min: {df['Redocking_Vina_Score_SAM'].min():.3f}")
print(f"Max: {df['Redocking_Vina_Score_SAM'].max():.3f}")