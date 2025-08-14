import pandas as pd
import os

# Import the CSV files
print("Importing CSV files...")
all_results_df = pd.read_csv("all_results_updated.csv")
distances_df = pd.read_csv("nearest_H3_distances.csv")

# Merge the dataframes on the Boltz_ID column
merged_df = pd.merge(all_results_df, distances_df, on="Boltz_ID", how="left")

# Save the merged dataframe to a new CSV file
merged_df.to_csv("all_results.csv", index=False)

print("Merged dataframe saved to all_results.csv")