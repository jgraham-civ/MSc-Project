import pandas as pd

df = pd.read_csv("SAM_results/SAM_gnina_redocking_data.tsv", sep="\t")

# Group by ligand and select the row with the lowest (most negative) vina_affinity
top_affinities = df.loc[df.groupby('ligand')['vina_affinity'].idxmin()]

print("Original dataframe shape:", df.shape)
print("Top affinities dataframe shape:", top_affinities.shape)
print("\nTop affinities (one row per ligand):")
print(top_affinities)

# Save the filtered dataframe to a file
output_file = "SAM_top_affinities.tsv"
top_affinities.to_csv(output_file, sep="\t", index=False)
print(f"\nTop affinities saved to: {output_file}")