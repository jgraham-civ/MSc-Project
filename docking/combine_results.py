import pandas as pd
import os

resdir = os.path.expandvars("${HOME}/pipeline/docking/results")
scoreonly_df = pd.read_csv(os.path.join(resdir, "gnina_vs_boltz_scoreonly.tsv"), sep="\t")

scoreonly_df.rename(columns={"vina_affinity": "vina_affinity_scoreonly"}, inplace=True)
scoreonly_df.rename(columns={"cnn_pose_score": "cnn_pose_score_scoreonly"}, inplace=True)
scoreonly_df.rename(columns={"cnn_affinity": "cnn_affinity_scoreonly"}, inplace=True)
scoreonly_df.rename(columns={"cnn_variance": "cnn_var_scoreonly"}, inplace=True)
scoreonly_df.rename(columns={"intramolecular": "intramol_scoreonly"}, inplace=True)

redocking_df = pd.read_csv(os.path.join(resdir, "top_affinities.tsv"), sep="\t")

probability_df = pd.read_csv(os.path.join(resdir, "boltz_probabilities.tsv"), sep="\t")

# Merge the two dataframes on the ligand column
merged_df = pd.merge(scoreonly_df, redocking_df, on="ligand", how="left").merge(probability_df, on="ligand", how="left")

# Save the merged dataframe to a new CSV file
merged_df.to_csv(os.path.join(resdir, "affinities_results.csv"), index=False)








