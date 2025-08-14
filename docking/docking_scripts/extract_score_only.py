import os
import re
import pandas as pd

log_dir = "SAM_outputfiles"

# List to store parsed results
results = []

# Loop through sorted log files
for filename in sorted(os.listdir(log_dir)):
    if filename.endswith("scoreonly.log"):
        filepath = os.path.join(log_dir, filename)
        # Extract ligand ID from the beginning of filename up to first underscore
        ligand_id = filename.split('_')[0]  # e.g., 'AB' from 'AB_H3S1_gnina_scoreonly.log'

        with open(filepath, "r") as f:
            content = f.read()

        # Extract values using regex
        vina_aff = re.search(r"Affinity:\s*([-\d.]+)", content)
        cnn_score = re.search(r"CNNscore:\s*([-\d.]+)", content)
        cnn_aff = re.search(r"CNNaffinity:\s*([-\d.]+)", content)
        cnn_var = re.search(r"CNNvariance:\s*([-\d.]+)", content)
        intramol = re.search(r"Intramolecular energy:\s*([-\d.]+)", content)

        # Make sure all fields are present
        if all([vina_aff, cnn_score, cnn_aff, cnn_var, intramol]):
            results.append({
                "ligand": ligand_id,
                "vina_affinity": float(vina_aff.group(1)),
                "cnn_pose_score": float(cnn_score.group(1)),
                "cnn_affinity": float(cnn_aff.group(1)),
                "cnn_variance": float(cnn_var.group(1)),
                "intramolecular": float(intramol.group(1)),
            })
        else:
            print(f"Warning: Missing field(s) in {filename}")

# Create DataFrame
df = pd.DataFrame(results)

# Show or save
print(df)

df2 = pd.read_csv("SAM_results/boltz_SAM_probabilities_affinities.tsv", sep="\t")

df2["boltz_affinity_kcal"] = -((6 - df2["boltz_affinity"]) * 1.364)

print(df2)

results_df = pd.merge(df, df2, on="ligand", how="left")

print(results_df)

results_df.to_csv("SAM_results/SAM_scoreonly.tsv", sep="\t", index=False)

