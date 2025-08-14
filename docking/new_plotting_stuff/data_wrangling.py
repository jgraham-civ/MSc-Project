import pandas as pd
import os

# Import the CSV files
print("Importing CSV files...")
all_results_df = pd.read_csv("all_results.csv")
h3s1_redocking_df = pd.read_csv("H3S1_gnina_redocking_data.csv")
sam_redocking_df = pd.read_csv("SAM_gnina_redocking_data.csv")

print(f"all_results.csv shape: {all_results_df.shape}")
print(f"H3S1_gnina_redocking_data.csv shape: {h3s1_redocking_df.shape}")
print(f"SAM_gnina_redocking_data.csv shape: {sam_redocking_df.shape}")

# Drop the specified columns from all_results
columns_to_drop = ['Ser1N', 'Arg2NH1', 'Arg2NH2', 'Lys4NZ', 'NearestResidue', 'DistanceAngstrom']
print(f"\nDropping columns: {columns_to_drop}")
all_results_df = all_results_df.drop(columns=columns_to_drop, errors='ignore')

# Create a copy of the original dataframe for comparison
original_df = all_results_df.copy()

# Replace H3S1 redocking columns
print("\nReplacing H3S1 redocking columns...")
# Create a mapping dictionary from H3S1 redocking data
h3s1_mapping = dict(zip(h3s1_redocking_df['ligand'], 
                       zip(h3s1_redocking_df['vina_affinity'], 
                           h3s1_redocking_df['cnn_pose_score'])))

# Update the H3S1 redocking columns
for idx, row in all_results_df.iterrows():
    boltz_id = row['Boltz_ID']
    if boltz_id in h3s1_mapping:
        vina_score, cnn_score = h3s1_mapping[boltz_id]
        all_results_df.at[idx, 'Redocking_Vina_Score_H3'] = vina_score
        all_results_df.at[idx, 'Redocking_CNN_Pose_Score_H3'] = cnn_score

# Replace SAM redocking columns
print("Replacing SAM redocking columns...")
# Create a mapping dictionary from SAM redocking data
sam_mapping = dict(zip(sam_redocking_df['ligand'], 
                      zip(sam_redocking_df['vina_affinity'], 
                          sam_redocking_df['cnn_pose_score'])))

# Update the SAM redocking columns
for idx, row in all_results_df.iterrows():
    boltz_id = row['Boltz_ID']
    if boltz_id in sam_mapping:
        vina_score, cnn_score = sam_mapping[boltz_id]
        all_results_df.at[idx, 'Redocking_Vina_Score_SAM'] = vina_score
        all_results_df.at[idx, 'Redocking_CNN_Pose_Score_SAM'] = cnn_score

# Show summary of changes
print(f"\nSummary of changes:")
print(f"Original dataframe shape: {original_df.shape}")
print(f"Updated dataframe shape: {all_results_df.shape}")

# Count how many rows were updated
h3s1_updated = sum(1 for idx, row in all_results_df.iterrows() 
                   if row['Boltz_ID'] in h3s1_mapping)
sam_updated = sum(1 for idx, row in all_results_df.iterrows() 
                  if row['Boltz_ID'] in sam_mapping)

print(f"H3S1 redocking scores updated for {h3s1_updated} ligands")
print(f"SAM redocking scores updated for {sam_updated} ligands")

# Show some examples of the changes
print(f"\nExample changes (first 5 rows):")
print("Boltz_ID | Original H3S1 Vina | Updated H3S1 Vina | Original SAM Vina | Updated SAM Vina")
print("-" * 80)

for i in range(min(5, len(all_results_df))):
    row = all_results_df.iloc[i]
    orig_row = original_df.iloc[i]
    boltz_id = row['Boltz_ID']
    
    orig_h3s1_vina = orig_row['Redocking_Vina_Score_H3']
    new_h3s1_vina = row['Redocking_Vina_Score_H3']
    orig_sam_vina = orig_row['Redocking_Vina_Score_SAM']
    new_sam_vina = row['Redocking_Vina_Score_SAM']
    
    print(f"{boltz_id:8} | {orig_h3s1_vina:15.2f} | {new_h3s1_vina:15.2f} | {orig_sam_vina:15.2f} | {new_sam_vina:15.2f}")

# Save the updated dataframe
output_filename = "all_results_updated.csv"
all_results_df.to_csv(output_filename, index=False)
print(f"\nUpdated dataframe saved to: {output_filename}")

# Verify the data
print(f"\nVerification:")
print(f"Unique Boltz_IDs in all_results: {all_results_df['Boltz_ID'].nunique()}")
print(f"Unique ligands in H3S1 redocking: {h3s1_redocking_df['ligand'].nunique()}")
print(f"Unique ligands in SAM redocking: {sam_redocking_df['ligand'].nunique()}")

# Check for any missing matches
h3s1_missing = set(all_results_df['Boltz_ID']) - set(h3s1_redocking_df['ligand'])
sam_missing = set(all_results_df['Boltz_ID']) - set(sam_redocking_df['ligand'])

if h3s1_missing:
    print(f"Boltz_IDs missing from H3S1 redocking data: {sorted(h3s1_missing)}")
if sam_missing:
    print(f"Boltz_IDs missing from SAM redocking data: {sorted(sam_missing)}")







