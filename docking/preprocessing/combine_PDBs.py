import pandas as pd
import os
import glob
from pdbUtils import pdbUtils

# Get all protein files to extract IDs
protein_files = glob.glob("complex_chains/*_H3S1_model_0_chainAB.pdb")

for protein_file in protein_files:
    # Extract ID from filename
    ID = os.path.basename(protein_file).split('_')[0]
    
    # Define file paths
    ligpdb = f"complex_chains/{ID}_H3S1_gnina.pdb"
    protpdb = f"complex_chains/{ID}_H3S1_model_0_chainAB.pdb"
    
    try:
        protdf = pdbUtils.pdb2df(protpdb)
        ligdf = pdbUtils.pdb2df(ligpdb)
        
        merged_df = pd.concat([protdf, ligdf], axis=0)
        merged_df["Atom_ID"] = list(range(1, len(merged_df) + 1))
        merged_df.loc[merged_df["RES_NAME"] == "UNL", "RES_NAME"] = "SAM"
        merged_df.loc[merged_df["RES_NAME"] == "SAM", "CHAIN_ID"] = "C"
        
        pdbUtils.df2pdb(merged_df, f"complex_structures/{ID}_H3S1.pdb")
        print(f"Successfully processed {ID}")
        
    except Exception as e:
        print(f"Error processing {ID}: {e}")








