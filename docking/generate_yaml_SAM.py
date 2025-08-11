import pandas as pd
from Bio import SeqIO

# === User Inputs ===

df = pd.read_csv("candidate_list.csv")

# Create dictionary from CSV: second column as key, first column as value
id_map = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

print(f"[INFO] Created dictionary with {len(id_map)} entries from candidate_list.csv")

# Path to the FASTA file
fasta_path = "tryp_combined_cleaned.fasta"

# Ligand SMILES
ligand_smiles = "C[S+](CC[C@@H](C(=O)[O-])[NH3+])C[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)O"

# === Function to extract sequence ===
def extract_sequence(fasta_file, target_substring):
    matches = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        if target_substring in record.id or target_substring in record.description:
            matches.append(str(record.seq))
    if len(matches) == 0:
        raise ValueError(f"No match found for substring: {target_substring}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for substring: {target_substring}")
    print(f"[INFO] Found unique match for '{target_substring}'")
    return matches[0]


# === Manual YAML Writer ===
for yaml_id, fasta_substring in id_map.items():
    protein_sequence = extract_sequence(fasta_path, fasta_substring)

    yaml_content = f"""sequences:
  - protein:
      id: {yaml_id}
      sequence: {protein_sequence}
  - ligand:
      id: SAM
      smiles: '{ligand_smiles}'
properties:
  - affinity:
      binder: SAM
"""

    filename = f"SAM_yamls/{yaml_id}_SAM.yaml"
    with open(filename, "w") as f:
        f.write(yaml_content)

    print(f"[SUCCESS] Wrote YAML: {filename}")