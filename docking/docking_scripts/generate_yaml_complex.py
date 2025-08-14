from Bio import SeqIO

# === User Inputs ===

# Dictionary mapping YAML ID -> FASTA header ID
id_map = {
    "ZZZ": "pdb|4V8M|Bs",
}



# Path to the FASTA file
fasta_path = "tryp_combined_cleaned.fasta"

# Fixed sequences
histone_H3_sequence = "SRTKETARTKKTITSKKSKKASKGSDAASGVKTAQRRWRPGTVALREIRQFQRSTDLLLQKAPFQRLVREVSGAQKEGLRFQSSAILAAQEATESYIVSLLADTNRACIHSGRVTIQPKDIHLALCLRGERA"
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
  - protein:
      id: H3
      sequence: {histone_H3_sequence}
  - ligand:
      id: SAM
      smiles: '{ligand_smiles}'
properties:
  - affinity:
      binder: SAM
"""

    filename = f"{yaml_id}_H3S1.yaml"
    with open(filename, "w") as f:
        f.write(yaml_content)

    print(f"[SUCCESS] Wrote YAML: {filename}")