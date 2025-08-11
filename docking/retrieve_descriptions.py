import pandas as pd
from Bio import SeqIO
from pathlib import Path

# Read candidate list
candidates = pd.read_csv("results/candidate_list.csv")
candidate_ids = candidates["Protein_ID"].tolist()

# Parse FASTA file and create ID mapping
protein_descriptions = {}
fasta_file = Path("~/pipeline/datasets/tryp_combined_cleaned.fasta").expanduser()

for record in SeqIO.parse(fasta_file, "fasta"):
    header = record.description
    
    # Extract description part (from first space to "OS=" if present)
    if ' ' in header:
        # Find the first space
        first_space_idx = header.find(' ')
        description_start = first_space_idx + 1
        
        # Find "OS=" if present
        os_idx = header.find(' OS=')
        if os_idx != -1:
            description = header[description_start:os_idx].strip()
        else:
            description = header[description_start:].strip()
    else:
        description = ""
    
    # Store by main ID and full header
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            main_id = parts[1]
            protein_descriptions[main_id] = description
            protein_descriptions[header] = description
    else:
        protein_descriptions[header] = description

# Match candidates to descriptions
results = []
unmatched = []

for candidate_id in candidate_ids:
    # Try exact match first
    if candidate_id in protein_descriptions:
        results.append({"Protein ID": candidate_id, "Description": protein_descriptions[candidate_id]})
        continue
    
    # Try matching by main ID part
    if '|' in candidate_id:
        main_id = candidate_id.split('|')[1]
        if main_id in protein_descriptions:
            results.append({"Protein ID": candidate_id, "Description": protein_descriptions[main_id]})
            continue
    
    unmatched.append(candidate_id)

print(unmatched)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results/protein_descriptions.csv", index=False)  # Default quoting - only quotes when needed

print(f"Matched {len(results)} proteins, {len(unmatched)} unmatched")
print(f"Results saved to: results/protein_descriptions.csv")