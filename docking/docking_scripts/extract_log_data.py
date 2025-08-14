import os
import re
import csv

# Set this to your log file directory
log_dir = "SAM_outputfiles"
output_tsv = "SAM_gnina_redocking_data.tsv"

data = []

# Match lines that start with mode number followed by five columns
line_re = re.compile(r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")

for filename in sorted(os.listdir(log_dir)):
    if filename.endswith("gnina.log"):
        filepath = os.path.join(log_dir, filename)
        # Extract ligand ID from filename (e.g., "AB" from "AB_H3S1_gnina.log")
        ligand_id = filename.split('_')[0]

        with open(filepath) as f:
            for line in f:
                match = line_re.match(line)
                if match:
                    mode, affinity, intramol, cnn_pose, cnn_aff = match.groups()
                    data.append({
                        "ligand": ligand_id,
                        "mode": int(mode),
                        "vina_affinity": float(affinity),
                        "intramolecular": float(intramol),
                        "cnn_pose_score": float(cnn_pose),
                        "cnn_affinity": float(cnn_aff)
                    })

# Write to TSV
with open(output_tsv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(data)

print(f"Wrote {len(data)} entries to {output_tsv}")