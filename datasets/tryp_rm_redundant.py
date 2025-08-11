from Bio import SeqIO
import os

def remove_redundant(input_file, output_file):
    input_file = os.path.expandvars(input_file)
    total_records = 0
    seen = set()
    unique_records = []
    for record in SeqIO.parse(input_file, "fasta"):
        total_records += 1
        seq_str = str(record.seq)
        if seq_str not in seen:
            seen.add(seq_str)
            unique_records.append(record)
    # Write the non-redundant sequences to a new file
    SeqIO.write(unique_records, output_file, "fasta")

    return(print(f"Removed {total_records - len(unique_records)} duplicates. Kept {len(unique_records)} unique sequences."))

remove_redundant("${HOME}/pipeline/datasets/tryp_combined.fasta", "tryp_combined_cleaned.fasta")