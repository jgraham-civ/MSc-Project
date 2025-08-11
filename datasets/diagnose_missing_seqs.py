import os
from pathlib import Path
from Bio import SeqIO

# --- Paths ---
fasta_file = os.path.expandvars("${HOME}/pipeline/datasets/tryp_combined_cleaned.fasta")
embedding_dir = Path(os.path.expandvars("${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings_YES"))

# --- Load all .pt file stems ---
pt_stems = set(f.stem for f in embedding_dir.glob("*.pt"))

# --- Identify missing entries ---
missing = []

for record in SeqIO.parse(fasta_file, "fasta"):
    full_desc = record.description.strip()

    # 1. Try matching full header directly (used in many ESM outputs)
    if full_desc in pt_stems:
        continue

    # 2. Try reconstructed flattened key (remove slashes ONLY)
    header_suffix = full_desc[len(record.id):]
    flattened_key = (record.id + header_suffix).replace("/", "")

    if flattened_key not in pt_stems:
        missing.append(full_desc)

# --- Report ---
print(f"✅ Total sequences in FASTA: {len(list(SeqIO.parse(fasta_file, 'fasta')))}")
print(f"📦 Total .pt files found:     {len(pt_stems)}")
print(f"❌ Missing embeddings:        {len(missing)}")

if missing:
    print("\n🔍 Example missing entries:")
    for header in missing[:5]:
        print(f"- {header}")
else:
    print("\n🎉 All embeddings are accounted for!")

'''
missing_fasta = os.path.expandvars("${HOME}/pipeline/datasets/tryp_missing_sequences.fasta")

missing_set = set(missing)  # from your previous step

with open(fasta_file) as in_handle, open(missing_fasta, "w") as out_handle:
    for record in SeqIO.parse(in_handle, "fasta"):
        if record.description.strip() in missing_set:
            SeqIO.write(record, out_handle, "fasta")

print(f"Saved {len(missing_set)} missing sequences to:\n  {missing_fasta}")
'''