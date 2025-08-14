import pandas as pd

# Read both files
candidates = pd.read_csv("results/candidate_list.csv")
descriptions = pd.read_csv("results/protein_descriptions.csv")

print("Checking protein order...")
print(f"Candidate list has {len(candidates)} proteins")
print(f"Descriptions file has {len(descriptions)} proteins")
print()

# Check if they have the same number of entries
if len(candidates) != len(descriptions):
    print("❌ Different number of entries!")
    exit()

# Compare each protein ID
mismatches = []
for i in range(len(candidates)):
    candidate_id = candidates.iloc[i]["Protein_ID"]
    description_id = descriptions.iloc[i]["Protein ID"]
    
    if candidate_id != description_id:
        mismatches.append((i+1, candidate_id, description_id))

if mismatches:
    print("❌ Order mismatch found!")
    print("Row | Candidate List | Descriptions File")
    print("-" * 50)
    for row, candidate_id, description_id in mismatches:
        print(f"{row:3d} | {candidate_id} | {description_id}")
else:
    print("✅ All proteins are in the same order!")

print()
print("First 5 proteins for verification:")
print("Row | Candidate List | Descriptions File")
print("-" * 50)
for i in range(min(5, len(candidates))):
    candidate_id = candidates.iloc[i]["Protein_ID"]
    description_id = descriptions.iloc[i]["Protein ID"]
    print(f"{i+1:3d} | {candidate_id} | {description_id}") 