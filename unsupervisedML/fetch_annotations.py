import os
import torch
import numpy as np
import requests
import re
import time
import pandas as pd
from pathlib import Path

def extract_uniprot_id(header):
    """Extract UniProt ID from FASTA-style header"""
    if not isinstance(header, str):
        print(f"Warning: header is not a string: {header}")
        return None
    
    # Extract UniProt ID from sp|XXXXX| or tr|XXXXX| format
    uniprot_match = re.match(r'^(?:sp|tr)\|([A-Z0-9]+)\|', header)
    if uniprot_match:
        return uniprot_match.group(1)
    
    print(f"Could not extract UniProt ID from header: {header}")
    return None

def get_protein_annotations(uniprot_id):
    """Fetch molecular function GO terms and EC numbers for a UniProt ID"""
    if not uniprot_id:
        return None, None
    print(f"Fetching annotations for UniProt ID: {uniprot_id}")
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    try:
        response = requests.get(url)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            
            # Get GO terms
            go_terms = []
            if 'uniProtKBCrossReferences' in data:
                for ref in data['uniProtKBCrossReferences']:
                    if ref['database'] == 'GO':
                        for prop in ref.get('properties', []):
                            val = prop.get('value', '')
                            if prop.get('key') == 'GoTerm' and val.startswith("F:"):
                                go_terms.append(val)
            
            # Get EC numbers from the correct path
            ec_numbers = []
            if 'proteinDescription' in data:
                recommended_name = data['proteinDescription'].get('recommendedName', {})
                if 'ecNumbers' in recommended_name:
                    ec_numbers = [ec['value'] for ec in recommended_name['ecNumbers']]
            
            # Get the most specific GO term (if any)
            go_term = max(go_terms, key=lambda x: len(x)) if go_terms else None
            # Join EC numbers with semicolon if multiple exist
            ec_number = "; ".join(ec_numbers) if ec_numbers else None
            
            print(f"Found GO term: {go_term}")
            print(f"Found EC numbers: {ec_number}")
            return go_term, ec_number
            
        return None, None
    except Exception as e:
        print(f"Error fetching annotations: {str(e)}")
        return None, None
    finally:
        time.sleep(0.1)  # Rate limiting

def find_pt_files(root_dir):
    """Find all .pt files, handling split filenames correctly"""
    root_dir = os.path.expandvars(root_dir)
    pt_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            # Handle direct .pt files in root
            for f in filenames:
                if f.endswith('.pt'):
                    pt_files.append(os.path.join(dirpath, f))
        else:
            # Handle split filenames in subdirectories
            base_dir = os.path.basename(dirpath)
            for f in filenames:
                if f.endswith('.pt'):
                    # Combine directory name and filename
                    full_name = base_dir + f
                    pt_files.append(os.path.join(dirpath, f))
    
    print(f"Found {len(pt_files)} .pt files in {root_dir}")
    return pt_files

def load_embeddings(all_pt_files):
    """Load embeddings and reconstruct full headers from file paths"""
    embeddings = []
    protein_ids = []
    
    for file in all_pt_files:
        data = torch.load(file, map_location="cpu")
        
        # Get the full path relative to the embeddings directory
        dir_name = os.path.dirname(file)
        base_name = os.path.basename(file)
        
        if dir_name.endswith('_esm1b_embeddings'):
            # Direct .pt file in root
            header = base_name[:-3]  # Remove .pt
        else:
            # Split filename
            dir_part = os.path.basename(dir_name)
            file_part = base_name[:-3]  # Remove .pt
            header = dir_part + file_part
            
        # Extract the UniProt ID part (sp|XXXXX|)
        uniprot_part = header.split('|')
        if len(uniprot_part) >= 3:
            protein_id = f"{uniprot_part[0]}|{uniprot_part[1]}|{uniprot_part[2].split()[0]}"
        else:
            protein_id = header
            
        protein_ids.append(protein_id)
        mean_embedding = data["mean_representations"][33]
        embeddings.append(mean_embedding.numpy())
    
    output = np.vstack(embeddings)
    return output, protein_ids

def save_results(protein_ids, annotations, dataset_name, output_dir="go_terms"):
    """Save results to TSV file with protein IDs, UniProt IDs, GO terms, and EC numbers"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create a list of dictionaries for the DataFrame
    results = []
    for protein_id in protein_ids:
        uniprot_id = extract_uniprot_id(protein_id)
        protein_data = annotations.get(protein_id, {})
        results.append({
            'protein_id': protein_id,
            'uniprot_id': uniprot_id,
            'go_term': protein_data.get('go_term'),
            'ec_number': protein_data.get('ec_number'),
            'dataset': dataset_name
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    output_file = Path(output_dir) / f"{dataset_name}_annotations.tsv"
    df.to_csv(output_file, sep='\t', index=False)
    
    # Print statistics
    print(f"\nStatistics for {dataset_name}:")
    print(f"Total proteins: {len(df)}")
    print(f"Proteins with GO terms: {df['go_term'].notna().sum()}")
    print(f"Proteins with EC numbers: {df['ec_number'].notna().sum()}")
    print(f"Results saved to {output_file}")
    
    return df

## Execute pipeline

ref_dirs = {"tier1":"${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings", 
           "tier2":"${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm1b_embeddings", 
           "tier3":"${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings", 
           "tier4":"${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm1b_embeddings"}
tryp_dir = "${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings"

# Process trypanosome proteins
print("Finding tryp .pt files...")
tryp_pt_files = find_pt_files(tryp_dir)
print("Loading tryp embeddings...")
tryp_embeddings, tryp_protein_ids = load_embeddings(tryp_pt_files)

# Get annotations for trypanosome proteins
print("Fetching annotations for trypanosome proteins...")
annotations = {}
for protein_id in tryp_protein_ids:
    uniprot_id = extract_uniprot_id(protein_id)
    if uniprot_id:
        go_term, ec_number = get_protein_annotations(uniprot_id)
        annotations[protein_id] = {'go_term': go_term, 'ec_number': ec_number}
    else:
        annotations[protein_id] = {'go_term': None, 'ec_number': None}

# Save trypanosome results
tryp_df = save_results(tryp_protein_ids, annotations, "trypanosome")

# Process reference proteins for each tier
all_dfs = [tryp_df]
for tier, dir_path in ref_dirs.items():
    print(f"\nProcessing {tier}...")
    ref_pt_files = find_pt_files(dir_path)
    ref_embeddings, ref_protein_ids = load_embeddings(ref_pt_files)
    
    # Get annotations for reference proteins
    annotations = {}
    for protein_id in ref_protein_ids:
        uniprot_id = extract_uniprot_id(protein_id)
        if uniprot_id:
            go_term, ec_number = get_protein_annotations(uniprot_id)
            annotations[protein_id] = {'go_term': go_term, 'ec_number': ec_number}
        else:
            annotations[protein_id] = {'go_term': None, 'ec_number': None}
    
    # Save reference results
    ref_df = save_results(ref_protein_ids, annotations, f"reference_{tier}")
    all_dfs.append(ref_df)

# Combine all results and save a complete summary
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv("go_terms/all_annotations.tsv", sep='\t', index=False)

print("\nOverall Statistics:")
print(f"Total proteins processed: {len(combined_df)}")
print(f"Total proteins with GO terms: {combined_df['go_term'].notna().sum()}")
print(f"Total proteins with EC numbers: {combined_df['ec_number'].notna().sum()}")
print("\nBreakdown by dataset:")
summary = combined_df.groupby('dataset').agg({
    'protein_id': 'count',
    'go_term': lambda x: x.notna().sum(),
    'ec_number': lambda x: x.notna().sum()
}).round(2)
summary.columns = ['Total Proteins', 'With GO Terms', 'With EC Numbers']
print(summary)