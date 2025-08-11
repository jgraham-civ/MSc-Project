import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from pathlib import Path
from collections import Counter

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

def get_protein_id_from_file(file_path):
    """Extract protein ID from file path, handling split filenames"""
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    
    if dir_name.endswith('_embeddings_flattened'):
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
        return f"{uniprot_part[0]}|{uniprot_part[1]}|{uniprot_part[2].split()[0]}"
    return header

def load_embeddings_with_descriptions(all_pt_files):
    embeddings = []
    protein_ids = []
    descriptions = []
    for file in all_pt_files:
        data = torch.load(file, map_location="cpu")
        protein_id, description = get_protein_info_from_file(file)
        protein_ids.append(protein_id)
        descriptions.append(description)
        mean_embedding = data["mean_representations"][33]
        embeddings.append(mean_embedding.numpy())
    output = np.vstack(embeddings)
    return output, protein_ids, descriptions

def run_tsne(X, perplexity):
    """Run t-SNE dimensionality reduction"""
    print(f"Running t-SNE with perplexity {perplexity} on {X.shape[0]} points...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    output = tsne.fit_transform(X)
    return output

def get_ec_class_name(ec_prefix):
    """Return human-readable name for EC class based on first digit"""
    ec_classes = {
        "1": "Oxidoreductases",
        "2": "Transferases",
        "3": "Hydrolases",
        "4": "Lyases",
        "5": "Isomerases",
        "6": "Ligases",
        "7": "Translocases"
    }
    
    # Special case for EC 2.1.1 (Methyltransferases)
    if ec_prefix == "2.1.1":
        return f"EC {ec_prefix} (Methyltransferases)"
    
    first_digit = ec_prefix.split('.')[0]
    return f"EC {ec_prefix} ({ec_classes.get(first_digit, 'Unknown')})"

def get_protein_info_from_file(file_path):
    """Extract protein ID and description from file path, handling split filenames"""
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    
    if dir_name.endswith('_embeddings_flattened'):
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
        
        # The description is what's between the protein ID and OS=
        id_last_part = uniprot_part[2].split()[0]
        
        try:
            # Everything after the ID part
            rest_of_it = uniprot_part[2].split(id_last_part, 1)[1]
            if 'OS=' in rest_of_it:
                description = rest_of_it.split('OS=')[0].strip()
            else:
                description = rest_of_it.strip()
            
            if not description: # If description is empty string
                description = 'No description available'
        except IndexError:
            description = 'No description available'

        return protein_id, description

    return header, 'No description available'


if __name__ == "__main__":
    esm_models = [
        {
            "name": "esm1b",
            "ref_dirs": {
                "tier1": "${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings_flattened",
                "tier3": "${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings_flattened"
            },
            "tryp_dir": "${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings_flattened"
        },
        {
            "name": "esm2",
            "ref_dirs": {
                "tier1": "${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm2_embeddings_flattened",
                "tier3": "${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm2_embeddings_flattened"
            },
            "tryp_dir": "${HOME}/pipeline/esm/tryp_combined_cleaned_esm2_embeddings_flattened"
        }
    ]

    for esm in esm_models:
        print(f"Processing {esm['name']} embeddings...")
        output_dir = Path(f"tsne_representations_{esm['name']}")
        output_dir.mkdir(exist_ok=True)

        # Load trypanosome embeddings
        tryp_pt_files = find_pt_files(os.path.expandvars(esm["tryp_dir"]))
        tryp_embeddings, tryp_protein_ids, tryp_descriptions = load_embeddings_with_descriptions(tryp_pt_files)

        for tier, dir_path in esm["ref_dirs"].items():
            ref_pt_files = find_pt_files(os.path.expandvars(dir_path))
            ref_embeddings, ref_protein_ids, ref_descriptions = load_embeddings_with_descriptions(ref_pt_files)

            # Combine
            combined_embeddings = np.concatenate([ref_embeddings, tryp_embeddings], axis=0)
            all_protein_ids = ref_protein_ids + tryp_protein_ids
            all_descriptions = ref_descriptions + tryp_descriptions
            group_labels = np.array(['reference'] * len(ref_embeddings) + ['trypanosome'] * len(tryp_embeddings))

            for perplexity in [30, 90]:
                tsne_result = run_tsne(combined_embeddings, perplexity)
                np.save(output_dir / f"tsne_coords_{tier}_perp{perplexity}.npy", tsne_result)
                with open(output_dir / f"protein_ids_{tier}.txt", 'w') as f:
                    for pid in all_protein_ids:
                        f.write(f"{pid}\n")
                with open(output_dir / f"descriptions_{tier}.txt", 'w') as f:
                    for desc in all_descriptions:
                        f.write(f"{desc}\n")
                np.save(output_dir / f"labels_{tier}.npy", group_labels)

                trust_scores = []
                neighbors_range = list(range(5, 51, 5))
                for n in neighbors_range:
                    trust = trustworthiness(combined_embeddings, tsne_result, n_neighbors=n)
                    trust_scores.append(trust)
                # Save trust_scores and neighbors_range for plotting later
                np.save(output_dir / f"trustworthiness_{tier}_perp{perplexity}.npy", np.array(trust_scores))