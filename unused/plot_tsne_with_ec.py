import os
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
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
    
    if dir_name.endswith('_esm2_embeddings_flattened'):
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

def load_embeddings(all_pt_files):
    """Load embeddings and return with protein IDs"""
    embeddings = []
    protein_ids = []
    
    for file in all_pt_files:
        data = torch.load(file, map_location="cpu")
        protein_id = get_protein_id_from_file(file)
        protein_ids.append(protein_id)
        mean_embedding = data["mean_representations"][33]
        embeddings.append(mean_embedding.numpy())
    
    output = np.vstack(embeddings)
    return output, protein_ids

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
    first_digit = ec_prefix.split('.')[0]
    return f"EC {ec_prefix} ({ec_classes.get(first_digit, 'Unknown')})"

def plot_tsne(fitted_tsne, labels, protein_ids, ec_terms_df, tier, perplexity, output_dir):
    """Create t-SNE plot with EC term annotations"""
    plt.figure(figsize=(20, 15))
    
    # Plot reference sequences in blue, as diamonds
    ref_mask = labels == 'reference'
    plt.scatter(
        fitted_tsne[ref_mask, 0],
        fitted_tsne[ref_mask, 1],
        alpha=0.6,
        label='Reference proteins',
        color='royalblue',
        s=50,
        marker='D'  # Diamond marker
    )

    # Get tryp indices
    tryp_mask = labels == 'trypanosome'
    tryp_indices = np.where(tryp_mask)[0]
    
    # Get EC terms for trypanosome proteins only
    tryp_protein_ids = [protein_ids[i] for i in tryp_indices]
    tryp_ec_terms = ec_terms_df[ec_terms_df['protein_id'].isin(tryp_protein_ids)]
    
    # Process EC terms to get first three digits
    def get_ec_prefix(ec_term):
        if pd.isna(ec_term):
            return None
        parts = str(ec_term).split('.')
        if len(parts) >= 3:
            return '.'.join(parts[:3])
        return None
    
    # Add EC prefix column
    tryp_ec_terms['ec_prefix'] = tryp_ec_terms['ec_number'].apply(get_ec_prefix)
    
    # Count EC prefix frequencies
    ec_prefix_counts = Counter(tryp_ec_terms[tryp_ec_terms['ec_prefix'].notna()]['ec_prefix'])
    
    # Get top 20 most common EC prefixes
    plot_terms = [term for term, count in ec_prefix_counts.most_common(20)]
    
    print(f"\nEC prefixes to plot ({len(plot_terms)}):")
    for term in plot_terms:
        print(f"EC {term}: {ec_prefix_counts[term]} proteins")
    
    # Color map for these terms
    n_terms = len(plot_terms)
    random.seed(42)  # For reproducibility
    random.shuffle(plot_terms)
    colors = plt.get_cmap('turbo', n_terms)
    label_color_map = {label: colors(i) for i, label in enumerate(plot_terms)}
    
    # Map each trypanosome protein to its EC prefix
    protein_to_group = {}
    for i in tryp_indices:
        pid = protein_ids[i]
        row = tryp_ec_terms[tryp_ec_terms['protein_id'] == pid]
        if not row.empty and pd.notna(row['ec_prefix'].values[0]):
            ec_prefix = row['ec_prefix'].values[0]
            if ec_prefix in plot_terms:
                protein_to_group[pid] = ec_prefix
            else:
                protein_to_group[pid] = 'Other/No EC number'
        else:
            protein_to_group[pid] = 'Other/No EC number'
    
    # Plot each EC group
    for term in plot_terms:
        group_mask = [i for i in tryp_indices if protein_to_group[protein_ids[i]] == term]
        if group_mask:
            plt.scatter(
                fitted_tsne[group_mask, 0],
                fitted_tsne[group_mask, 1],
                alpha=0.8,
                label=f"{get_ec_class_name(term)} ({len(group_mask)} proteins)",
                color=label_color_map[term],
                s=100,
                edgecolor='black',
                linewidth=1
            )
    
    # Plot all other/none
    other_mask = [i for i in tryp_indices if protein_to_group[protein_ids[i]] == 'Other/No EC number']
    if other_mask:
        plt.scatter(
            fitted_tsne[other_mask, 0],
            fitted_tsne[other_mask, 1],
            alpha=0.3,
            label=f'Other/No EC number ({len(other_mask)} proteins)',
            color='lightgrey',
            s=50
        )
    
    # Add title and labels
    plt.title(f"t-SNE of ESM Embeddings with EC Numbers\n{tier}, perplexity={perplexity}", 
              fontsize=14, pad=20)
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.grid(True)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0.,
              fontsize=10,
              title="EC Numbers\n(top 20 by 3-digit prefix)")
    
    # Save plot
    output_path = output_dir / f"esm2_tsne_plot_EC_{tier}_perp{perplexity}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    # Load EC terms
    ec_terms_file = Path("annotations/all_annotations.tsv")
    if not ec_terms_file.exists():
        raise FileNotFoundError(
            f"EC terms file not found at {ec_terms_file}. "
            "Please ensure you have the EC terms file in the correct location."
        )
    
    ec_terms_df = pd.read_csv(ec_terms_file, sep='\t')
    print(f"Loaded EC terms for {len(ec_terms_df)} proteins")
    print(f"Found {ec_terms_df['ec_number'].notna().sum()} proteins with EC numbers")
    
    # Define directories
    ref_dirs = {
        "tier1": "${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm2_embeddings_flattened",
        "tier2": "${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm2_embeddings_flattened",
        "tier3": "${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm2_embeddings_flattened",
        "tier4": "${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm2_embeddings_flattened"
    }
    tryp_dir = "${HOME}/pipeline/esm/tryp_combined_cleaned_esm2_embeddings_flattened"
    
    # Create output directory
    output_dir = Path("tsne_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load trypanosome embeddings
    print("\nProcessing trypanosome proteins...")
    tryp_pt_files = find_pt_files(os.path.expandvars(tryp_dir))
    tryp_embeddings, tryp_protein_ids = load_embeddings(tryp_pt_files)
    
    # Process each reference tier
    for tier, dir_path in ref_dirs.items():
        print(f"\nProcessing {tier}...")
        ref_pt_files = find_pt_files(os.path.expandvars(dir_path))
        ref_embeddings, ref_protein_ids = load_embeddings(ref_pt_files)
        
        # Combine embeddings and create labels
        combined_embeddings = np.concatenate([ref_embeddings, tryp_embeddings], axis=0)
        all_protein_ids = ref_protein_ids + tryp_protein_ids
        group_labels = np.array(['reference'] * len(ref_embeddings) + ['trypanosome'] * len(tryp_embeddings))
        
        # Run t-SNE with different perplexity values
        for perplexity in [30, 50, 90]:
            print(f"Running t-SNE for {tier} with perplexity {perplexity}...")
            tsne_result = run_tsne(combined_embeddings, perplexity)
            
            # Create and save plot
            plot_tsne(tsne_result, group_labels, all_protein_ids, ec_terms_df, 
                     tier, perplexity, output_dir)

if __name__ == "__main__":
    main() 