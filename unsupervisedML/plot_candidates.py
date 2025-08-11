import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

def load_txt_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def plot_tsne(fitted_tsne, labels, protein_ids, candidate_proteins, tier, perplexity, output_dir, esm_model, 
              x_range=None, y_range=None, point_size=None):
    """Create t-SNE plot with candidate protein annotations"""
    # Create figure with space for legend below
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot reference sequences in red, as diamonds
    ref_mask = labels == 'reference'
    ref_count = np.sum(ref_mask)
    ref_size = point_size if point_size is not None else 40
    ax.scatter(
        fitted_tsne[ref_mask, 0],
        fitted_tsne[ref_mask, 1],
        alpha=0.6,
        label='Reference proteins',
        color='#E67066',
        s=ref_size,
        marker='D',
        edgecolor='black',
        linewidth=0.5
    )

    # Get trypanosome indices
    tryp_mask = labels == 'trypanosome'
    tryp_indices = np.where(tryp_mask)[0]
    
    # Separate candidate proteins from other trypanosome proteins
    candidate_indices = []
    other_tryp_indices = []
    
    for i in tryp_indices:
        pid = protein_ids[i]
        if pid in candidate_proteins:
            candidate_indices.append(i)
        else:
            other_tryp_indices.append(i)
    
    # Plot candidate proteins as blue circles
    if candidate_indices:
        candidate_size = point_size if point_size is not None else 100
        ax.scatter(
            fitted_tsne[candidate_indices, 0],
            fitted_tsne[candidate_indices, 1],
            alpha=0.8,
            label='Candidate proteins',
            color='#007896',
            s=candidate_size,
            marker='o',
            edgecolor='black',
            linewidth=1,
            zorder=5
        )
    
    # Plot other trypanosome proteins as grey circles (no EC labeling)
    if other_tryp_indices:
        other_size = point_size if point_size is not None else 40
        ax.scatter(
            fitted_tsne[other_tryp_indices, 0],
            fitted_tsne[other_tryp_indices, 1],
            alpha=0.4,
            label='Other trypanosome proteins',
            color='lightgrey',
            s=other_size,
            marker='o'
        )
    
    # Add title and labels
    ax.set_title(f"t-SNE of ESM Embeddings with Candidate Proteins\n{tier}, perplexity={perplexity}", 
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=14, fontweight='bold')
    ax.set_ylabel("t-SNE 2", fontsize=14, fontweight='bold')
    
    # Set plotting space if specified
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    
    # Adjust tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust legend - place below the plot
    legend = ax.legend(bbox_to_anchor=(0.5, -0.15), 
                      loc='upper center', 
                      borderaxespad=0.,
                      fontsize=11,
                      title="Annotations",
                      title_fontsize=12,
                      ncol=3,  # Arrange legend items in 3 columns
                      frameon=True,
                      fancybox=True,
                      shadow=False)
    

    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend below
    
    # Save plot
    output_path = output_dir / f"{esm_model}_tsne_plot_candidates_{tier}_perp{perplexity}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create t-SNE plots with candidate protein annotations')
    parser.add_argument('--tier', type=str, choices=['tier1', 'tier3'], help='Tier to plot (tier1 or tier3)')
    parser.add_argument('--esm-model', type=str, choices=['esm1b', 'esm2'], help='ESM model to use (esm1b or esm2)')
    parser.add_argument('--x-range', type=float, nargs=2, help='X-axis range (min max)')
    parser.add_argument('--y-range', type=float, nargs=2, help='Y-axis range (min max)')
    parser.add_argument('--point-size', type=float, help='Size of all points in the plot')
    args = parser.parse_args()
    
    # Load candidate proteins
    candidate_file = Path("~/pipeline/unsupervisedML/candidate_list.csv")
    candidate_df = pd.read_csv(candidate_file)
    candidate_proteins = set(candidate_df['Protein_ID'].tolist())
    
    print(f"Loaded {len(candidate_proteins)} candidate proteins")

    output_dir = Path("tsne_plots")
    output_dir.mkdir(exist_ok=True)

    # Determine which models and tiers to process
    esm_models = [args.esm_model] if args.esm_model else ["esm1b", "esm2"]
    tiers = [args.tier] if args.tier else ["tier1", "tier3"]
    
    for esm_model in esm_models:
        data_dir = Path(f"tsne_representations_{esm_model}")
        for tier in tiers:
            for perplexity in [30]:
                tsne_coords_file = data_dir / f"tsne_coords_{tier}_perp{perplexity}.npy"
                protein_ids_file = data_dir / f"protein_ids_{tier}.txt"
                labels_file = data_dir / f"labels_{tier}.npy"
                if not (tsne_coords_file.exists() and protein_ids_file.exists() and labels_file.exists()):
                    print(f"Missing data for {esm_model} {tier} perplexity {perplexity}, skipping.")
                    continue
                tsne_coords = np.load(tsne_coords_file)
                protein_ids = load_txt_list(protein_ids_file)
                labels = np.load(labels_file)
                plot_tsne(tsne_coords, labels, protein_ids, candidate_proteins, tier, perplexity, 
                         output_dir, esm_model, args.x_range, args.y_range, args.point_size)