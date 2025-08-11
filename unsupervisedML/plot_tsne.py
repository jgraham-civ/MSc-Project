import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

def load_txt_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]
    
def get_ec_class_name(ec_prefix):
    """Return human-readable name for EC class based on first digit"""
    if ec_prefix == '2.1.1':
        return f"EC {ec_prefix} (Methyltransferases)"
    
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

def plot_tsne(fitted_tsne, labels, protein_ids, ec_terms_df, tier, perplexity, output_dir, esm_model, highlight_protein=None):
    """Create t-SNE plot with EC term annotations"""
    # Create figure with space for legend below
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot reference sequences in blue, as diamonds
    ref_mask = labels == 'reference'
    ref_count = np.sum(ref_mask)
    ax.scatter(
        fitted_tsne[ref_mask, 0],
        fitted_tsne[ref_mask, 1],
        alpha=0.6,
        label=f'Reference proteins ({ref_count} proteins)',
        color='#E67066',
        s=40,
        marker='D',
        edgecolor='black',
        linewidth=0.5
    )

    # Get tryp indices
    tryp_mask = labels == 'trypanosome'
    tryp_indices = np.where(tryp_mask)[0]
    
    # Get EC terms for trypanosome proteins only
    tryp_protein_ids = [protein_ids[i] for i in tryp_indices]
    tryp_ec_terms = ec_terms_df[ec_terms_df['protein_id'].isin(tryp_protein_ids)].copy()
    
    # Process EC terms to get first digit, with special handling for 2.1.1.*
    def get_ec_prefix(ec_term):
        if pd.isna(ec_term):
            return None
        parts = str(ec_term).split('.')
        if len(parts) >= 1:
            # Special case for EC 2.1.1.*
            if parts[0] == '2' and len(parts) >= 3 and parts[1] == '1' and parts[2] == '1':
                return '2.1.1'
            # All other cases use first digit only
            return parts[0]
        return None
    
    # Add EC prefix column
    tryp_ec_terms.loc[:, 'ec_prefix'] = tryp_ec_terms['ec_number'].apply(get_ec_prefix)
    
    # Count EC prefix frequencies
    ec_prefix_counts = Counter(tryp_ec_terms[tryp_ec_terms['ec_prefix'].notna()]['ec_prefix'])
    
    # Get EC prefixes
    plot_terms = sorted([term for term, count in ec_prefix_counts.items()])
    
    print(f"\nEC prefixes to plot ({len(plot_terms)}):")
    for term in sorted(plot_terms):
        print(f"EC {term}: {ec_prefix_counts[term]} proteins")
    
    # EC terms colours
    ec_colors = ["#F0BE4D", "#ACD242", "#5D8F24", "#007896", "#1E3F59", "#9F93C3", "#AB5A8D"]
    
    ec_color_map = {label: ec_colors[i] for i, label in enumerate(plot_terms)}
    
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
            ax.scatter(
                fitted_tsne[group_mask, 0],
                fitted_tsne[group_mask, 1],
                alpha=0.6,
                label=f"{get_ec_class_name(term)} ({len(group_mask)} proteins)",
                color=ec_color_map[term],
                s=80,
                marker='D' if term == '2.1.1' else 'o',
                edgecolor='black' if term == '2.1.1' else None,
                linewidth=0.5
            )
    
    # Plot all other/none
    other_mask = [i for i in tryp_indices if protein_to_group[protein_ids[i]] == 'Other/No EC number']
    if other_mask:
        ax.scatter(
            fitted_tsne[other_mask, 0],
            fitted_tsne[other_mask, 1],
            alpha=0.3,
            label=f'Other/No EC number ({len(other_mask)} proteins)',
            color='lightgrey',
            s=40
        )
    
    # Add title and labels
    ax.set_title(f"t-SNE of ESM1b Embeddings with EC Numbers\n{tier}, perplexity={perplexity}", 
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=14, fontweight='bold')
    ax.set_ylabel("t-SNE 2", fontsize=14, fontweight='bold')
    
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
    
    # Highlight specific protein if requested
    if highlight_protein:
        try:
            highlight_idx = protein_ids.index(highlight_protein)
            highlight_coords = fitted_tsne[highlight_idx]
            ax.scatter(
                highlight_coords[0],
                highlight_coords[1],
                color='black',
                s=200,
                marker='s',
                edgecolor='white',
                linewidth=2,
                zorder=10,
                label=f'Highlighted: {highlight_protein}'
            )
            print(f"Highlighted protein {highlight_protein} at coordinates ({highlight_coords[0]:.4f}, {highlight_coords[1]:.4f})")
        except ValueError:
            print(f"Warning: Protein {highlight_protein} not found in this dataset")
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend below
    
    # Save plot
    if highlight_protein:
        output_path = output_dir / f"found_{esm_model}_tsne_plot_EC_{tier}_perp{perplexity}.png"
    else:
        output_path = output_dir / f"{esm_model}_tsne_plot_EC_{tier}_perp{perplexity}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create t-SNE plots with EC annotations')
    parser.add_argument('--highlight-protein', type=str, help='Protein ID to highlight in plots')
    args = parser.parse_args()
    
    ec_terms_file = Path("annotations/out.emapper.annotations")
    ec_terms_df = pd.read_csv(
        ec_terms_file, 
        sep='\t',
        comment='#',
        usecols=[0, 10],
        names=['protein_id', 'ec_number']
    )
    ec_terms_df['ec_number'] = ec_terms_df['ec_number'].apply(
        lambda x: x.split(',')[0] if x != '-' else pd.NA
    )

    output_dir = Path("tsne_plots")
    output_dir.mkdir(exist_ok=True)

    for esm_model in ["esm1b", "esm2"]:
        data_dir = Path(f"tsne_representations_{esm_model}")
        for tier in ["tier1", "tier3"]:
            for perplexity in [30, 90]:
                tsne_coords_file = data_dir / f"tsne_coords_{tier}_perp{perplexity}.npy"
                protein_ids_file = data_dir / f"protein_ids_{tier}.txt"
                labels_file = data_dir / f"labels_{tier}.npy"
                if not (tsne_coords_file.exists() and protein_ids_file.exists() and labels_file.exists()):
                    print(f"Missing data for {esm_model} {tier} perplexity {perplexity}, skipping.")
                    continue
                tsne_coords = np.load(tsne_coords_file)
                protein_ids = load_txt_list(protein_ids_file)
                labels = np.load(labels_file)
                plot_tsne(tsne_coords, labels, protein_ids, ec_terms_df, tier, perplexity, output_dir, esm_model, args.highlight_protein)