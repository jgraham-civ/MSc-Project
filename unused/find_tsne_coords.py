import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def find_pt_files(root_dir):
    root_dir = os.path.expandvars(root_dir)
    pt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.pt'):
                pt_files.append(os.path.join(dirpath, f))
    print(f"Found {len(pt_files)} .pt files in {root_dir}")
    return pt_files

def load_embeddings(all_pt_files):
    embeddings = []
    for file in all_pt_files:
        data = torch.load(file, map_location="cpu")

        # Extract mean embedding from a specific layer (e.g. layer 33)
        mean_embedding = data["mean_representations"][33]  # shape: (hidden_dim,)
        embeddings.append(mean_embedding.numpy())
    
    # Convert to array: shape (N_proteins, hidden_dim)
    output = np.vstack(embeddings)
    return output

def run_tsne(X):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    output = tsne.fit_transform(X)
    return output

def plot_tsne(fitted_tsne, labels, tier):
    plt.figure(figsize=(12, 10))

    plt.scatter(
        fitted_tsne[labels == 'reference', 0],
        fitted_tsne[labels == 'reference', 1],
        alpha=0.8,
        label='reference',
        color='coral'
    )

    plt.scatter(
        fitted_tsne[labels == 'trypanosome', 0],
        fitted_tsne[labels == 'trypanosome', 1],
        alpha=0.8,
        label='trypanosome',
        color='lightgrey'
    )

    plt.title(f"t-SNE of ESM Mean Embeddings ({tier})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_plot_{tier}.png")
    plt.close()

## Execute pipeline

ref_dirs = {"tier1":"${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings"}
tryp_dir = "${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings"

print("Finding tryp .pt files...")
tryp_pt_files = find_pt_files(tryp_dir)
print("Loading tryp embeddings...")
tryp_embeddings = load_embeddings(tryp_pt_files) 

for tier, i in ref_dirs.items():
    print(f"\n### Processing {tier}... ###")
    print(f"Finding reference .pt files...")
    ref_pt_files = find_pt_files(i)
    print(f"Loading reference embeddings...")
    ref_embeddings = load_embeddings(ref_pt_files)

    print("Running t-SNE...")
    tsne_result = run_tsne(np.concatenate([ref_embeddings, tryp_embeddings], axis=0))

    print("Plotting...")
    group_labels = np.array(['reference'] * len(ref_embeddings) + ['trypanosome'] * len(tryp_embeddings))

    plot_tsne(tsne_result, group_labels, tier)

    for coord, label in zip(tsne_result, group_labels):
            print(f"{coord[0]:.4f}\t{coord[1]:.4f}\t{label}")

    target_x = -87.5
    target_y = 12.5

    tryp_mask = group_labels == 'trypanosome'
    tryp_tsne = tsne_result[tryp_mask]
    tryp_filenames = np.array(tryp_pt_files)

    target = np.array([target_x, target_y])
    distances = np.linalg.norm(tryp_tsne - target, axis=1)
    closest_idx = np.argmin(distances)
    print(f"Closest label: {group_labels[closest_idx]}")
    print(f"Coordinates: {tryp_tsne[closest_idx]}")
    print(f"Index: {closest_idx}")
    print(f"Filename: {tryp_filenames[closest_idx]}")