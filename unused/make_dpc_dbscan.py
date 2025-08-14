import os
import torch
import numpy as np
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import dbcv
from sklearn.cluster import DBSCAN

score_log_path = os.path.expandvars("${HOME}/pipeline/esm/dbscan_plots/dbscan_dbcv_scores.tsv")
with open(score_log_path, "w") as f:
    f.write("tier\teps\tmin_samples\tdbcv_score\n")

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

    eps_params = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40]
    minsamples_params = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50]

    for eps in eps_params:
        for minsamples in minsamples_params:
            save_dir = os.path.expandvars("${HOME}/pipeline/esm/dbscan_plots")
            os.makedirs(save_dir, exist_ok=True)

            # Fit DPC
            print(f"Running DBSCAN with eps={eps}, minsamples={minsamples}...")
            
            clustering = DBSCAN(eps=eps, min_samples=minsamples).fit(tsne_result)
            labels = clustering.labels_

            # Compute DBCV
            try:
                score = dbcv.dbcv(tsne_result, labels, noise_id=-1)
                print(f"DBCV score for eps={eps}, minsamples={minsamples}: {score:.4f}")
            except Exception as e:
                print(f"DBCV computation failed for eps={eps}, minsamples={minsamples}: {str(e)}")
                score = None

            with open(score_log_path, "a") as f:
                f.write(f"{tier}\t{eps}\t{minsamples}\t{score:.4f}\n")
            
            # Plot DPC
            print("Plotting DPC...")
            plt.figure(figsize=(12, 10))
            colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(labels))))
            label_to_color = dict(zip(np.unique(labels), colors))

            for k in np.unique(labels):
                class_members = labels == k
                if k == -1:
                    plt.scatter(tsne_result[class_members, 0], tsne_result[class_members, 1],
                                c='lightgrey', alpha=0.5, label='Noise')
                else:
                    plt.scatter(tsne_result[class_members, 0], tsne_result[class_members, 1],
                                c=[label_to_color[k]], label=f'Cluster {k}', alpha=0.8)

            plt.title(f"DPC Clustering: {tier} (eps={eps}, minsamples={minsamples})")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"dpc_eps{eps}_minsamples{minsamples}.png"))
            plt.close()