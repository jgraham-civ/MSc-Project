import os
import torch
import numpy as np
from matplotlib import colormaps
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import dbcv
from sklearn.cluster import HDBSCAN
import random

score_log_path = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_plots/hdbscan_dbcv_scores.tsv")
with open(score_log_path, "w") as f:
    f.write("tier\tmin_samples\tnum_clusters\tdbcv_score\n")

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

    minsamples_params = range(5, 50, 2)

    for minsamples in minsamples_params:
        save_dir = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_plots")
        os.makedirs(save_dir, exist_ok=True)

        # Fit DPC
        print(f"Running HDBSCAN with minsamples={minsamples}...")
        
        clustering = HDBSCAN(min_cluster_size=minsamples).fit(tsne_result)
        labels = clustering.labels_

        # Compute DBCV
        try:
            score = dbcv.dbcv(tsne_result, labels, noise_id=-1)
            print(f"DBCV score for minsamples={minsamples}: {score:.4f}")
        except Exception as e:
            print(f"DBCV computation failed for minsamples={minsamples}: {str(e)}")
            score = None
        
        # Compute number of clusters
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        with open(score_log_path, "a") as f:
            f.write(f"{tier}\t{minsamples}\t{num_clusters}\t{score:.4f}\n")

        # Plot DPC
        print("Plotting DPC...")
        plt.figure(figsize=(12, 10))
        
        #  Get cluster labels (excluding noise)
        cluster_labels = sorted(set(labels) - {-1})
        random.seed(42)  # For reproducibility
        random.shuffle(cluster_labels)

        # Assign a unique color per label
        colors = plt.get_cmap('turbo', num_clusters)
        label_color_map = {label: colors(i) for i, label in enumerate(cluster_labels)}

        for k in np.unique(labels):
            class_members = labels == k
            if k == -1:
                plt.scatter(tsne_result[class_members, 0], tsne_result[class_members, 1],
                            c='lightgrey', alpha=0.5, label='Noise')
            else:
                plt.scatter(tsne_result[class_members, 0], tsne_result[class_members, 1],
                            c=[label_color_map[k]], alpha=0.8, label=f'Cluster {k}')

        plt.title(f"HDBSCAN Clustering: {tier} (min_cluster_size={minsamples})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"hdbscan_minclustersize{minsamples}.png"))
        plt.close()