import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dbcv
from sklearn.cluster import DBSCAN
import random
import pandas as pd
import seaborn as sns

score_log_path = os.path.expandvars("${HOME}/pipeline/esm/dbscan_plots/dbscan_dbcv_scores.tsv")
with open(score_log_path, "w") as f:
    f.write("tier\teps\tmin_samples\tnum_clusters\tdbcv_score\n")

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

    eps_params = range(2, 50, 4)
    minsamples_params = range(2, 50, 4)

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
            
            # Compute number of clusters
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            with open(score_log_path, "a") as f:
                f.write(f"{tier}\t{eps}\t{minsamples}\t{num_clusters}\t{score:.4f}\n")

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

            plt.title(f"DBSCAN Clustering: {tier} (eps={eps}, minsamples={minsamples})")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"dbscan_eps{eps}_minsamples{minsamples}.png"))
            plt.close()
    
    # Make heatmap of DBCV scores
    df = pd.read_csv(score_log_path, sep='\t')
    heatmap_data = df.pivot(index='eps', columns='min_samples', values='dbcv_score')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='PiYG', cbar_kws={'label': 'DBCV Score'}, center = 0)
    plt.title(f'DBCV Score Heatmap for {tier}')
    plt.xlabel('min_samples')
    plt.ylabel('eps')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'dbscan_dbcv_heatmap_{tier}.png'))
    plt.close()
    