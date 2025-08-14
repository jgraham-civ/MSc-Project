import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import dbcv
from sklearn.cluster import HDBSCAN
import random

score_log_path = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_cubesearch/hdbscan_dbcv_scores.tsv")
with open(score_log_path, "w") as f:
    f.write("tier\tperplexity\tlearning_rate\tmin_samples\tnum_clusters\tdbcv_score\n")

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

def run_tsne(X, perplexity, learning_rate):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=0)
    output = tsne.fit_transform(X)
    return output

## Execute pipeline

ref_dirs = {"tier1":"${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings", "tier2":"${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm1b_embeddings", "tier3":"${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings", "tier4":"${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm1b_embeddings"}
tryp_dir = "${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings"
perplexities = range(5, 50, 5)
learning_rates = range(10, 1000, 100)
minsamples_params = range(4, 20, 4)

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

    for perplexity in perplexities:
        for learning_rate in learning_rates:
            print(f"Running t-SNE with perplexity={perplexity}, learning_rate={learning_rate}...")
            tsne_result = run_tsne(np.concatenate([ref_embeddings, tryp_embeddings], axis=0), perplexity, learning_rate)
            for minsamples in minsamples_params:
                save_dir = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_cubesearch")
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
                    if score is not None:
                        f.write(f"{tier}\t{perplexity}\t{learning_rate}\t{minsamples}\t{num_clusters}\t{score:.4f}\n")
                    else:
                        f.write(f"{tier}\t{perplexity}\t{learning_rate}\t{minsamples}\t{num_clusters}\tNA\n")

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

                plt.title(f"HDBSCAN Clustering: {tier} (perplexity={perplexity}, learning_rate={learning_rate}, min_cluster_size={minsamples})")
                plt.xlabel("t-SNE 1")
                plt.ylabel("t-SNE 2")
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, f"hdbscan_{tier}_perplexity{perplexity}_learningrate{learning_rate}_minclustersize{minsamples}.png"))
                plt.close()