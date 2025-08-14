import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from itertools import product
import multiprocessing

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

def fit_tsne(X, perplexity, learning_rate):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=0)
    output = tsne.fit_transform(X)
    return output

def fit_and_save_tsne(args):
    X, perplexity, learning_rate = args
    print(f"Fitting TSNE with perplexity = {perplexity} and learning rate = {learning_rate}...")
    tsne_result = fit_tsne(X, perplexity=perplexity, learning_rate=learning_rate)
    out_path = f"hdbscan_cubesearch/tryp_tsne_perplexity{perplexity}_learningrate{learning_rate}.npy"
    print(f"Saving TSNE to {out_path}...")
    np.save(out_path, tsne_result)

tryp_dir = "${HOME}/pipeline/esm/tryp_combined_cleaned_esm2_embeddings_flattened"

print("Finding tryp .pt files...")
tryp_pt_files = find_pt_files(tryp_dir)

print("Loading tryp embeddings...")
tryp_embeddings = load_embeddings(tryp_pt_files)

perplexities = range(30, 200, 20)
learning_rates = range(10, 1000, 200)

param_grid = list(product(perplexities, learning_rates))
args_list = [(tryp_embeddings, perplexity, learning_rate) for perplexity, learning_rate in param_grid]

if __name__ == "__main__":
    with multiprocessing.Pool(processes=2) as pool:
        pool.map(fit_and_save_tsne, args_list)