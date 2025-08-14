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
        mean_embedding = data["mean_representations"][33]
        embeddings.append(mean_embedding.numpy())
    return np.vstack(embeddings), all_pt_files  # return filenames too

def run_tsne(X):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    output = tsne.fit_transform(X)
    return output

def plot_tsne(fitted_tsne, labels, tier, filenames, target_filename):
    plt.figure(figsize=(12, 10))

    ref_mask = labels == 'reference'
    tryp_mask = labels == 'trypanosome'

    plt.scatter(
        fitted_tsne[ref_mask, 0], fitted_tsne[ref_mask, 1],
        alpha=0.8, label='reference', color='coral'
    )
    plt.scatter(
        fitted_tsne[tryp_mask, 0], fitted_tsne[tryp_mask, 1],
        alpha=0.8, label='trypanosome', color='lightgrey'
    )

    # Highlight the target file
    idx = filenames.index(target_filename)
    x, y = fitted_tsne[idx]
    plt.scatter(x, y, color='yellowgreen', s=90, label='Target')
    plt.text(x + 3, y + 3, "Q584S3|Q584S3_TRYB2 Alpha N-terminal protein methyltransferase 1", fontsize=8, color='black')

    plt.title(f"t-SNE of ESM Mean Embeddings ({tier})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tsne_plot_{tier}_labelled.png")
    plt.close()

# === Main pipeline ===

target_filename = os.path.expandvars("${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings/tr|Q584S3|Q584S3_TRYB2 Alpha N-terminal protein methyltransferase 1 OS=Trypanosoma brucei brucei (strain 927/4 GUTat10.1) OX=185431 GN=Tb06.4M18.660 PE=3 SV=1.pt")

ref_dirs = {
    "tier1":"${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings",
    "tier2":"${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm1b_embeddings",
    "tier3":"${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings",
    "tier4":"${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm1b_embeddings"
}
tryp_dir = "${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings"

print("Finding tryp .pt files...")
tryp_pt_files = find_pt_files(tryp_dir)
print("Loading tryp embeddings...")
tryp_embeddings, tryp_filenames = load_embeddings(tryp_pt_files) 

for tier, ref_path in ref_dirs.items():
    print(f"\n### Processing {tier}... ###")
    print("Finding reference .pt files...")
    ref_pt_files = find_pt_files(ref_path)
    print("Loading reference embeddings...")
    ref_embeddings, ref_filenames = load_embeddings(ref_pt_files)

    print("Running t-SNE...")
    all_embeddings = np.concatenate([ref_embeddings, tryp_embeddings], axis=0)
    all_filenames = ref_filenames + tryp_filenames
    tsne_result = run_tsne(all_embeddings)

    print("Plotting...")
    group_labels = np.array(['reference'] * len(ref_embeddings) + ['trypanosome'] * len(tryp_embeddings))
    plot_tsne(tsne_result, group_labels, tier, all_filenames, target_filename)
