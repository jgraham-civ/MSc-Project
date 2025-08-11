import matplotlib.pyplot as plt

files = {90: "ref_tier5_cdhit_90.clstr"}

for threshold, file in files.items():
    with open(file, 'r') as f:
        clstr_file = f.read()

    cluster_sizes = []
    lines = clstr_file.strip().split('\n')
    current_count = 0

    for line in lines:
        if line.startswith('>Cluster'):
            if current_count > 0:
                cluster_sizes.append(current_count)
            current_count = 0
        else:
            current_count += 1
        # Append the last cluster count
        if current_count > 0:
            cluster_sizes.append(current_count)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(cluster_sizes, bins=range(1, max(cluster_sizes) + 2), align='left', color='lightcoral', edgecolor='black')
    plt.title(f"Distribution of Sequence Counts per CD-HIT Cluster (Threshold: {threshold}%)")
    plt.xlabel("Number of Sequences in Cluster")
    plt.ylabel("Number of Clusters")
    plt.xticks(range(1, max(cluster_sizes) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"cdhit_results_summary_{threshold}.png")
    plt.close()