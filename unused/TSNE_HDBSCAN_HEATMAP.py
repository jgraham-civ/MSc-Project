import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

metrics = [
    ('dbcv_score', 'DBCV Score', 'PiYG', 0, '.2f', -1, 1),
    ('noise_ratio', 'Noise Ratio', 'Blues', None, '.2f', 0, 0.6),
    ('num_clusters', 'Number of Clusters', 'Oranges', None, 'd', None, None)
]

score_log_path = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_cubesearch/hdbscan_dbcv_scores.tsv")
df = pd.read_csv(score_log_path, sep='\t')

save_dir = os.path.dirname(score_log_path)
os.makedirs(save_dir, exist_ok=True)

min_samples_list = sorted(df['min_samples'].unique())
n_cols = 2
n_rows = (len(min_samples_list) + n_cols - 1) // n_cols

for metric, label, cmap, center, fmt, vmin, vmax in metrics:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for idx, min_samples in enumerate(min_samples_list):
        ax = axes[idx // n_cols, idx % n_cols]
        sub_df = df[df['min_samples'] == min_samples]
        heatmap_data = sub_df.pivot(index='learning_rate', columns='perplexity', values=metric).fillna(-1)
        sns.heatmap(
            heatmap_data, annot=True, fmt=fmt, cmap=cmap,
            cbar_kws={'label': label},
            center=center, ax=ax, annot_kws={"size": 6},
            vmin=vmin, vmax=vmax
        )
        ax.set_title(f'min_samples={min_samples}')
        ax.set_xlabel('perplexity')
        ax.set_ylabel('learning_rate')
    
    for idx in range(len(min_samples_list), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])
    plt.suptitle(f'{label} Heatmaps by Minimum Cluster Size', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f'hdbscan_{metric}_heatmaps.png'))
    plt.close()