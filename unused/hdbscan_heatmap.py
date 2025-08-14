import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

score_log_path = os.path.expandvars("${HOME}/pipeline/esm/hdbscan_cubesearch/hdbscan_dbcv_scores.tsv")
df = pd.read_csv(score_log_path, sep='\t')

tiers = df['tier'].unique()
save_dir = os.path.dirname(score_log_path)
os.makedirs(save_dir, exist_ok=True)

for tier in tiers:
    tier_df = df[df['tier'] == tier]
    min_samples_list = sorted(tier_df['min_samples'].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(min_samples_list) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

    for idx, min_samples in enumerate(min_samples_list):
        ax = axes[idx // n_cols, idx % n_cols]
        sub_df = tier_df[tier_df['min_samples'] == min_samples]
        heatmap_data = sub_df.pivot(index='perplexity', columns='learning_rate', values='dbcv_score').fillna(-1)
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='PiYG', cbar_kws={'label': 'DBCV Score'}, center=0, ax=ax, annot_kws={"size": 6})
        ax.set_title(f'min_samples={min_samples}')
        ax.set_xlabel('learning_rate')
        ax.set_ylabel('perplexity')
    
    plt.suptitle(f'DBCV Score Heatmaps for {tier}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f'hdbscan_dbcv_heatmaps_{tier}.png'))
    plt.close()