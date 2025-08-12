import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from collections import Counter

### Plot t-SNE embeddings for reference and trypanosome datasets, interactively with Plotly

def load_txt_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def get_ec_class_name(ec_prefix):
    if ec_prefix == '2.1.1':
        return f"EC {ec_prefix} (Methyltransferases)"
    ec_classes = {
        "1": "Oxidoreductases",
        "2": "Transferases",
        "3": "Hydrolases",
        "4": "Lyases",
        "5": "Isomerases",
        "6": "Ligases",
        "7": "Translocases"
    }
    first_digit = ec_prefix.split('.')[0]
    return f"EC {ec_prefix} ({ec_classes.get(first_digit, 'Unknown')})"

def plot_tsne_interactive(fitted_tsne, labels, protein_ids, descriptions, annotation_df, tier, perplexity, output_dir, esm_model):
    plot_df = pd.DataFrame({
        'tsne_1': fitted_tsne[:, 0],
        'tsne_2': fitted_tsne[:, 1],
        'protein_id': protein_ids,
        'description': descriptions,
        'label': labels
    })
    plot_df = pd.merge(plot_df, annotation_df, on='protein_id', how='left')
    tryp_df = plot_df[plot_df['label'] == 'trypanosome'].copy()
    def get_ec_prefix(ec_term):
        if pd.isna(ec_term):
            return None
        parts = str(ec_term).split('.')
        if len(parts) >= 1:
            if parts[0] == '2' and len(parts) >= 3 and parts[1] == '1' and parts[2] == '1':
                return '2.1.1'
            return parts[0]
        return None
    tryp_df.loc[:, 'ec_prefix'] = tryp_df['ec_number'].apply(get_ec_prefix)
    ec_prefix_counts = Counter(tryp_df[tryp_df['ec_prefix'].notna()]['ec_prefix'])
    plot_ec_terms = sorted([term for term, count in ec_prefix_counts.items()])
    def hex_to_rgba(hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    ec_colors = ["#F0BE4D", "#ACD242", "#5D8F24", "#007896", "#1E3F59", "#9F93C3", "#AB5A8D"]
    ec_color_map = {label: hex_to_rgba(ec_colors[i], 0.9) for i, label in enumerate(plot_ec_terms)}
    plot_df['group_name'] = 'No EC Number'
    plot_df['marker_symbol'] = 'circle'
    plot_df['marker_size'] = 8
    plot_df['marker_color'] = 'rgba(211, 211, 211, 0.3)'
    tryp_indices = plot_df[plot_df['label'] == 'trypanosome'].index
    for i in tryp_indices:
        row = plot_df.loc[i]
        pid = row['protein_id']
        tryp_annot_row = tryp_df[tryp_df['protein_id'] == pid]
        if not tryp_annot_row.empty:
            ec_prefix = tryp_annot_row['ec_prefix'].values[0]
            if pd.notna(ec_prefix) and ec_prefix in plot_ec_terms:
                group_name = f"{get_ec_class_name(ec_prefix)} ({ec_prefix_counts[ec_prefix]} proteins)"
                plot_df.loc[i, 'group_name'] = group_name
                plot_df.loc[i, 'marker_color'] = ec_color_map[ec_prefix]
                plot_df.loc[i, 'marker_symbol'] = 'diamond' if ec_prefix == '2.1.1' else 'circle'
                plot_df.loc[i, 'marker_size'] = 12 if ec_prefix == '2.1.1' else 10
                continue
    ref_mask = plot_df['label'] == 'reference'
    plot_df.loc[ref_mask, 'group_name'] = 'Reference proteins'
    plot_df.loc[ref_mask, 'marker_symbol'] = 'diamond'
    plot_df.loc[ref_mask, 'marker_size'] = 8
    plot_df.loc[ref_mask, 'marker_color'] = hex_to_rgba('#E67066', 0.6)
    plot_df['hover_text'] = plot_df.apply(
        lambda row: f"<b>Protein ID:</b> {row['protein_id']}<br>"
                    f"<b>Description:</b> {row['description']}<br>"
                    f"<b>Category:</b> {row['group_name']}",
        axis=1
    )
    fig = go.Figure()
    unique_group_names = plot_df['group_name'].unique()
    def sort_key(group_name):
        if group_name == 'Reference proteins':
            return (0, '')
        if group_name.startswith('EC'):
            try:
                ec_part = group_name.split(' ')[1]
                return (1, [int(x) for x in ec_part.split('.')])
            except (IndexError, ValueError):
                return (4, group_name)
        if group_name == 'No EC Number':
            return (2, '')
        return (4, group_name)
    sorted_group_names = sorted(unique_group_names, key=sort_key)
    for group_name in sorted_group_names:
        group_df = plot_df[plot_df['group_name'] == group_name]
        fig.add_trace(go.Scatter(
            x=group_df['tsne_1'],
            y=group_df['tsne_2'],
            mode='markers',
            marker=dict(
                color=group_df['marker_color'].iloc[0],
                size=group_df['marker_size'].iloc[0],
                symbol=group_df['marker_symbol'].iloc[0],
                line=dict(width=1, color='black') if 'diamond' in group_df['marker_symbol'].unique() else None
            ),
            name=group_name,
            hovertext=group_df['hover_text'],
            hoverinfo='text'
        ))
    fig.update_layout(
        title=f"t-SNE of {esm_model.upper()} Embeddings with EC Numbers<br>{tier}, perplexity={perplexity}",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend_title="Functional Categories",
        font=dict(size=12),
        legend=dict(itemsizing='constant'),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black', mirror=True),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black', mirror=True)
    )
    output_path = output_dir / f"{esm_model}_tsne_plot_EC_{tier}_perp{perplexity}.html"
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")

if __name__ == "__main__":
    annotations_file = Path("annotations/out.emapper.annotations")
    annotations_df = pd.read_csv(
        annotations_file, 
        sep='\t',
        comment='#',
        usecols=[0, 10],
        names=['protein_id', 'ec_number'],
        header=None
    )
    annotations_df['ec_number'] = annotations_df['ec_number'].apply(
        lambda x: x.split(',')[0] if x != '-' else pd.NA
    )
    output_dir = Path("tsne_plots_interactive")
    output_dir.mkdir(exist_ok=True)
    for esm_model in ["esm1b", "esm2"]:
        data_dir = Path(f"tsne_representations_{esm_model}")
        for tier in ["tier1", "tier3"]:
            for perplexity in [30, 90]:
                tsne_coords_file = data_dir / f"tsne_coords_{tier}_perp{perplexity}.npy"
                protein_ids_file = data_dir / f"protein_ids_{tier}.txt"
                descriptions_file = data_dir / f"descriptions_{tier}.txt"
                labels_file = data_dir / f"labels_{tier}.npy"
                if not (tsne_coords_file.exists() and protein_ids_file.exists() and descriptions_file.exists() and labels_file.exists()):
                    print(f"Missing data for {esm_model} {tier} perplexity {perplexity}, skipping.")
                    continue
                tsne_coords = np.load(tsne_coords_file)
                protein_ids = load_txt_list(protein_ids_file)
                descriptions = load_txt_list(descriptions_file)
                labels = np.load(labels_file)
                plot_tsne_interactive(tsne_coords, labels, protein_ids, descriptions, annotations_df, tier, perplexity, output_dir, esm_model)
