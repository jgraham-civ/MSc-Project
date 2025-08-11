import numpy as np
import pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from collections import Counter
import os
import re
import string

# Helper functions

def load_txt_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def get_available_options():
    # Scan for available models, tiers, and perplexities
    base = Path('.')
    models = [d.name.replace('tsne_representations_', '') for d in base.glob('tsne_representations_*') if d.is_dir()]
    options = {}
    pattern = re.compile(r'tsne_coords_(.+)_perp(\d+)\.npy')
    for model in models:
        model_dir = base / f'tsne_representations_{model}'
        tiers = set()
        perplexities = set()
        for f in model_dir.glob('tsne_coords_*.npy'):
            m = pattern.match(f.name)
            if m:
                tier = m.group(1)
                perp = int(m.group(2))
                tiers.add(tier)
                perplexities.add(perp)
        options[model] = {'tiers': sorted(tiers), 'perplexities': sorted(perplexities)}
    return options

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

def load_data(esm_model, tier, perplexity):
    data_dir = Path(f"tsne_representations_{esm_model}")
    tsne_coords = np.load(data_dir / f"tsne_coords_{tier}_perp{perplexity}.npy")
    protein_ids = load_txt_list(data_dir / f"protein_ids_{tier}.txt")
    descriptions = load_txt_list(data_dir / f"descriptions_{tier}.txt")
    labels = np.load(data_dir / f"labels_{tier}.npy")
    return tsne_coords, protein_ids, descriptions, labels

def make_plot_df(tsne_coords, protein_ids, descriptions, labels):
    return pd.DataFrame({
        'x': tsne_coords[:, 0],
        'y': tsne_coords[:, 1],
        'protein_id': protein_ids,
        'description': descriptions,
        'label': labels
    })

def make_figure(df):
    color_map = {'reference': '#E67066', 'trypanosome': '#5D8F24'}
    fig = go.Figure()
    for group in df['label'].unique():
        group_df = df[df['label'] == group]
        fig.add_trace(go.Scattergl(
            x=group_df['x'],
            y=group_df['y'],
            mode='markers',
            marker=dict(
                color=color_map.get(group, 'gray'),
                size=8,
                line=dict(width=0.5, color='black')
            ),
            name=group,
            customdata=group_df[['protein_id', 'description']].values,
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>'
        ))
    fig.update_layout(
        dragmode='lasso',
        title="t-SNE Plot (Lasso-select points to see word frequencies below)",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend_title="Group",
        font=dict(size=12),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black', mirror=True),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black', mirror=True)
    )
    return fig

def get_word_frequencies(descriptions, n=20):
    words = []
    for desc in descriptions:
        # Remove punctuation and split
        desc_clean = desc.translate(str.maketrans('', '', string.punctuation))
        words.extend([
            w.lower() for w in desc_clean.split()
            if len(w) > 2
        ])
    counter = Counter(words)
    most_common = counter.most_common(n)
    return most_common

# Dash app
options = get_available_options()
default_model = next(iter(options))
default_tier = options[default_model]['tiers'][0]
default_perp = options[default_model]['perplexities'][0]

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive t-SNE Explorer with Cluster Labeling"),
    html.Div([
        html.Label("ESM Model:"),
        dcc.Dropdown(
            id='esm-model',
            options=[{'label': m, 'value': m} for m in options],
            value=default_model,
            clearable=False
        ),
        html.Label("Tier:"),
        dcc.Dropdown(
            id='tier',
            options=[{'label': t, 'value': t} for t in options[default_model]['tiers']],
            value=default_tier,
            clearable=False
        ),
        html.Label("Perplexity:"),
        dcc.Dropdown(
            id='perplexity',
            options=[{'label': p, 'value': p} for p in options[default_model]['perplexities']],
            value=default_perp,
            clearable=False
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    dcc.Graph(id='tsne-plot', style={'height': '80vh', 'width': '90vw'}),
    html.H3("Most Common Words in Selected Descriptions:"),
    html.Div(id='word-frequencies')
])

@app.callback(
    Output('tier', 'options'),
    Output('tier', 'value'),
    Output('perplexity', 'options'),
    Output('perplexity', 'value'),
    Input('esm-model', 'value')
)
def update_tier_perp_options(esm_model):
    tiers = options[esm_model]['tiers']
    perps = options[esm_model]['perplexities']
    return (
        [{'label': t, 'value': t} for t in tiers], tiers[0],
        [{'label': p, 'value': p} for p in perps], perps[0]
    )

@app.callback(
    Output('tsne-plot', 'figure'),
    Input('esm-model', 'value'),
    Input('tier', 'value'),
    Input('perplexity', 'value')
)
def update_figure(esm_model, tier, perplexity):
    tsne_coords, protein_ids, descriptions, labels = load_data(esm_model, tier, perplexity)
    df = make_plot_df(tsne_coords, protein_ids, descriptions, labels)
    return make_figure(df)

@app.callback(
    Output('word-frequencies', 'children'),
    Input('tsne-plot', 'selectedData'),
    State('esm-model', 'value'),
    State('tier', 'value'),
    State('perplexity', 'value')
)
def display_word_frequencies(selectedData, esm_model, tier, perplexity):
    if not selectedData or 'points' not in selectedData:
        return "Select points with the lasso tool to see word frequencies."
    
    tsne_coords, protein_ids, descriptions, labels = load_data(esm_model, tier, perplexity)
    df = make_plot_df(tsne_coords, protein_ids, descriptions, labels)
    
    # Get the selected descriptions by properly mapping trace indices to dataset indices
    selected_descs = []
    selected_groups = []
    
    for point in selectedData['points']:
        curve_number = point.get('curveNumber', 0)  # Which trace this point belongs to
        point_index = point.get('pointIndex', 0)    # Index within that trace
        
        # Get the group name for this trace
        groups = sorted(df['label'].unique())
        if curve_number < len(groups):
            group = groups[curve_number]
            # Get the indices of all points in this group
            group_indices = df[df['label'] == group].index.tolist()
            # Map the point_index to the actual dataset index
            if point_index < len(group_indices):
                actual_index = group_indices[point_index]
                selected_descs.append(df.iloc[actual_index]['description'])
                selected_groups.append(group)
    
    if not selected_descs:
        return "No valid points selected."
    
    # Show selection info
    group_counts = Counter(selected_groups)
    info_text = f"Selected {len(selected_descs)} points: " + ", ".join([f"{group}: {count}" for group, count in group_counts.items()])
    
    most_common = get_word_frequencies(selected_descs, n=20)
    
    return html.Div([
        html.P(info_text, style={'fontWeight': 'bold', 'marginBottom': '10px'}),
        html.Ul([html.Li(f"{word}: {count}") for word, count in most_common])
    ])

if __name__ == '__main__':
    app.run(debug=True) 