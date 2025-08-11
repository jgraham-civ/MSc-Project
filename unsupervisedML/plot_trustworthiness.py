import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dirs = ["tsne_representations_esm1b", "tsne_representations_esm2"]
tiers = ["tier1", "tier3"]
perplexities = [30, 90]
neighbors_range = list(range(5, 51, 5))

plt.figure(figsize=(8, 6))

# Define consistent styling
model_colors = {"esm1b": "#007896", "esm2": "#E67066"}
tier_markers = {"tier1": "o", "tier3": "D"}  # circle for tier1, diamond for tier3
perplexity_styles = {30: "--", 90: "-"}  # dashed for 30, solid for 90

# Create legend handles for each category
from matplotlib.lines import Line2D

# Create a single legend with all combinations
legend_handles = []

# ESM1b combinations
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm1b"], marker=tier_markers["tier1"], 
                            linestyle=perplexity_styles[30], lw=1.5, markersize=8, 
                            label="ESM1b, Tier 1, Perplexity 30"))
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm1b"], marker=tier_markers["tier1"], 
                            linestyle=perplexity_styles[90], lw=1.5, markersize=8, 
                            label="ESM1b, Tier 1, Perplexity 90"))
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm1b"], marker=tier_markers["tier3"], 
                            linestyle=perplexity_styles[30], lw=1.5, markersize=8, 
                            label="ESM1b, Tier 3, Perplexity 30"))
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm1b"], marker=tier_markers["tier3"], 
                            linestyle=perplexity_styles[90], lw=1.5, markersize=8, 
                            label="ESM1b, Tier 3, Perplexity 90"))

# ESM2 combinations
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm2"], marker=tier_markers["tier1"], 
                            linestyle=perplexity_styles[30], lw=1.5, markersize=8, 
                            label="ESM2, Tier 1, Perplexity 30"))
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm2"], marker=tier_markers["tier1"], 
                            linestyle=perplexity_styles[90], lw=1.5, markersize=8, 
                            label="ESM2, Tier 1, Perplexity 90"))
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm2"], marker=tier_markers["tier3"], 
                            linestyle=perplexity_styles[30], lw=1.5, markersize=8, 
                            label="ESM2, Tier 3, Perplexity 30"))
legend_handles.append(Line2D([0, 8], [0, 0], color=model_colors["esm2"], marker=tier_markers["tier3"], 
                            linestyle=perplexity_styles[90], lw=1.5, markersize=8, 
                            label="ESM2, Tier 3, Perplexity 90"))

for outdir in output_dirs:
    for tier in tiers:
        for perp in perplexities:
            trust_file = Path(outdir) / f"trustworthiness_{tier}_perp{perp}.npy"
            if trust_file.exists():
                trust_scores = np.load(trust_file)
                model = outdir.split("_")[2]
                
                color = model_colors[model]
                marker = tier_markers[tier]
                linestyle = perplexity_styles[perp]
                
                plt.plot(neighbors_range, trust_scores, 
                        marker=marker, 
                        linestyle=linestyle,
                        color=color)

plt.xlabel("k")
plt.ylabel("Trustworthiness")
plt.title("Trustworthiness of t-SNE embeddings")
plt.grid(True)

# Create single legend with increased width
plt.legend(handles=legend_handles, loc='upper right', handlelength=4)

plt.tight_layout()
plt.savefig("trustworthiness.png", bbox_inches='tight', dpi=300)