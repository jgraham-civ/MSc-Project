#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "all_results.csv")
    out_path = os.path.join(here, "vina_vs_distance_scatter.png")

    df = pd.read_csv(csv_path)
    # Ensure numeric
    x = pd.to_numeric(df.get("Redocking_Vina_Score_SAM"), errors="coerce")
    y = pd.to_numeric(df.get("Distance_Angstrom"), errors="coerce")
    plot_df = pd.DataFrame({
        "Redocking_Vina_Score_SAM": x,
        "Distance_Angstrom": y,
    }).dropna()
    # Carry over nearest residue labels for targeting SER1
    if "Nearest_Residue" in df.columns:
        plot_df["Nearest_Residue"] = df.loc[plot_df.index, "Nearest_Residue"].astype(str)

    # Categorize by quadrants relative to x=-7 (vertical) and y=6 (horizontal)
    x_cut = -6
    y_cut = 5
    def categorize(row):
        if row["Redocking_Vina_Score_SAM"] > x_cut:
            return "Non-SAM-binder"
        if row["Distance_Angstrom"] > y_cut:
            return "SAM-binder, Non-H3-binder"
        return "SAM-binder, H3 methyltransferase"
    plot_df["Category"] = plot_df.apply(categorize, axis=1)

    # Highlight the protein that orients SAM closest to Ser1 (SER1)
    if "Nearest_Residue" in plot_df.columns:
        ser_mask = plot_df["Nearest_Residue"].str.replace(" ", "", regex=False).str.upper().str.startswith("SER1")
        if ser_mask.any():
            closest_idx = plot_df.loc[ser_mask, "Distance_Angstrom"].idxmin()
            plot_df.loc[closest_idx, "Category"] = "H3 N-terminal methyltransferase"

    plt.figure(figsize=(7, 6))
    palette = {
        "Non-SAM-binder": "lightgray",
        "SAM-binder, Non-H3-binder": "steelblue",
        "SAM-binder, H3 methyltransferase": "yellowgreen",
        "H3 N-terminal methyltransferase": "gold",
    }
    sns.scatterplot(
        data=plot_df,
        x="Redocking_Vina_Score_SAM",
        y="Distance_Angstrom",
        hue="Category",
        palette=palette,
        s=50,
        edgecolor="none",
        alpha=1,
    )
    # Requested guides and limits
    plt.axhline(y=y_cut, linestyle='--', color='gray', alpha=0.8)
    plt.axvline(x=x_cut, linestyle='--', color='gray', alpha=0.8)
    plt.axhline(y=1, linestyle='--', color='gray', alpha=0.8)
    plt.ylim(bottom=0)
    plt.legend(title="", loc="upper left")
    plt.xlabel("Re-docking Vina Affinity (kcal/mol) [C + S]")
    plt.ylabel("Distance to H3 N (Å)")
    plt.title("Distance to H3 N vs. Re-docking Vina Affinity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    main()

