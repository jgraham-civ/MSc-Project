#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "all_results.csv")
    out_path = os.path.join(here, "vina_vs_distance_scatter_cmap.png")

    df = pd.read_csv(csv_path)
    # Ensure numeric
    x = pd.to_numeric(df.get("Redocking_Vina_Score_H3"), errors="coerce")
    y = pd.to_numeric(df.get("Distance_Angstrom"), errors="coerce")
    c = pd.to_numeric(df.get("protein_iptm"), errors="coerce")
    plot_df = pd.DataFrame({
        "Redocking_Vina_Score_H3": x,
        "Distance_Angstrom": y,
        "protein_iptm": c,
    }).dropna()

    plt.figure(figsize=(8, 6))
    # Continuous colormap with colorbar on the right
    sc = plt.scatter(
        plot_df["Redocking_Vina_Score_H3"],
        plot_df["Distance_Angstrom"],
        c=plot_df["protein_iptm"],
        cmap="magma_r",
        s=50,
        edgecolors="none",
        alpha=1,
    )
    cbar = plt.colorbar(sc, orientation="vertical", pad=0.02)
    cbar.set_label("Protein ipTM score")
    # Requested guides and limits
    plt.axhline(y=5, linestyle='--', color='gray', alpha=0.8)
    plt.axvline(x=-6, linestyle='--', color='gray', alpha=0.8)
    plt.ylim(bottom=0)
    plt.xlabel("Redocking Vina Score (H3) [kcal/mol]")
    plt.ylabel("Distance to H3 N (Å)")
    plt.title("Distance to H3 N vs. Re-docking Vina Score (H3)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    main()

