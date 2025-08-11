#!/usr/bin/env python3
"""
Simple script to find protein coordinates in t-SNE plots given a protein ID.
"""

import numpy as np
import argparse
from pathlib import Path

def load_txt_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def find_protein_coordinates(protein_id, base_dir="."):
    """Find coordinates of a protein across all t-SNE plots."""
    
    for esm_model in ["esm1b", "esm2"]:
        data_dir = Path(base_dir) / f"tsne_representations_{esm_model}"
        
        for tier in ["tier1", "tier3"]:
            for perplexity in [30, 90]:
                # Check if files exist
                tsne_coords_file = data_dir / f"tsne_coords_{tier}_perp{perplexity}.npy"
                protein_ids_file = data_dir / f"protein_ids_{tier}.txt"
                
                if not (tsne_coords_file.exists() and protein_ids_file.exists()):
                    continue
                
                # Load data
                tsne_coords = np.load(tsne_coords_file)
                protein_ids = load_txt_list(protein_ids_file)
                
                # Find protein
                try:
                    idx = protein_ids.index(protein_id)
                    coords = tsne_coords[idx]
                    print(f"{esm_model.upper()} - {tier} - perp{perplexity}: ({coords[0]:.4f}, {coords[1]:.4f})")
                except ValueError:
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find protein coordinates in t-SNE plots')
    parser.add_argument('protein_id', help='Protein ID to search for')
    parser.add_argument('--base-dir', default='.', help='Base directory containing t-SNE data')
    
    args = parser.parse_args()
    
    print(f"Searching for protein: {args.protein_id}")
    print("=" * 50)
    
    find_protein_coordinates(args.protein_id, args.base_dir) 