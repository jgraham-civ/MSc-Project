#!/usr/bin/env python3
import os
import glob
import csv
import mdtraj as md
import numpy as np
from typing import Tuple, List

SRTKET = ['SER', 'ARG', 'THR', 'LYS', 'GLU', 'THR']

def chain_label(chain):
    """Return a readable identifier if available; otherwise the index as string."""
    for attr in ("id", "chain_id", "name"):
        val = getattr(chain, attr, None)
        if val not in (None, ""):
            return str(val)
    return str(chain.index)

def chain_starts_with_sequence(chain: md.core.topology.Chain, sequence: List[str]) -> bool:
    """True if the first len(sequence) protein residues of chain match exactly."""
    protein_res = [r for r in chain.residues if r.is_protein]
    if len(protein_res) < len(sequence):
        return False
    names = [r.name for r in protein_res[:len(sequence)]]
    return names == sequence

def find_h3_chain(top: md.Topology) -> md.core.topology.Chain:
    """Find the chain whose N-terminus starts with SRTKET."""
    for ch in top.chains:
        if chain_starts_with_sequence(ch, SRTKET):
            return ch
    raise ValueError("No chain starts with SRTKET (SER-ARG-THR-LYS-GLU-THR).")

def nearest_residue_to_atom(traj: md.Trajectory, atom_index: int, chain: md.core.topology.Chain
                            ) -> Tuple[md.core.topology.Residue, int, float]:
    """
    For frame 0, find the residue in `chain` whose closest atom is nearest to `atom_index`.
    Returns (residue, closest_atom_index, min_distance_nm).
    """
    xyz = traj.xyz[0]
    target = xyz[atom_index]

    best_res = None
    best_atom_idx = None
    best_dist = np.inf

    for res in chain.residues:
        res_atom_indices = [a.index for a in res.atoms]
        if not res_atom_indices:
            continue
        res_xyz = xyz[res_atom_indices]
        dists = np.linalg.norm(res_xyz - target, axis=1)
        i_min = int(np.argmin(dists))
        if dists[i_min] < best_dist:
            best_dist = float(dists[i_min])
            best_atom_idx = res_atom_indices[i_min]
            best_res = res

    if best_res is None:
        raise RuntimeError("Could not find any atoms in the specified chain.")
    return best_res, best_atom_idx, best_dist

def process_pdb(pdb_path: str, atom_selection: str = "name C41"
                ) -> Tuple[str, str, float]:
    """
    Process a single PDB and return Boltz_ID(, NearestResidue, DistanceAngstrom).
    Boltz_ID = everything before first '_' in the base filename.
    NearestResidue = name + resSeq, e.g., LYS9
    """
    basename = os.path.basename(pdb_path)
    boltz_id = basename.split('_', 1)[0]

    traj = md.load(pdb_path)
    h3_chain = find_h3_chain(traj.topology)

    sel = traj.topology.select(atom_selection)
    if len(sel) == 0:
        raise ValueError(f"No atoms match selection '{atom_selection}' in {basename}")
    if len(sel) > 1:
        print(f"Warning: {len(sel)} atoms match selection in {basename}; using the first.")
    target_idx = int(sel[0])

    nearest_res, nearest_atom_idx, min_dist_nm = nearest_residue_to_atom(traj, target_idx, h3_chain)

    distance_angstrom = float(min_dist_nm * 10.0)
    nearest_residue_str = f"{nearest_res.name}{nearest_res.resSeq}"

    return boltz_id, nearest_residue_str, distance_angstrom

def main(folder: str = ".", atom_selection: str = "name C41", output_csv: str = "nearest_H3_distances.csv"):
    pdb_files = sorted(glob.glob(os.path.join(folder, "*.pdb")))
    if not pdb_files:
        print(f"No .pdb files found in {folder}")
        return

    rows = []
    print(f"Found {len(pdb_files)} PDB files in {folder}")
    for pdb_path in pdb_files:
        try:
            boltz_id, nearest_residue, dist_ang = process_pdb(pdb_path, atom_selection)
            rows.append((boltz_id, nearest_residue, dist_ang))
            print(f"{os.path.basename(pdb_path)} -> {boltz_id}, {nearest_residue}, {dist_ang:.2f} Å")
        except Exception as e:
            basename = os.path.basename(pdb_path)
            boltz_id = basename.split('_', 1)[0]
            print(f"ERROR processing {basename}: {e}")
            rows.append((boltz_id, "NA", ""))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Boltz_ID", "NearestResidue", "DistanceAngstrom"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    atom_sel = "name C41"  # You can make this more specific
    out_csv = "nearest_H3_distances.csv"
    folder = "."
    main(folder=folder, atom_selection=atom_sel, output_csv=out_csv)
