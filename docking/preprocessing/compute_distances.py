#!/usr/bin/env python3
import os
import glob
import csv
import mdtraj as md
import numpy as np
from typing import Tuple, List, Dict

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

def get_sidechain_nitrogen_indices(chain: md.core.topology.Chain, include_n_term_backbone: bool = True) -> List[int]:
    """
    Return atom indices of nitrogen atoms to consider:
      - All side-chain nitrogens (non-backbone) in the chain
      - Optionally include the N-terminus backbone nitrogen (residue 1) if present
    This matches cases where SER1 backbone N should be considered.
    """
    nitrogen_indices: List[int] = []
    # Identify the first protein residue in this chain (N-terminus)
    first_protein_res = next((r for r in chain.residues if r.is_protein), None)
    for atom in chain.atoms:
        # Robust element detection: fall back to first letter of atom name
        elem = getattr(atom, "element", None)
        symbol = (elem.symbol if elem is not None else (atom.name.strip()[0].upper() if atom.name else None))
        if symbol != 'N':
            continue
        if atom.is_backbone:
            if include_n_term_backbone and first_protein_res is not None and atom.residue == first_protein_res and atom.name.strip().upper() in ('N', 'NT'):
                nitrogen_indices.append(atom.index)
            continue
        nitrogen_indices.append(atom.index)
    return nitrogen_indices

def build_neighbor_map(top: md.Topology) -> Dict[int, List[int]]:
    """Build adjacency list of atom neighbors from topology bonds."""
    neighbors: Dict[int, List[int]] = {}
    for bond in top.bonds:
        a = bond[0].index
        b = bond[1].index
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    return neighbors

def find_sam_methyl_carbon_index(traj: md.Trajectory) -> int:
    """
    Find the methyl carbon that is directly bonded to a sulfur atom in the SAM ligand.
    Strategy:
      - Iterate non-protein residues, preferring resname SAM, then UNL, else any non-protein
      - In each residue, find sulfur atoms (element S)
      - For each sulfur, find carbon neighbors that have no other heavy-atom neighbors besides this sulfur
        (i.e., a methyl carbon attached to S). Hydrogens may be absent; heavy-neighbor criterion is robust.
    Returns the first match found; raises ValueError if none found.
    """
    top = traj.topology
    xyz = traj.xyz[0]

    # Prefer explicit SAM/UNL residues first
    candidate_residues: List[md.core.topology.Residue] = []
    sam_like = [r for r in top.residues if not r.is_protein and r.name == 'SAM']
    unl_like = [r for r in top.residues if not r.is_protein and r.name == 'UNL']
    others   = [r for r in top.residues if not r.is_protein and r.name not in ('SAM', 'UNL')]
    candidate_residues.extend(sam_like or [])
    candidate_residues.extend(unl_like or [])
    candidate_residues.extend(others or [])

    neighbors = build_neighbor_map(top)

    for res in candidate_residues:
        sulfur_atoms = []
        for a in res.atoms:
            elem = getattr(a, 'element', None)
            sym = (elem.symbol if elem is not None else (a.name.strip()[0].upper() if a.name else None))
            if sym == 'S':
                sulfur_atoms.append(a)
        for s_atom in sulfur_atoms:
            s_idx = s_atom.index
            for n_idx in neighbors.get(s_idx, []):
                n_atom = top.atom(n_idx)
                elem = getattr(n_atom, 'element', None)
                sym = (elem.symbol if elem is not None else (n_atom.name.strip()[0].upper() if n_atom.name else None))
                if sym != 'C':
                    continue
                # Count heavy neighbors of this carbon
                heavy_neighbors = []
                for ni in neighbors.get(n_idx, []):
                    a = top.atom(ni)
                    e = getattr(a, 'element', None)
                    s = (e.symbol if e is not None else (a.name.strip()[0].upper() if a.name else None))
                    if s != 'H':
                        heavy_neighbors.append(ni)
                # Methyl carbon attached to sulfur should have exactly one heavy neighbor (the sulfur)
                if len(heavy_neighbors) == 1 and heavy_neighbors[0] == s_idx:
                    return n_idx

    # Fallback: distance-based detection if CONECT/bonds are missing
    # Thresholds (in nm): S-C single bond ~1.82 Å; use a relaxed 0.22 nm. Heavy-atom covalent neighborhood cutoff ~0.22 nm.
    s_c_cutoff_nm = 0.22
    heavy_cutoff_nm = 0.22

    for res in candidate_residues:
        res_atom_indices = [a.index for a in res.atoms]
        sulfur_atoms = []
        carbon_atoms = []
        for a in res.atoms:
            elem = getattr(a, 'element', None)
            sym = (elem.symbol if elem is not None else (a.name.strip()[0].upper() if a.name else None))
            if sym == 'S':
                sulfur_atoms.append(a)
            if sym == 'C':
                carbon_atoms.append(a)
        if not sulfur_atoms or not carbon_atoms:
            continue
        for s_atom in sulfur_atoms:
            s_idx = s_atom.index
            s_pos = xyz[s_idx]
            for c_atom in carbon_atoms:
                c_idx = c_atom.index
                c_pos = xyz[c_idx]
                sc_dist = float(np.linalg.norm(c_pos - s_pos))
                if sc_dist > s_c_cutoff_nm:
                    continue
                # Count heavy neighbors around this carbon within covalent range among same-residue atoms
                heavy_neighbors = []
                for ni in res_atom_indices:
                    if ni == c_idx:
                        continue
                    elem = top.atom(ni).element
                    if elem is None or elem.symbol == 'H':
                        continue
                    dist = float(np.linalg.norm(xyz[ni] - c_pos))
                    if dist <= heavy_cutoff_nm:
                        heavy_neighbors.append(ni)
                if len(heavy_neighbors) == 1 and heavy_neighbors[0] == s_idx:
                    return c_idx

    raise ValueError("Could not find methyl carbon bonded to sulfur in SAM/ligand (bonds or distances).")

def nearest_sidechain_nitrogen_to_atom(
    traj: md.Trajectory,
    target_atom_index: int,
    chain: md.core.topology.Chain,
) -> Tuple[md.core.topology.Residue, int, float]:
    """
    For frame 0, among side-chain nitrogens of `chain`, find the atom nearest to `target_atom_index`.
    Returns (residue, nitrogen_atom_index, min_distance_nm).
    """
    xyz = traj.xyz[0]
    target_xyz = xyz[target_atom_index]

    candidate_indices = get_sidechain_nitrogen_indices(chain, include_n_term_backbone=True)
    if not candidate_indices:
        raise RuntimeError("No side-chain nitrogen atoms found in the specified chain.")

    candidate_xyz = xyz[candidate_indices]
    dists = np.linalg.norm(candidate_xyz - target_xyz, axis=1)
    i_min = int(np.argmin(dists))
    nearest_atom_idx = candidate_indices[i_min]
    nearest_atom = traj.topology.atom(nearest_atom_idx)
    nearest_residue = nearest_atom.residue
    min_dist_nm = float(dists[i_min])
    return nearest_residue, nearest_atom_idx, min_dist_nm

def process_pdb(pdb_path: str) -> Tuple[str, str, str, float]:
    """
    Compute distance between SAM C41 and nearest side-chain nitrogen on histone H3.
    Returns (Boltz_ID, NearestResidue, DistanceAngstrom).
    """
    basename = os.path.basename(pdb_path)
    boltz_id = basename.split('_', 1)[0]

    traj = md.load(pdb_path)
    h3_chain = find_h3_chain(traj.topology)

    target_idx = find_sam_methyl_carbon_index(traj)
    nearest_res, nearest_atom_idx, min_dist_nm = nearest_sidechain_nitrogen_to_atom(
        traj, target_idx, h3_chain
    )

    distance_angstrom = float(min_dist_nm * 10.0)
    nearest_residue_str = f"{nearest_res.name}{nearest_res.resSeq}"
    nearest_atom_name = traj.topology.atom(nearest_atom_idx).name
    return boltz_id, nearest_residue_str, nearest_atom_name, distance_angstrom

def main(folder: str = "complex_structures", output_csv: str = "nearest_H3_distances.csv"):
    pdb_files = sorted(glob.glob(os.path.join(folder, "*.pdb")))
    if not pdb_files:
        print(f"No .pdb files found in {folder}")
        return

    rows = []
    print(f"Found {len(pdb_files)} PDB files in {folder}")
    for pdb_path in pdb_files:
        try:
            boltz_id, nearest_residue, atom_name, dist_ang = process_pdb(pdb_path)
            rows.append((boltz_id, nearest_residue, atom_name, dist_ang))
            print(f"{os.path.basename(pdb_path)} -> {boltz_id}, {nearest_residue}, {atom_name}, {dist_ang:.2f} Å")
        except Exception as e:
            basename = os.path.basename(pdb_path)
            boltz_id = basename.split('_', 1)[0]
            print(f"ERROR processing {basename}: {e}")
            rows.append((boltz_id, "NA", "NA", ""))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Boltz_ID", "Nearest_Residue", "Atom_Name", "Distance_Angstrom"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    out_csv = "nearest_H3_distances.csv"
    folder = "complex_structures"
    main(folder=folder, output_csv=out_csv)
