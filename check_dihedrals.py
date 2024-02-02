import os
import glob
import ase
from ase.io import read
from ase import neighborlist
from ase import Atoms
import numpy as np
from actual_mcmc import get_rotatable_bonds
from openbabel import openbabel as ob
import io
from rdkit import Chem

def ase_to_rdkit(atoms):
    bytesio = io.BytesIO(b"")
    txtwrapper = io.TextIOWrapper(bytesio, write_through=True, errors="replace")
    ase.io.write(txtwrapper, atoms, format="xyz")
    bytesio.seek(0)
    xyz_string = bytesio.read().decode("utf-8")
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")
    mol = ob.OBMol()
    obConversion.ReadString(mol, xyz_string)

    # Convert the Open Babel molecule to an RDKit molecule
    mol_block = obConversion.WriteString(mol)
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    return mol


def main():
    Ts = [40, 50, 60, 70, 80, 90, 100, 125, 150]
    for T in Ts:
        base_dir = "outputs/qm9_mc/flexible_mols/diffusion/T{:0>2d}".format(T)
        for mol_dir in glob.glob(os.path.join(base_dir, "040*")):
            mol_id = os.path.basename(mol_dir)
            # compute rotatable bonds from ground state positions
            gs_atoms = read(os.path.join(mol_dir, "gs.xyz"))
            rotatable_bonds = get_rotatable_bonds(ase_to_rdkit(gs_atoms))

            chain_fn = os.path.join(mol_dir, "chain.npz")
            chain = np.load(chain_fn)
            atomic_nums = chain["atomic_nums"]
            xyz = chain["xyz_chain"]
            dihedrals = []
            for i in range(xyz.shape[0]):
                atoms = Atoms(numbers=atomic_nums, positions=xyz[i])
                dihedrals.append([atoms.get_dihedral(*bond) for bond in rotatable_bonds])
            out_fn = "diff_{}_T{}_dihedrals.npy".format(mol_id, T)
            np.save(os.path.join(mol_dir, out_fn), np.array(dihedrals))
            #np.save(os.path.join(mol_dir, "dihedrals"), np.array(dihedrals))



"""
def get_rotatable_bonds(atoms):
    # Get the neighbor list
    cutoff = neighborlist.natural_cutoffs(atoms)
    nl = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Identify potentially rotatable bonds
    rotatable_bonds = []
    for atom in atoms:
        indices, offsets = nl.get_neighbors(atom.index)
        # Only consider bonds where the current atom is heavy (i.e., not Hydrogen)
        if atom.symbol != 'H':
            for i in indices:
                # Check that the other atom in the bond is also heavy
                if atoms[i].symbol != 'H':
                    # Check if atoms are not in a ring and are not terminal atoms
                    d = atoms.get_distance(atom.index, i)
                    if d > 1.15 and d < 1.6:
                        bond = sorted((atom.index, i))
                        # Check if the bond is already in the list
                        if bond not in rotatable_bonds:
                            rotatable_bonds.append(bond)
    return rotatable_bonds, nl

def get_all_dihedrals(atoms, nl, rotatable_bonds):
    dihedrals = [] 

    # Compute dihedral angles
    for bond in rotatable_bonds:
        begin_atom = bond[0]
        end_atom = bond[1]

        # Get the neighboring atoms
        begin_neighbors = [a for a in nl.get_neighbors(begin_atom)[0] if a != end_atom]
        end_neighbors = [a for a in nl.get_neighbors(end_atom)[0] if a != begin_atom]

        # Skip if we don't have enough atoms to form a dihedral
        if len(begin_neighbors) < 1 or len(end_neighbors) < 1:
            continue

        begin_neighbor = begin_neighbors[0]
        end_neighbor = end_neighbors[0]

        # Get the atom positions
        pos1 = atoms[begin_neighbor].position
        pos2 = atoms[begin_atom].position
        pos3 = atoms[end_atom].position
        pos4 = atoms[end_neighbor].position

        # Compute the dihedral angle
        dihedrals.append(atoms.get_dihedral(begin_neighbor, begin_atom, end_atom, end_neighbor))
    return dihedrals
"""

if __name__ == "__main__":
    main()
