from matplotlib import pyplot as plt
import numpy as np
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromXYZBlock
from psi4_chain import get_ef
import ase
from ase import Atoms
from ase.visualize import view
import os
import time

# Load the molecule

def main():
    #0074  0403  0550  0622  0777  ->0832  1050  1259  1289  1408
    gs_fn = "outputs/qm9_mc/flexible_mols/T75/0074/gs.xyz"
    out_dir = "outputs/qm9_mc/flexible_mols/mcmc/T75/0074/"
    temperature = 350  # in Kelvin
    energies, dihedral, all_atoms, accept_rate = run_mcmc(gs_fn, temperature, out_dir=None)
    print("accept:", accept_rate)
    plt.hist(energies[2000:], bins=50)

    plt.figure()
    plt.plot(energies)
    plt.figure()
    plt.plot(dihedral)
    plt.show()
    view(all_atoms)

    return
    out_fn = os.path.join(out_dir, "psi4_mcmc_energies_{}.txt".format(temperature))
    with open(out_fn, "w") as f:
        for e in energies:
            f.write(str(e) + "\n")
    with open(os.path.join(out_dir, "accept_rate.txt"), "w") as f:
        f.write(str(accept_rate))


def rdkit_to_ase(rdkit_mol):
    """
    Convert a RDKit molecule object to an ASE atoms object.
    :param rdkit_mol: RDKit Molecule object
    :return: ASE atoms object
    """

    # Ensure that the molecule has at least one conformer
    if rdkit_mol.GetNumConformers() == 0:
        raise ValueError('Molecule must have at least one conformer')

    # Get the first (0th) conformer
    conformer = rdkit_mol.GetConformer(0)

    # Get atom symbols and positions
    symbols = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
    positions = [conformer.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())]
    positions = np.array([(p.x, p.y, p.z) for p in positions])

    # Create an ASE atoms object
    ase_atoms = Atoms(symbols=symbols, positions=positions)

    return ase_atoms

def get_energy(mol):
    #return AllChem.UFFGetMoleculeForceField(mol).CalcEnergy()

    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    # Get the energy of the current state
    return ff.CalcEnergy()

    e, f = get_ef(rdkit_to_ase(mol), num_threads=32)
    return e

def run_mcmc(gs_fn, temperature, out_dir):
    # Load the molecule with Open Babel and infer the bonds
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, gs_fn)

    # Convert the Open Babel molecule to an RDKit molecule
    mol_block = obConversion.WriteString(mol)
    rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)

    #rdkit_mol = Chem.MolFromSmiles("CC")
    #rdkit_mol = AllChem.AddHs(rdkit_mol)
    #AllChem.EmbedMolecule(rdkit_mol)

    # Get ground-state energy first
    gs_energy = get_energy(rdkit_mol)

    #bonds = rdkit_mol.GetBonds()
    #for i, bond in enumerate(bonds):
    #    print(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())


    """
    with open(file_path, "r") as f:
        lines = []
        for i, line in enumerate(f):
            if i > 1:
                lines.append(line[:-15])
            else:
                lines.append(line.strip())
    molecule = Chem.MolFromXYZBlock("\n".join(lines))

    #molecule = MolFromXYZFile(file_path)


    # Initialize the forcefield
    ff = AllChem.MMFFGetMoleculeForceField(molecule, AllChem.MMFFGetMoleculeProperties(molecule))
    """

    # Initialize an empty list to save the energies
    energies = []
    dihedral = []
    all_atoms = []

    # MCMC parameters
    n_steps = (5000 - 500) * 5 + 2000
    k_B = 0.0019872041  # Boltzmann constant in kcal/mol/K

    energy = gs_energy
    n_accepts = 0
    n_rejects = 0
    # Run the MCMC
    start_time = time.time()
    for step in range(n_steps):

        energies.append(energy - gs_energy)

        # Generate a new state: here we're just performing a random perturbation of the positions
        conf = rdkit_mol.GetConformer()
        old_positions = []
        for i in range(rdkit_mol.GetNumAtoms()):
            position = np.array(conf.GetAtomPosition(i))
            old_positions.append(position)
            new_position = position + 4e-2 * np.random.normal(0, np.sqrt(k_B * temperature), 3)
            conf.SetAtomPosition(i, new_position)

        # Get the energy of the new state
        new_energy = get_energy(rdkit_mol)

        # Accept or reject the new state based on the Metropolis criterion
        if np.random.rand() < np.exp((energy - new_energy) / (k_B * temperature)):
            # The new state is accepted, do nothing
            #print("!", end="")
            energy = new_energy
            n_accepts += 1
        else:
            # The new state is rejected, revert to the old positions
            #print("x", end="")
            for i in range(rdkit_mol.GetNumAtoms()):
                conf.SetAtomPosition(i, old_positions[i])
            n_rejects += 1


        atoms = rdkit_to_ase(rdkit_mol)
        all_atoms.append(atoms)
        dihedral.append(atoms.get_dihedral(2, 0, 1, 6))
        if out_dir is not None:
            ase.io.write(os.path.join(out_dir, "step_{:0>4d}.xyz".format(step)), atoms)


    print("time", time.time() - start_time)
    # Convert the energies to a numpy array for further processing
    energies = np.array(energies)
    accept_rate = n_accepts / (n_accepts + n_rejects)
    return energies, dihedral, all_atoms, accept_rate

if __name__ == "__main__":
    main()