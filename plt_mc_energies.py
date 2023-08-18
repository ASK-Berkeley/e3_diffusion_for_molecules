import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import torch
from actual_mcmc import get_energy, run_mcmc
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem
import ase
from multiprocessing import Pool

from qm9.rdkit_functions import build_molecule
from configs.datasets_config import get_dataset_info

d50_dir = "outputs/qm9_mc/flexible_mols/diffusion/T50/0074"

def load_energy_file(fn, burnin):
    energies = []
    with open(fn, "r") as f:
        for i, line in enumerate(f):
            if i < burnin:
                continue
            energies.append(float(line))
    return energies

def get_mcmc_energies(temperature, molid, burnin=5000):
    mol_dir = "outputs/qm9_mc/flexible_mols/diffusion/T01/{}".format(molid)
    #mcmc_dir = "outputs/qm9_mc/flexible_mols/mcmc/T{}/0074".format(temperature)
    #energy_fn = "psi4_mcmc_energies_{}.txt".format(temperature)
    #return load_energy_file(os.path.join(mcmc_dir, energy_fn), burnin)

    gs_fn = os.path.join(mol_dir, "gs.xyz")
    #gs_fn = "outputs/qm9_mc/flexible_mols/mcmc/T75/0074/gs.xyz"
    #energies, _, _, _ = run_mcmc(gs_fn, temperature, 24000, out_dir=None)
    energies, _, _, _ = run_mcmc(gs_fn, temperature, 10000, out_dir=None)
    return energies[burnin:]

def get_diffusion_energies(start_T, molid, burnin=1000):
    mol_dir = "outputs/qm9_mc/flexible_mols/diffusion/T{}/{}".format(start_T, molid)
    #energy_fn = "energies.txt"
    #return load_energy_file(os.path.join(diffusion_dir, energy_fn), burnin)

    # Get gs molecule first
    gs_fn = os.path.join(mol_dir, "gs.xyz")
    gs_atoms = ase.io.read(gs_fn)
    xyz = torch.tensor(gs_atoms.get_positions())
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, gs_fn)

    # Convert the Open Babel molecule to an RDKit molecule
    mol_block = obConversion.WriteString(mol)
    rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)

    """
    dataset_info = get_dataset_info("qm9", False)
    atom_encoder = dataset_info["atom_encoder"]
    gs_atoms = ase.io.read(gs_fn)
    atom_types = [atom_encoder[a] for a in gs_atoms.get_chemical_symbols()]
    atom_types = torch.tensor(atom_types)
    xyz = torch.tensor(gs_atoms.get_positions())
    rdkit_mol = build_molecule(xyz, atom_types, dataset_info)
    """
    Chem.SanitizeMol(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol)
    conf = rdkit_mol.GetConformer()
    for i in range(xyz.shape[0]):
        conf.SetAtomPosition(i, xyz[i].numpy())

    gs_energy = get_energy(rdkit_mol)
    print("gs_energy", gs_energy)

    energies = []
    #fns = sorted(glob.glob(os.path.join(diffusion_dir, "step_*.xyz")))
    #for fn in fns:
    chain = np.load(os.path.join(mol_dir, "chain.npz"))
    for i in range(burnin, chain["xyz_chain"].shape[0]):
        # Load the molecule with Open Babel and infer the bonds
        #atoms = ase.io.read(fn)
        positions = chain["xyz_chain"][i]
        conf = rdkit_mol.GetConformer()
        for i in range(positions.shape[0]):
            conf.SetAtomPosition(i, positions[i].tolist())

        energies.append(get_energy(rdkit_mol) - gs_energy)

    return energies


#mmff_50_energies = get_energies(os.path.join(d50_dir, "mmff_energies_50.txt"), 200)
#mmff_75_energies = get_energies(os.path.join(d50_dir, "mmff_energies_75.txt"), 200)
#mmff_100_energies = get_energies(os.path.join(d50_dir, "mmff_energies_100.txt"), 200)

bins = np.linspace(0, 15, 50)


#plt.plot(get_mcmc_energies(75))
#plt.show()

#d65 = get_diffusion_energies(65)
d50 = np.array(get_diffusion_energies("20", "0074", 15000))

#plt.hist(get_diffusion_energies(75), label="diffusion75", bins=bins, alpha=0.5)
#plt.hist(d65, label="diffusion65", bins=bins, alpha=0.5)
#plt.hist(get_diffusion_energies(60), label="diffusion60", bins=bins, alpha=0.5)
plt.hist(d50, label="diffusion50", bins=bins, alpha=0.5)

#plt.hist(mmff_100_energies, label="mmff", bins=bins, alpha=0.5)
#plt.hist(mmff_75_energies, label="mmff", bins=bins, alpha=0.5)
x = get_mcmc_energies(100, "0074", 5000)
plt.hist(x, label="mcmc", bins=bins, alpha=0.5)#, weights=0.2*np.ones_like(x))
#plt.hist(get_mcmc_energies(100, 2000), label="mcmc_100", bins=bins, alpha=0.5)
plt.legend()
plt.tight_layout()
#plt.savefig("writeup/figures/boltzmann_comparison.pdf")
plt.show()

