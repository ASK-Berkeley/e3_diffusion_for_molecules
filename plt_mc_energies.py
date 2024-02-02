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
import pickle
from itertools import repeat
import copy

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

def get_gaussian_energies(sigma, molid, gs_energy, burnin=5000):
    mol_dir = "outputs/qm9_mc/flexible_mols/diffusion/T01/{}".format(molid)
    gs_fn = os.path.join(mol_dir, "gs.xyz")
    gs_atoms = ase.io.read(gs_fn)
    atoms = gs_atoms
    gs_pos = copy.deepcopy(gs_atoms.positions)

    energies = []
    for i in range(19000):
        delta = sigma * np.random.randn(*atoms.positions.shape)
        #atoms.positions = atoms.positions + delta
        atoms.positions = gs_pos + delta
        energies.append(get_energy(None, atoms=gs_atoms) - gs_energy)
        if np.isnan(energies[-1]):
            print("{} / {}".format(np.isnan(energies).sum(), len(energies)))

    return energies[burnin:]

def get_mcmc_energies(temperature, molid, burnin=5000):
    mol_dir = "outputs/qm9_mc/flexible_mols/diffusion/T01/{}".format(molid)
    #mcmc_dir = "outputs/qm9_mc/flexible_mols/mcmc/T{}/0074".format(temperature)
    #energy_fn = "psi4_mcmc_energies_{}.txt".format(temperature)
    #return load_energy_file(os.path.join(mcmc_dir, energy_fn), burnin)

    gs_fn = os.path.join(mol_dir, "gs.xyz")
    #gs_fn = "outputs/qm9_mc/flexible_mols/mcmc/T75/0074/gs.xyz"
    energies, _, _, _ = run_mcmc(gs_fn, temperature, 24000, out_dir=None)
    return np.array(energies[burnin:])

def get_diffusion_energies(start_T, molid, gs_energy, burnin=1000):
    mol_dir = "outputs/qm9_mc/flexible_mols/diffusion/T{:0>2d}/{}".format(start_T, molid)
    #energy_fn = "energies.txt"
    #return load_energy_file(os.path.join(diffusion_dir, energy_fn), burnin)
    gs_fn = "outputs/qm9_mc/flexible_mols/diffusion/T01/{}/gs.xyz".format(molid)
    atoms = ase.io.read(gs_fn)

    energies = []
    #fns = sorted(glob.glob(os.path.join(diffusion_dir, "step_*.xyz")))
    #for fn in fns:
    chain = np.load(os.path.join(mol_dir, "chain.npz"))
    for i in range(burnin, chain["xyz_chain"].shape[0]):
        # Load the molecule with Open Babel and infer the bonds
        #atoms = ase.io.read(fn)
        positions = chain["xyz_chain"][i]
        atoms.positions = positions

        energies.append(get_energy(None, atoms=atoms) - gs_energy)

    return np.array(energies)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    #mmff_50_energies = get_energies(os.path.join(d50_dir, "mmff_energies_50.txt"), 200)
    #mmff_75_energies = get_energies(os.path.join(d50_dir, "mmff_energies_75.txt"), 200)
    #mmff_100_energies = get_energies(os.path.join(d50_dir, "mmff_energies_100.txt"), 200)

    start_Ts = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]#, 60, 70, 80, 90, 100, 125, 150]
    temperatures = [10, 20, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400]
    sigmas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    sigmas = [0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002, 0.004, 0.007, 0.01, 0.2]

    molids = ["0074", "0403", "0550", "0622", "0777", "0832", "1050", "1259", "1289", "1408"]
    all_diffusion_energies = {}
    all_mcmc_energies = {}
    all_gaussian_energies = {}

    #pool = Pool(len(start_Ts) + len(temperatures) + len(sigmas))
    pool = Pool(44)
    results_d = {}
    results_m = {}
    results_g = {}
    for molid in molids:
        # Get gs molecule first
        gs_fn = "outputs/qm9_mc/flexible_mols/diffusion/T01/{}/gs.xyz".format(molid)
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


        #results_d[molid] = pool.starmap_async(get_diffusion_energies,
        #                                      zip(start_Ts, repeat(molid), repeat(gs_energy), repeat(1000)))

        #for start_T in start_Ts:
        #    diffusion_energies = get_diffusion_energies(start_T, molid, 19990)
        #    all_diffusion_energies[start_T] = np.array(diffusion_energies)

        #results_m[molid] = pool.starmap_async(get_mcmc_energies,
        #                                      zip(temperatures, repeat(molid), repeat(5000)))

        #for t in temperatures:
        #    mcmc_energies = get_mcmc_energies(t, molid, 0)
        #    all_mcmc_energies[t] = np.array(mcmc_energies)

        results_g[molid] = pool.starmap_async(get_gaussian_energies,
                                              zip(sigmas, repeat(molid), repeat(gs_energy), repeat(0)))

    for molid in molids:
        #all_diffusion_energies[molid] = {start_T: r for start_T, r in zip(start_Ts, results_d[molid].get())}
        #all_mcmc_energies[molid] = {t: r for t, r in zip(temperatures, results_m[molid].get())}
        all_gaussian_energies[molid] = {s: r for s, r in zip(sigmas, results_g[molid].get())}

    pool.close()
    pool.join()

    with open("mcmc_diffusion_energies_allmol_gaussian2_kcal.pkl", "wb") as f:
        pickle.dump({"gaussian_energies": all_gaussian_energies}, f)
        #pickle.dump({"diffusion_energies": all_diffusion_energies,
        #             "mcmc_energies": all_mcmc_energies,
        #             "gaussian_energies": all_gaussian_energies}, f)
    exit()
    # wrong
    # d: 11, 14, 17, 20
    # m: 0, 3, 8, 13

    # d: 3, 7, 10, 11
    # m: 1, 4, 9, 15
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

