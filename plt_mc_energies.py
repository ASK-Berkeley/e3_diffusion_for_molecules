import os
import glob
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from actual_mcmc import get_energy, run_mcmc
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem
import ase
from multiprocessing import Pool
import pickle

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

def parse_energies(energies, molid):
    diffusion_energies = energies["diffusion_energies"][molid]
    mcmc_energies = energies["mcmc_energies"][molid]
    gaussian_energies = energies["gaussian_energies"][molid]

    Ts = np.array(sorted(diffusion_energies.keys()))
    diffusion_offset = diffusion_energies[1].min()
    d = {T: diffusion_energies[T] - diffusion_offset for T in Ts}
    avg_d = np.array([d[T].mean() for T in Ts])
    std_d = np.array([d[T].std() for T in Ts])

    ts = np.array(sorted(mcmc_energies.keys()))
    mcmc_offset = mcmc_energies[10].min()
    print("Offsets (d-m): {} - {} = {}".format(diffusion_offset, mcmc_offset, diffusion_offset - mcmc_offset))
    m = {t: mcmc_energies[t] - mcmc_offset for t in ts}
    avg_m = np.array([m[t].mean() for t in ts])
    std_m = np.array([m[t].std() for t in ts])

    sigmas = np.array(sorted(gaussian_energies.keys()))
    g = {s: gaussian_energies[s] for s in sigmas}
    avg_g = np.array([np.mean(g[s]) for s in sigmas])
    std_g = np.array([np.std(g[s]) for s in sigmas])

    return Ts, avg_d, std_d, ts, avg_m, std_m, sigmas, avg_g, std_g


def main():
    with open("mcmc_diffusion_energies_allmol_kcal.pkl", "rb") as f:
        energies = pickle.load(f)
    fig, axes = plt.subplots(2, 5, figsize=(8,4))
    for axid, molid in enumerate(energies["diffusion_energies"]):
        if axid == 8:
            axis = axes[axid % 2, axid % 5]
            axis.set_yticks([])
            axis.set_xticks([])
            legend_elems = [Patch(facecolor='C0', label='Diffusion', alpha=0.5),
                            Patch(facecolor='C1', label='MCMC', alpha=0.5)]
            axis.legend(handles=legend_elems, loc='upper center')
            continue
        Ts = np.array(sorted(energies["diffusion_energies"][molid].keys()))
        ts = np.array(sorted(energies["mcmc_energies"][molid].keys()))

        diffusion_energies = energies["diffusion_energies"][molid]
        mcmc_energies = energies["mcmc_energies"][molid]
        gaussian_energies = energies["gaussian_energies"][molid]

        diffusion_offset = diffusion_energies[1].min()
        mcmc_offset = mcmc_energies[10].min()

        for T in diffusion_energies:
            diffusion_energies[T] -= diffusion_offset
        for t in mcmc_energies:
            mcmc_energies[t] -= mcmc_offset

        bins = np.linspace(-2, 40, 200)
        #bins = np.logspace(np.log10(0.01), np.log10(40), 200)
        axis = axes[axid % 2, axid % 5]
        axis.set_yscale("log")
        print(axid % 2, axid % 5)

        T_idxs = [3, 7, 10, 11]
        axis.violinplot([diffusion_energies[Ts[i]] for i in T_idxs], showextrema=False)
        #for i in [2, 7, 10, 11]:
            #axis.hist(diffusion_energies[Ts[i]], bins=bins, alpha=0.5, color="C0")

        t_axis = axis.twiny()
        t_idxs = [1, 5, 9, 15]
        parts = t_axis.violinplot([mcmc_energies[ts[i]] for i in t_idxs], showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('C1')
        #for i in [1, 5, 9, 15]:
        #    axis.hist(mcmc_energies[ts[i]], bins=bins, alpha=0.5, color="C1")
        #axis.set_ylim(10, 20000)
        #axis.set_yscale("log")
        #axis.set_xscale("log")
        if axid % 2 != 1:
            #axis.set_xticks([])
            t_axis.set_xlabel("Temperature (K)")
            t_axis.set_xticks(np.arange(len(t_idxs)) + 1, labels=[ts[i] for i in t_idxs])
            axis.set_xticks([])
        else:
            #axis.set_xlabel("kcal/mol")
            axis.set_xlabel("Diffusion Step")
            axis.set_xticks(np.arange(len(T_idxs)) + 1, labels=[Ts[i] for i in T_idxs])
            t_axis.set_xticks([])
        if axid % 5 != 0:
            axis.set_yticks([])
        else:
            #axis.set_ylabel("Count")
            axis.set_ylabel("kcal/mol")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("writeup/figures/mcmc_violins.pdf")
    plt.show()
    exit()

    for molid in energies["diffusion_energies"]:
        if molid != "0074":
            continue

        Ts, avg_d, std_d, ts, avg_m, std_m, sigmas, avg_g, std_g = parse_energies(energies, molid)
        Ts = Ts[:-1]
        avg_d = avg_d[:-1]
        std_d = std_d[:-1]

        plt.figure(figsize=(4,3))
        T_ax = plt.gca()
        t_ax = plt.twiny()
        #s_ax = plt.twiny()

        Ts = Ts[::-1]
        line1 = T_ax.plot(Ts, avg_d[::-1], color="C0", label="Diffusion", alpha=0.5)
        T_ax.scatter(Ts, avg_d[::-1], color="C0", alpha=0.5)
        T_ax.fill_between(Ts, (avg_d - std_d)[::-1], (avg_d + std_d)[::-1], color="C0", alpha=0.2)
        T_ax.set_xlabel("Diffusion Step")
        #T_ax.invert_xaxis()
        #T_ax.set_xlim(-5, 160)

        line2 = t_ax.plot(ts, avg_m, color="C1", label="MCMC", alpha=0.5)
        t_ax.scatter(ts, avg_m, color="C1", alpha=0.5)
        t_ax.fill_between(ts, avg_m - std_m, avg_m + std_m, color="C1", alpha=0.2)
        t_ax.set_xlabel("Temperature")
        T_ax.set_ylabel("Energy (kcal/mol)")

        #line3 = s_ax.plot(sigmas, avg_g, color="C2", label="Gaussian")
        #s_ax.scatter(sigmas, avg_g, color="C2")
        #s_ax.fill_between(sigmas, avg_g - std_g, avg_g + std_g, color="C2", alpha=0.2)
        #s_ax.set_xlabel("Sigma")

        lines = line2 + line1
        labels = [l.get_label() for l in lines]
        t_ax.legend(lines, labels, loc="upper left")
        plt.tight_layout()
        #plt.savefig("writeup/figures/mcmc_diffusion_mol0.pdf")
        #plt.show()

    all_ts = []
    all_Ts = []
    all_avg_d = []
    all_avg_m = []

    for molid in energies["diffusion_energies"]:
        Ts, avg_d, std_d, ts, avg_m, std_m, sigmas, avg_g, std_g = parse_energies(energies, molid)
        Ts = Ts[:-1]
        avg_d = avg_d[:-1]
        std_d = std_d[:-1]

        all_ts += ts.tolist()
        all_Ts += Ts.tolist()
        all_avg_d += avg_d.tolist()
        all_avg_m += avg_m.tolist()

    slope_d = np.linalg.lstsq(np.array(all_Ts)[:,None]**2, np.array(all_avg_d), rcond=None)[0][0]
    slope_m = np.linalg.lstsq(np.array(all_ts)[:,None], np.array(all_avg_m), rcond=None)[0][0]
    print("slope ratio:", slope_m / slope_d)

    #plt.figure(figsize=(4,3))
    fig, axes = plt.subplots(2, 5, figsize=(8,4))
    for axid, molid in enumerate(energies["diffusion_energies"]):
        #if molid != "0074":
        #    continue

        Ts, avg_d, std_d, ts, avg_m, std_m, sigmas, avg_g, std_g = parse_energies(energies, molid)
        Ts = Ts[:-1]
        avg_d = avg_d[:-1]
        std_d = std_d[:-1]
        #sigmas = np.array(sorted(gaussian_energies.keys()))
        #g = {t: np.nan_to_num(gaussian_energies[s], nan=np.nanmax(gaussian_energies[s]))
        #     for s in sigmas}
        #avg_g = np.array([gaussian_energies[s].mean() for s in sigmas])
        #std_g = np.array([gaussian_energies[s].std() for s in sigmas])

        # y = slope_d * Ts**2
        # y = slope_m * ts

        #slope_d = np.linalg.lstsq(Ts[:,None]**2, avg_d, rcond=None)[0][0]
        #slope_m = np.linalg.lstsq(ts[:,None], avg_m, rcond=None)[0][0]

        # T**2 = t * slope_m / slope_d
        def t2T(t):
            return (t * slope_m / slope_d)**0.5
        #print(min([diffusion_energies[T].min() for T in Ts]))

        T_ax = axes[axid % 2, axid % 5]
        #T_ax = plt.gca()
        t_ax = T_ax.twiny()

        plt_x = Ts**2 * slope_d / slope_m

        #Ts = (1000 - Ts)[::-1]

        if axid != 8:
            t_ax.plot(plt_x, avg_d[::1], alpha=0.5, color="C0", zorder=1)
            #plt.plot([Ts.min(), Ts.max()], [slope_m * Ts.min(), slope_m * Ts.max()])
            t_ax.scatter(plt_x, avg_d[::1], alpha=0.5, color="C0", zorder=2)
            t_ax.fill_between(plt_x, (avg_d - std_d)[::1], (avg_d + std_d)[::1], color="C0", alpha=0.2)
            #T_ax.set_xlabel("Diffusion Step")

            t_ax.plot(ts, avg_m, color="C1", alpha=0.5, zorder=3)
            t_ax.scatter(ts, avg_m, color="C1", alpha=0.5, zorder=4)
            t_ax.fill_between(ts, avg_m - std_m, avg_m + std_m, color="C1", alpha=0.2)
            #t_ax.set_xlabel("Temperature")
            #T_ax.set_ylabel("Energy (kcal/mol)")
        #t_ax.set_xlim(-25, 540)

        max_T = 41
        t_ax.set_xlim(0, max_T**2 * slope_d / slope_m)

        t_ax.set_ylim(-5, 45)
        T_ax.set_xlim(0, max_T)
        # label = sqrt(T / 40) * 40
        # T = (label / 40)**2 * 40
        # desired_labels = [0, 14, 20, 25, 30, 35, 40]
        if axid == 5:
            desired_labels = [0, 20, 30, 40]
        else:
            desired_labels = [20, 30, 40]
        fake_x = [max_T * (desired_label / max_T)**2 for desired_label in desired_labels]
        T_ax.set_xticks(fake_x)
        T_ax.set_xticklabels([str(round((T/max_T)**0.5 * max_T)) for T in fake_x])
        #T_ax.set_xlim(1000+5, 1000 - 540**0.5 * slope_m / slope_d)
        #T_ax.invert_xaxis()
        #print("Ts:", Ts)
        #tmp = np.hstack(Ts**0.5 * slope_m / slope_d)
        #print("tmp:", tmp)
        #T_ax.set_xticks(tmp)
        #tmp = [str(1000 - T) for T in Ts]
        #print("tmp2:", tmp)
        #T_ax.set_xticklabels(tmp)

        if axid % 2 != 1:
            t_ax.set_xlabel("Temp. (K)")
            T_ax.set_xlabel("")
            T_ax.set_xticks([])
            #T_ax.set_xlabel("Diffusion Step")
        else:
            t_ax.set_xlabel("")
            t_ax.set_xticks([])
            T_ax.set_xlabel("Diffusion Step")
        if axid % 5 != 0:
            T_ax.set_yticks([])
            T_ax.set_ylabel("")
        else:
            T_ax.set_ylabel("kcal/mol")



        #plt.plot(sigmas, avg_m, color="C2")
        #plt.fill_between(sigmas, avg_s - std_s, avg_s + std_s, color="C2", alpha=0.2)
        #plt.tight_layout()
        #plt.show()
        #plt.savefig("writeup/figures/mcmc_diffusion_mol0_sqrt.pdf")
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    #plt.savefig("writeup/figures/mcmc_diffusion_sqrt.pdf")


if __name__ == "__main__":
    main()
    exit()
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

