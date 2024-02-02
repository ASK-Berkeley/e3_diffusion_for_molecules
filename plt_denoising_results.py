import pickle
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from ase.io import write
from ase import units
from ase import Atoms
from ase.visualize import view
from actual_mcmc import get_rotatable_bonds
from test_denoising import xyz_to_rdkit
import copy

matplotlib.use("QtAgg")

geom_failed_idxs = [22, 37, 41, 49]
geom_failed_idxs = [22, 37, 41, 49, 102, 190, 324, 414, 463, 482, 484, 490]

qm9_failed_idxs = [1, 10, 19, 29, 31, 43]

old_qm9_failed_idxs = [7, 26, 28, 29, 34, 39]

LEGACY = False

def load_results_pkl(fn, dataset="qm9"):
    with open(fn, "rb") as f:
        results = pickle.load(f)

    if len(results["initial_xyzs"]) == len(results["diffused_xyzs"]):
        # these .pkl files have None in mmff_* and diffused_* if the calculation
        # failed. So we need to remove those Nones and the corresponding entries
        # in initial_* (which are not None because it's before the calculation failed)
        for i, nsteps in reversed(list(enumerate(results["diffused_nsteps"]))):
            if nsteps is None:
                for k in results.keys():
                    if "chain" in k:
                        continue
                    results[k].pop(i)
    else:
        # I included entries in initial_* that were excluded from mmff_* and diffused_*
        # because they failed along the way. So we need to remove those from initial_*
        # since otherwise initial_*[i] won't correspond to e.g. mmff_*[i]

        # remove failed relaxations
        if dataset == "qm9":
            if LEGACY:
                failed = old_qm9_failed_idxs
            else:
                failed = qm9_failed_idxs
        elif dataset == "geom":
            failed = geom_failed_idxs
        for k in results.keys():
            if "initial" in k or "symbols" in k:
                results[k] = [x for i, x in enumerate(results[k]) if i not in failed]

    # np.array-ify
    for k in results.keys():
        if k[-3:] == "_es" or k[-6:] == "nsteps":
            results[k] = np.array(results[k])


    return results

def get_fn_base(epoch, dataset):
    if dataset == "qm9":
        if epoch is None:
            return "denoising_results/denoising_results_qm9_sT{}.pkl"
        else:
            return "denoising_results/denoising_results_qm9{}".format(epoch) + "_sT{}.pkl"
    elif dataset == "geom":
        return "denoising_results/denoising_results_geom{}".format(epoch) + "_sT{}.pkl"


def main(epoch=None, dataset="qm9"):
    fn_base = get_fn_base(epoch, dataset)

    results20 = load_results_pkl(fn_base.format(20), dataset)
    results30 = load_results_pkl(fn_base.format(30), dataset)
    results50 = load_results_pkl(fn_base.format(50), dataset)
    #results20 = copy.deepcopy(results50)
    #results30 = copy.deepcopy(results50)

    if LEGACY:
        assert dataset == "qm9"

        results20 = load_results_pkl("denoising_results/denoising_results_sT20.pkl")
        results30 = load_results_pkl("denoising_results/denoising_results_sT20.pkl")
        results50 = load_results_pkl("denoising_results/denoising_results_qm9_sT50.pkl.bak")

        num_steps_sT20 = []
        with open("relaxation_steps_parsed_sT20.txt", "r") as f:
            for line in f:
                if "None" not in line:
                    num_steps_sT20.append(list(map(int, line.split())))

        num_steps_sT50 = []
        with open("relaxation_steps_parsed_sT50.txt", "r") as f:
            for line in f:
                steps = line.split()
                if steps[0] == "None":
                    continue
                else:
                    num_steps_sT50.append((int(steps[0]), -1, int(steps[2]), int(steps[3])))

        num_steps_sT20 = np.array(num_steps_sT20)
        num_steps_sT50 = np.array(num_steps_sT50)

        mmff_steps = num_steps_sT20[:,2]
        sT20_steps = num_steps_sT20[:,3]
        sT50_steps = num_steps_sT50[:,3]

        symbols = [['C' for _ in range(results20["initial_relaxed_xyzs"][i].shape[0])]
                   for i in range(len(results20["initial_relaxed_xyzs"]))]

    else:
        symbols = results20["all_symbols"]

        if epoch is not None:
            # load MMFF results from epoch 0 (didn't relcaluate for other epochs)
            fn_base = get_fn_base(0, dataset)

            results_mmff = load_results_pkl(fn_base.format(50), dataset)
            def add_mmff_key(results, mmff_results, key):
                if key not in results or len(results[key]) == 0 or all(r is None for r in results[key]):
                    results[key] = mmff_results[key]
            for r in [results20, results30, results50]:
                add_mmff_key(r, results_mmff, "mmff_nsteps")
                add_mmff_key(r, results_mmff, "mmff_relaxed_es")
                add_mmff_key(r, results_mmff, "mmff_es")


        mmff_steps = np.array(results20["mmff_nsteps"])
        sT20_steps = np.array(results20["diffused_nsteps"])
        sT50_steps = np.array(results50["diffused_nsteps"])

    #results_keys = ['initial_xyzs', 'initial_es', 'initial_fs', 'initial_relaxed_xyzs', 'initial_relaxed_es', 'initial_relaxed_fs', 'initial_relax_times', 'guess_xyzs', 'guess_es', 'guess_fs', 'guess_relaxed_xyzs', 'guess_relaxed_es', 'guess_relaxed_fs', 'guess_relax_times', 'mmff_xyzs', 'mmff_es', 'mmff_fs', 'mmff_relaxed_xyzs', 'mmff_relaxed_es', 'mmff_relaxed_fs', 'mmff_relax_times', 'diffused_xyzs', 'diffused_es', 'diffused_fs', 'diffused_relaxed_xyzs', 'diffused_relaxed_es', 'diffused_relaxed_fs', 'diffused_relax_times', 'chain_es', 'chain_fs']


    initial_atoms = [Atoms(symbols=s, positions=xyz)
                 for s, xyz in zip(symbols, results20["initial_relaxed_xyzs"])]
    mols = [xyz_to_rdkit(s, xyz)
            for s, xyz in zip(symbols, results20["initial_relaxed_xyzs"])]
    rotatable_bonds = [get_rotatable_bonds(mol) if mol is not None else []
                       for mol in mols]
    gs_dihedrals = [np.array([atoms.get_dihedral(*bond) for bond in rotatable_bonds[i]])
                    for i, atoms in enumerate(initial_atoms)]
    mmff_atoms = [Atoms(symbols=s, positions=xyz)
                  for s, xyz in zip(symbols, results50["mmff_relaxed_xyzs"])]
    mmff_dihedrals = [np.array([a.get_dihedral(*bond) for bond in r])
                      for a, r in zip(mmff_atoms, rotatable_bonds)]
    max_delta_angle = np.array([np.abs(((m - g + 180) % 360) - 180).max() if len(m) > 0 else 0
                                for m, g in zip(mmff_dihedrals, gs_dihedrals)])
    
    #for i in range(20):
    #    fn = "writeup/figures/structures/geom_{:0>2d}.png".format(i)
    #    a = Atoms(symbols=symbols[i], positions=results20["initial_xyzs"][i])
    #    write(fn, a, scale=100)
    #exit()

    #print((results20["mmff_es"] - results20["diffused_es"]).mean() * 23)
    #print((results50["mmff_es"] - results50["diffused_es"]).mean() * 23)

    delta_es20 = results20["diffused_relaxed_es"] - results20["mmff_relaxed_es"]
    #delta_es20 = results20["mmff_relaxed_es"] - results20["initial_relaxed_es"]
    delta_es20 /= (units.kcal / units.mol)

    delta_es30 = results30["diffused_relaxed_es"] - results30["mmff_relaxed_es"]
    delta_es30 /= (units.kcal / units.mol)

    delta_es50 = results50["diffused_relaxed_es"] - results50["mmff_relaxed_es"]
    #delta_es50 = results50["diffused_relaxed_es"] - results50["mmff_relaxed_es"]
    delta_es50 /= (units.kcal / units.mol)


    def prepare_chain_es(results):
        chain_es = results["chain_es"]
        return np.hstack([(results["mmff_es"] - results["initial_relaxed_es"])[:,None],
                          chain_es]) / (units.kcal / units.mol)

    chain_es20 = prepare_chain_es(results20)
    chain_es30 = prepare_chain_es(results30)
    chain_es50 = prepare_chain_es(results50)

    plt.figure(figsize=(4,3))
    #colors = ["r","orange","y","g","b","purple","k"]
    x20 = list(reversed(range(22)))
    x30 = list(reversed(range(32)))
    x50 = list(reversed(range(52)))
    for i in range(delta_es20.shape[0]):
        c = "C{}".format(i)
        plt.plot(x20, chain_es20[i], alpha=0.3, color=c, linestyle="-")
        plt.plot(x30, chain_es30[i], alpha=0.3, color=c, linestyle="--")
        plt.plot(x50, chain_es50[i], alpha=0.3, color=c, linestyle=":")
    plt.plot(x20, chain_es20.mean(axis=0), color="k", linestyle="-", label="N=20")
    plt.plot(x30, chain_es30.mean(axis=0), color="k", linestyle="--", label="N=30")
    plt.plot(x50, chain_es50.mean(axis=0), color="k", linestyle=":", label="N=50")
    plt.grid(True, axis="y")
    plt.gca().invert_xaxis()
    #plt.ylim(-1, 1)
    #plt.xlim(992,1001)
    #plt.xticks(x50[1::2])
    plt.xlabel("Diffusion Step")
    plt.ylabel("Relative Energy (kcal/mol)")
    plt.xlim(7, -1)
    plt.ylim(-5, 10)
    #plt.legend()
    plt.tight_layout()
    #plt.savefig("writeup/figures/relaxation_energies_zoom.pdf")
    plt.show()

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax2 = ax.twinx()
    ax.set_title("EDM Ckpt 5150")

    bar_width = 1/7
    x = np.arange(len(delta_es20))
    #ax.bar(x, results20["mmff_relax_times"], width=0.25, label="MMFF")
    #ax.bar(x+0.25, results20["diffused_relax_times"], width=0.25, label="MMFF+Diffusion")
    ax.bar(x,              max_delta_angle, width=bar_width, label="$\\Delta\\theta$")
    ax.bar(x+1*bar_width,  mmff_steps, width=bar_width, label="MMFF")
    ax.bar(x+2*bar_width,  sT20_steps, width=bar_width, label="MMFF+Diffusion20")
    ax.bar(x+3*bar_width,  sT50_steps, width=bar_width, label="MMFF+Diffusion50")
    ax2.bar(x+4*bar_width, delta_es20, width=bar_width, color="g", label="$\Delta E_{20}$")
    ax2.bar(x+5*bar_width, delta_es50, width=bar_width, color="b", label="$\Delta E_{50}$")
    ax.legend()
    ax2.legend(loc="upper left")

    #ax.set_ylim(-250, 1000)
    ax.set_ylim(-100, 300)
    ax2.set_ylim(-10, 30)
    ax.set_ylabel("Steps to solution")
    ax2.set_ylabel("Solution delta energy (kcal/mol)")
    plt.tight_layout()
    #plt.savefig("writeup/figures/relaxation_speedup.pdf")
    plt.show()

    median_speedup_sT20 = 100 * np.median(1 - (sT20_steps / mmff_steps))
    median_speedup_sT50 = 100 * np.median(1 - (sT50_steps / mmff_steps))
    mean_speedup_sT20 = 100 * np.mean(1 - (sT20_steps / mmff_steps))
    mean_speedup_sT50 = 100 * np.mean(1 - (sT50_steps / mmff_steps))
    print("median speedup sT20: {}%".format(median_speedup_sT20))
    print("median speedup sT50: {}%".format(median_speedup_sT50))
    print("mean speedup sT20: {}%".format(mean_speedup_sT20))
    print("mean speedup sT50: {}%".format(mean_speedup_sT50))

    plt.figure(figsize=(4,3))
    sizes = [len(a) for a in initial_atoms]
    plt.scatter(mmff_steps, sT20_steps, alpha=0.5, label="20 Steps", s=sizes, linewidth=0)
    plt.scatter(mmff_steps, sT50_steps, alpha=0.5, label="50 Steps", s=sizes, linewidth=0)
    arrow_kwargs = {"width": 0.3, "length_includes_head": True, "edgecolor": "k", "linewidth": 0.2}
    for i in range(mmff_steps.shape[0]):
        plt.arrow(mmff_steps[i], sT20_steps[i], 0, delta_es20[i]*20, facecolor="C0", **arrow_kwargs)
        plt.arrow(mmff_steps[i], sT50_steps[i], 0, delta_es50[i]*20, facecolor="C1", **arrow_kwargs)

    plt.plot([0, mmff_steps.max()], [0, mmff_steps.max()], color="k", label="y=x")

    scalebar = AnchoredSizeBar(plt.gca().transData,
                               0.5, '1kcal/mol', 'lower right',
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=20)
                               #fontproperties=fontprops)

    plt.gca().add_artist(scalebar)
    plt.xlabel("xTB Steps/MMFF")
    plt.ylabel("xTB Steps/Diffusion")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("writeup/figures/relaxation_speedup_scatter_geom250.pdf")
    #plt.show()
    exit()

    fig = plt.figure(figsize=(4,3))

    def mean_cos_similarity(v1s, v2s):
        # v1s and v2s are (chain_len x n_atoms x 3)
        # returns mean cos similarity, size (chain_len)
        dot = (v1s * v2s).sum(axis=2)
        norm1 = np.linalg.norm(v1s, axis=2)
        norm2 = np.linalg.norm(v2s, axis=2)

        cos = np.nan_to_num(dot / (norm1 * norm2))
        return cos.mean(axis=1)

    def get_cos_theta(results, k):
        n_chains = len(results["chain_xyzs"])
        all_cos_deltas_fs = []
        all_cos_deltas_direct = []
        all_cos_fs_direct = []
        #xs = []
        for i in range(n_chains):
            chain_len = len(results["chain_xyzs"][i])

            xyz1 = np.stack(results["chain_xyzs"][i][:-k])
            xyz2 = np.stack(results["chain_xyzs"][i][k:])
            gs_xyz = np.array(results["diffused_relaxed_xyzs"][i])

            chain_deltas = xyz2 - xyz1
            chain_fs = np.stack(results["chain_fs"][i][:-k])
            #chain_fs = np.stack([sum(results["chain_fs"][i][j:j+k]) for j in range(chain_len - k)])
            chain_direct = gs_xyz - xyz1

            #scale = (np.linalg.norm(chain_deltas, axis=2) / np.linalg.norm(chain_fs, axis=2)).mean()
            #print(np.abs(chain_fs).mean(), np.abs(chain_deltas / scale - chain_fs).mean())

            cos_deltas_fs = mean_cos_similarity(chain_deltas, chain_fs)
            cos_deltas_direct = mean_cos_similarity(chain_deltas, chain_direct)
            cos_fs_direct = mean_cos_similarity(chain_fs, chain_direct)

            all_cos_deltas_fs.append(cos_deltas_fs)
            all_cos_deltas_direct.append(cos_deltas_direct)
            all_cos_fs_direct.append(cos_fs_direct)

            #xs += list(range(chain_len - k))

        #xs = np.array(xs)
        all_cos_deltas_fs = np.stack(all_cos_deltas_fs)
        all_cos_deltas_direct = np.stack(all_cos_deltas_direct)
        all_cos_fs_direct = np.stack(all_cos_fs_direct)

        return all_cos_deltas_fs, all_cos_deltas_direct, all_cos_fs_direct

    for i, k in enumerate(reversed([1, 10, 30])):
    #for i, k in enumerate(reversed([1, 5, 9, 13, 17, 21, 25, 29])):
        deltas_fs, deltas_direct, fs_direct = get_cos_theta(results50, k)

        #plt.scatter(x, cos_direct, alpha=0.5, color=color)

        #x_line = np.arange(x.max() + 1)
        color = "C{}".format(i)

        #line = Polynomial(np.polyfit(x, deltas_fs, 1)[::-1])
        #plt.plot(x_line, line(x_line), color=color, label="k={}".format(k))

        #line = Polynomial(np.polyfit(x, deltas_direct, 1)[::-1])
        #plt.plot(x_line, line(x_line), color=color, linestyle="--")

        #line = Polynomial(np.polyfit(x, fs_direct, 1)[::-1])
        #plt.plot(x_line, line(x_line), color=color, linestyle=":")

        xs = np.arange(deltas_fs.shape[1])
        mean = deltas_fs.mean(axis=0)
        std = deltas_fs.std(axis=0)
        plt.plot(xs, mean, label="k={}".format(k), color=color)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

        mean = deltas_direct.mean(axis=0)
        std = deltas_direct.std(axis=0)
        plt.plot(xs, mean, color=color, linestyle="--")
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    mean = fs_direct.mean(axis=0)
    std = fs_direct.std(axis=0)
    plt.plot(xs, mean, color="k", linestyle=":", linewidth=3)
    #plt.fill_between(xs, mean - std, mean + std, alpha=0.2, color="k")

    plt.plot([2,2], [-2,-2], color="k", label="$\\cos\\theta_{\\Delta_k,f}$")
    plt.plot([2,2], [-2,-2], color="k", linestyle="--", label="$\\cos\\theta_{\\Delta_k,gs}$")
    plt.plot([2,2], [-2,-2], color="k", linestyle=":", label="$\\cos\\theta_{f,gs}$")
    plt.grid(True, axis="y")
    plt.xlabel("Diffusion Step")
    plt.ylabel("$\\cos\\theta$")
    plt.ylim(-0.1,1.15)
    plt.xlim(-1,59)
    fig.legend(loc="upper center", bbox_to_anchor=(0.855, 1.02))
    plt.title("{} Training Epochs".format(epoch), loc="left")
    plt.tight_layout()
    #plt.savefig("writeup/figures/force_alignment_qm95150.pdf")
    plt.show()

    #return mean_speedup_sT20, mean_speedup_sT50, median_speedup_sT20, median_speedup_sT50
    return mmff_steps, sT50_steps, delta_es50

if __name__ == "__main__":
    #main(100, "qm9")
    main(250, "geom")
