import glob as glob
import numpy as np
import io
import sys

from openbabel import openbabel as ob
from openbabel import pybel
import ase
import ase.io
from ase import Atoms
from matplotlib import pyplot as plt

from qm9.bond_analyze import get_bond_order

n_atoms = []
step_sizes = []
bond_lengths_deviation = []
all_atom_stable = []
all_bo_stable = []
all_valid_bo = []
all_ch_bond_lengths = []

fns = sorted(glob.glob("eval/qm9_5150/chain*.npz"))
atom_type_lookup = np.array([1, 6, 7, 8, 9])
symbol_lookup = np.array(["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F"])

def process_fn(fn):
    with np.load(fn) as data:
        zs = atom_type_lookup[np.argmax(data["one_hot"], axis=2)]
        charges = data["charges"]
        positions = data["x"]

        n_atoms.append(zs.shape[1])

        # step size
        step_sizes.append(
            np.linalg.norm(np.diff(positions, axis=0), axis=2).mean(axis=1)
        )

        # bond length
        obmols = []
        for frame in range(zs.shape[0]):
            atoms = Atoms(numbers=zs[frame], positions=positions[frame])
            with io.StringIO() as f:
                tmp = sys.stdout
                sys.stdout = f
                ase.io.write("-", atoms, format="xyz")
                sys.stdout = tmp
                ase_xyz = f.getvalue()
                obmol = pybel.readstring("xyz", ase_xyz)
                obmols.append(obmol)

        final_mol = obmols[-1]
        final_bonds = np.array(
            [[bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1]
             for bond in ob.OBMolBondIter(final_mol.OBMol)]
        )
        cur_bond_lengths = np.linalg.norm(
                np.diff(positions[:,final_bonds], axis=2).squeeze(),
                axis=2
        )
        bond_lengths_deviation.append(cur_bond_lengths - cur_bond_lengths[-1,:])

        # C-H bond lengths
        dist_matrix = np.linalg.norm(
            positions[:,None,:,:] - positions[:,:,None,:], axis=3
        )

        cur_ch = []
        for mol, cur_zs, D in zip(obmols, zs, dist_matrix):
            bonds = np.asarray(
                [[bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1]
                 for bond in ob.OBMolBondIter(mol.OBMol)]
            )
            ch_mask = (
                ((cur_zs[bonds[:,0]] == 1) | (cur_zs[bonds[:,0]] == 6)) &
                ((cur_zs[bonds[:,1]] == 1) | (cur_zs[bonds[:,1]] == 6)) &
                (cur_zs[bonds[:,0]] != cur_zs[bonds[:,1]])
            )
            ch_bonds = bonds[ch_mask]
            cur_ch.append(D[ch_bonds[:,0], ch_bonds[:,1]])
        all_ch_bond_lengths.append(cur_ch)

        # atom species finalized
        not_final_rev = np.maximum.accumulate(~(zs == zs[-1,:])[::-1,:], axis=0)
        atom_finalized = (1 - not_final_rev)[::-1,:]
        all_atom_stable.append(atom_finalized)
        # bond order finalized
        bond_orders = []
        #for mol in obmols:
        #    bond_orders.append(
        #        np.bincount(
        #            np.asarray(
        #                [[bond.GetBeginAtomIdx() - 1,
        #                  bond.GetEndAtomIdx() - 1,
        #                  bond.GetBondOrder()]
        #                 for bond in ob.OBMolBondIter(mol.OBMol)],
        #            ).reshape(-1),
        #            minlength=zs.shape[1]
        #        )
        #    )
        for frame in range(zs.shape[0]):
            symbols = symbol_lookup[zs[frame]]
            bonds = [(
                i,
                j,
                get_bond_order(symbols[i], symbols[j], dist_matrix[frame,i,j])
            ) for i in range(zs.shape[1]) for j in range(i + 1, zs.shape[1])]
            bonds = np.array([b for b in bonds if b[2] > 0])
            bond_orders.append(
                np.bincount(
                    bonds[:,:2].reshape(-1),
                    weights=np.repeat(bonds[:,2], 2),
                    minlength=zs.shape[1]
                )
            )
        bond_orders = np.asarray(bond_orders)
        not_final_rev = np.maximum.accumulate(
            ~(bond_orders == bond_orders[-1,:])[::-1,:],
            axis=0
        )
        bo_finalized = (1 - not_final_rev)[::-1,:]
        all_bo_stable.append(bo_finalized)

        # valid bond order
        # atomic numbers Z: [1, 6, 7, 8, 9]
        # valid BO:         [1, 4, 3, 2, 1])
        # BO = (8 - (Z - 2)) % 8
        valid_bo = (8 - (zs - 2)) % 8
        all_valid_bo.append((bond_orders == valid_bo))


for fn in fns[:9]:
    print(fn)
    process_fn(fn)

n_atoms = np.asarray(n_atoms)
step_sizes = np.asarray(step_sizes).T
bond_lengths_deviation = np.concatenate(bond_lengths_deviation, axis=1)
atom_stable = np.concatenate(all_atom_stable, axis=1)
mol_stable = np.stack([s.min(axis=1) for s in all_atom_stable]).T
atom_bo_stable = np.concatenate(all_bo_stable, axis=1)
mol_bo_stable = np.stack([s.min(axis=1) for s in all_bo_stable]).T
valid_bo = np.concatenate(all_valid_bo, axis=1)
ch_bond_lengths_mean = np.asarray(
    [np.hstack([mol[i] for mol in all_ch_bond_lengths]).mean()
     for i in range(mol_stable.shape[1])]
)
ch_bond_lengths_std = np.asarray(
    [np.hstack([mol[i] for mol in all_ch_bond_lengths]).std()
     for i in range(mol_stable.shape[1])]
)

#print("-" * 20)
#print("all")
#print("-" * 20)
#print("\n\n===================\n\n".join([str(mol[-10:]) for mol in all_ch_bond_lengths]))
#print("-" * 20)
#print("mean")
#print("-" * 20)
#print(ch_bond_lengths_mean[-10:])
#print("-" * 20)
#print("std")
#print("-" * 20)
#print(ch_bond_lengths_std[-10:])

fig, ax1 = plt.subplots(figsize=(4.2, 3.5))
ax1.invert_xaxis()
ax2 = ax1.twinx()
ax1.set_xlabel("Diffusion Step")
ax1.set_ylabel("Angstroms")
ax2.set_ylabel("Percent")
percent_min = -2
percent_max = 102
angstrom_max = 1.75
angstrom_min = angstrom_max * percent_min / percent_max
ax1.set_ylim(angstrom_min, angstrom_max)
ax2.set_ylim(percent_min, percent_max)

ss_x = range(1, 1000)[::-1]
x = range(1000)[::-1]
graphs = [
    (ss_x, "Step Size", step_sizes, ax1),
    (x, "|Bond Len. Err.|", np.abs(bond_lengths_deviation), ax1),
    (x, "Atom Elem. Final", 100*atom_stable, ax2),
    (x, "Mol. Elem. Final", 100*mol_stable, ax2),
    (x, "Atom BO Final", 100*atom_bo_stable, ax2),
    (x, "Mol. BO Final", 100*mol_bo_stable, ax2),
    (x, "Valid BO", 100*valid_bo, ax2)
]

for i, (x, label, vals, ax) in enumerate(graphs):
    mean = vals.mean(axis=1)
    std = vals.std(axis=1)
    ax.plot(x, mean, label=label, color=f"C{j}")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=f"C{j}")

#x = range(1, 1000)
#ss_mean = step_sizes.mean(axis=1)
#ss_std = step_sizes.std(axis=1)
#plt.plot(x, ss_mean, label="Step Size", color="C0")
#plt.fill_between(x, ss_mean - ss_std, ss_mean + ss_std, alpha=0.2, color="C0")
#
#x = range(1000)
#bl_mean = bond_lengths_deviation.abs().mean(axis=1)
#bl_std = bond_lengths_deviation.abs().std(axis=1)
#plt.plot(x, bl_mean, label="Bond Len. RMSD", color="C1")
#plt.fill_between(x, bl_mean - bl_std, bl_mean + bl_std, alpha=0.2, color="C1")
#
#as_mean = atom_stable.mean(axis=1)
#as_std = atom_stable.std(axis=1)
#plt.plot(x, as_mean, label="Atom Elem. Final", color="C2")
#plt.fill_between(x, as_mean - as_std, as_mean + as_std, alpha=0.2, color="C2")
#
#plt.plot(x, mol_stable.mean(axis=1), label="Mol Elem. Final")
#plt.plot(x, atom_bo_stable.mean(axis=1), label="Atom BO Final")
#plt.plot(x, mol_bo_stable.mean(axis=1), label="Mol BO Final")
#plt.plot(x, valid_bo.mean(axis=1), label="Atom BO Valid")
#plt.plot(x, ch_bond_lengths_mean, label="CH Bond Len.")
#plt.fill_between(x, ch_bond_lengths_mean - ch_bond_lengths_std, ch_bond_lengths_mean + ch_bond_lengths_std, alpha=0.2)

zoom = False
if zoom:
    angstrom_max = 0.25
    angstrom_min = angstrom_max * percent_min / percent_max
    ax1.set_ylim(angstrom_min, angstrom_max)

    plt.xlim(75, 0)
    fn = "writeup/figures/new_fig1_zoom.pdf"
else:
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.97))
    fn = "writeup/figures/new_fig1.pdf"

plt.tight_layout()
#plt.savefig(fn)
plt.show()
