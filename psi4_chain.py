import psi4
import glob
import sys
import copy
import numpy as np
import ase
from ase.io import xyz
from ase.optimize import BFGS
from ase import units
from ase.calculators.psi4 import Psi4
from ase.build import molecule
from psi4.driver.p4util.exceptions import OptimizationConvergenceError


def xyz_to_mol(xyz_fn):
    return ase.io.read(xyz_fn, format="xyz")

def get_psi4_calc(atoms, num_threads=1):
    # qm9 functional/basis
    #return Psi4(atoms=atoms, method="B3LYP", basis="6-31G",
    #            memory="16GB", num_threads=num_threads)
    return Psi4(atoms=atoms, method="B3LYP", basis="6-31G_2df_p_",
                memory="16GB", num_threads=num_threads)
    # faster/less accurate
    #return Psi4(atoms=atoms, method="pbe", basis="6-31g", memory="16GB", num_threads=8)

    # I think cc-pVDZ is more common than 6-31G(2df,p)
    #return Psi4(atoms=atoms, method="B3LYP", basis="cc-pVDZ", memory="16GB", num_threads=8)

def get_ef(atoms, num_threads=1):
    calc = get_psi4_calc(atoms, num_threads=num_threads)
    atoms.calc = calc
    return atoms.get_potential_energy(), atoms.get_forces()

def relax(atoms):
    atoms = ase.Atoms(atoms)
    calc = get_psi4_calc(atoms)
    atoms.calc = calc
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    return atoms

def chain_fn(chain_id, frame_id):
    fn_pattern = "outputs/edm_qm9/eval/chain_{}/chain_{:0>3d}.txt"
    return fn_pattern.format(chain_id, frame_id)

def chain_summary_fn(chain_id):
    return "outputs/edm_qm9/eval/chain_{}/chain_summary.npy".format(chain_id)

def process_chain(chain_id):
    mol = xyz_to_mol(chain_fn(chain_id, 999))

    final_atomic_numbers = mol.get_atomic_numbers()

    # compute ground state from the last frame in the chain
    molgs = relax(mol)
    egs, fgs = get_ef(molgs)

    avg_cos_similarities = []
    energy_suboptimalities = []

    all_positions = [molgs.get_positions()]
    all_forces = [fgs]

    frame_id = 998
    # we're looping backwards over frames, so keep track of the "next" mol
    # rather than the "previous" one
    next_mol = ase.Atoms(mol)
    while frame_id >= 0:
        mol = xyz_to_mol(chain_fn(chain_id, frame_id))
        # only look at frames that have the same atomic numbers as the final frame
        if not (mol.get_atomic_numbers() == final_atomic_numbers).all():
            break

        e, f = get_ef(mol)

        # compare energy to ground-state energy
        energy_suboptimalities.append((e - egs) / (units.kcal / units.mol))

        # compare forces on current frame to the displacement btwn current & next
        displacement = next_mol.get_positions() - mol.get_positions()
        f_norm = np.linalg.norm(f, axis=1)
        displacement_norm = np.linalg.norm(displacement, axis=1)
        f_dot_displacement = (f * displacement).sum(axis=1)
        cos_theta = f_dot_displacement / (f_norm * displacement_norm)
        avg_cos_similarities.append(cos_theta.mean())

        all_positions.append(mol.get_positions())
        all_forces.append(f)

        frame_id -= 1
        next_mol = ase.Atoms(mol)

    with open(chain_summary_fn(chain_id), "wb") as f:
        np.savez(f,
                 energy_suboptimalities=energy_suboptimalities,
                 avg_cos_similarities=avg_cos_similarities,
                 all_positions=np.array(all_positions),
                 all_forces=np.array(all_forces),
                 gs_positions=molgs.get_positions())
    return energy_suboptimalities, avg_cos_similarities

if __name__ == "__main__":
    epoch = int(sys.argv[1])

    psi4.set_memory("32 GB")
    #psi4.core.set_output_file("psi4_output.{}.txt".format(epoch))
    #psi4.set_options({'reference': 'uhf'})
    #psi4.set_num_threads(2)


    for chain_id in range(100):
        delta_e, cos_sim = process_chain(chain_id)
        print("chain", chain_id)
        print(delta_e)
        print(cos_sim)

    exit()
    fn99 = "outputs/edm_qm9/epoch_{}_0/chain/chain_099.txt".format(epoch)
    fn98 = "outputs/edm_qm9/epoch_{}_0/chain/chain_098.txt".format(epoch)

    mol99 = xyz_to_mol(fn99)
    mol98 = xyz_to_mol(fn98)

    e98, f98 = get_ef(mol98)
    e99, f99 = get_ef(mol99)

    molgs = relax(mol99)
    egs, fgs = get_ef(molgs)

    force_norm = np.linalg.norm(f99, axis=1)
    print(", ".join(map(str, [epoch, force_norm.min(), force_norm.mean(), force_norm.max(), (e99 - egs) / (kcal / mol)])))
    breakpoint()
