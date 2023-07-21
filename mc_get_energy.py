import os
import time
import sys
import glob
from ase.io import read
import numpy as np
from psi4_chain import get_ef
import psi4

def get_gs_energy():
    psi4.set_memory("32 GB")
    start_T = sys.argv[1]
    mol_idx = sys.argv[2]
    chain_dir = "outputs/qm9_mc/flexible_mols/diffusion/T{}/{}".format(start_T, mol_idx)
    gs_fn = os.path.join(chain_dir, "gs.xyz")
    gs_atoms = read(gs_fn)

    start_time = time.time()
    e, f = get_ef(gs_atoms, num_threads=16)
    with open(os.path.join(chain_dir, "gs.energy"), "w") as f:
        f.write(str(e))

def get_chain_energy():
    psi4.set_memory("32 GB")
    xyz_fn = sys.argv[1]
    gs_energy = None
    with open(os.path.join(os.path.dirname(xyz_fn), "gs.energy"), "r") as f:
        gs_energy = float(next(f))

    atoms = read(xyz_fn)
    e, f = get_ef(atoms)
    print(e - gs_energy)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        get_gs_energy()
    elif len(sys.argv) == 2:
        get_chain_energy()
