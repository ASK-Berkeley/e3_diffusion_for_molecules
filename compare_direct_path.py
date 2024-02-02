import numpy as np
import ase
import os
from configs.datasets_config import get_dataset_info

from psi4_chain import get_ef

in_dir = "eval/qm9_5150"

dataset_info = get_dataset_info("qm9", False)

n_even = 0
n_odd = 0
for i in range(100):
    chain_fn = os.path.join(in_dir, "chain_{:0>3d}.npz".format(i))
    chain = np.load(chain_fn)
    atom_idx = chain["one_hot"].argmax(axis=2)

    idx_to_num = np.array([1, 6, 7, 8, 9])

    atomic_nums = idx_to_num[atom_idx[-1]]
    charges = chain["charges"][-1].reshape(-1)
    #print("ASDASD" if np.abs(charges - atomic_nums).sum() > 0 else "")
    if sum(atomic_nums) % 2 == 0:
        n_even += 1
    else:
        n_odd += 1

    # find where atom species are finalized
    final_symbols = [dataset_info["atom_decoder"][a] for a in atom_idx[-1]]
    for image_idx in reversed(range(chain["x"].shape[0])):
        symbols = [dataset_info["atom_decoder"][a] for a in atom_idx[image_idx]]
        if any([s != f for s, f in zip(symbols, final_symbols)]):
            last_stable_image = image_idx + 1
            break

    es = []
    for image_idx in range(last_stable_image, chain["x"].shape[0]):
        atoms = ase.Atoms(symbols=final_symbols, positions=chain["x"][image_idx])
        e, f = get_ef(atoms, num_threads=28)
        es.append(e)

    # now compute the direct path
    direct_es = []
    n_direct_images = (chain["x"].shape[0] - last_stable_image) // 2
    direct_xs = np.linspace(chain["x"][last_stable_image], chain["x"][-1],
                            num=n_direct_images)
    for x in direct_xs:
        atoms = ase.Atoms(symbols=final_symbols, positions=x)
        e, f = get_ef(atoms, num_threads=28)
        direct_es.append(e)

    print(es)
    print(direct_es)
    print(es[0], np.max(es), es[-1])
    print(direct_es[0], np.max(direct_es), direct_es[-1])

print("even:", n_even)
print("odd:", n_odd)
