from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pickle
import torch
import numpy as np
from argparse import Namespace
from qm9.models import get_model
from qm9.sampling import sample
from qm9.rdkit_functions import mol2smiles, build_molecule
from qm9 import dataset
from configs.datasets_config import get_dataset_info
from ase.visualize import view
from ase import Atoms
import psi4
from psi4_chain import get_ef, relax
import time

def main(epoch):
    torch.manual_seed(4)
    np.random.seed(1)

    psi4.set_memory("32 GB")

    model_path = "outputs/"
    args_fn = os.path.join(model_path, "args_{}.pickle").format(epoch)

    with open(args_fn, "rb") as f:
        args = pickle.load(f)

    if not hasattr(args, "normalization_factor"):
        args.normalization_factor = 1
    if not hasattr(args, "aggregation_method"):
        args.aggregation_method = "sum"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    model, _, _ = get_model(args, args.device, dataset_info, None)
    model.to(args.device)

    model_fn = "generative_model_ema_{}.npy" if args.ema_decay > 0 else "generative_model_{}.npy"
    model_fn = os.path.join(model_path, model_fn.format(epoch))
    state_dict = torch.load(model_fn, map_location=args.device)
    model.load_state_dict(state_dict)

    atom_encoder = dataset_info["atom_encoder"]
    atom_decoder = dataset_info["atom_decoder"]

    args.batch_size = 1
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    #first_datum = next(iter(dataloaders["valid"]))
    
    for data_idx, datum in enumerate(dataloaders["valid"]):
        if data_idx > 10:
            break
        # [0] to get first datum in "batch" of size 1
        xyz = datum["positions"][0]
        one_hot = datum["one_hot"][0]
        atom_types = one_hot.float().argmax(dim=1)

        symbol2num = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
        gs = Atoms(numbers=[symbol2num[dataset_info["atom_decoder"][a]]
                            for a in atom_types],
                   positions=xyz)
        gs_e, gs_f = get_ef(gs)

        mol = build_molecule(xyz, atom_types, dataset_info)
        Chem.SanitizeMol(mol)
        success = AllChem.EmbedMolecule(mol)
        if success == -1:
            continue
        AllChem.MMFFOptimizeMolecule(mol)

        mmff_pos = torch.tensor(mol.GetConformer().GetPositions())
        mmff_pos -= mmff_pos.mean(dim=0)
        num_atoms = mol.GetNumAtoms()
        charges = [mol.GetAtoms()[i].GetAtomicNum() for i in range(num_atoms)]
        charges = torch.tensor(charges).view(-1, 1)


        max_n_nodes = dataset_info["max_n_nodes"]
        mmff_pos_padded = torch.zeros((1, max_n_nodes, mmff_pos.shape[1]), dtype=torch.float32)
        mmff_pos_padded[:,:num_atoms,:] = mmff_pos
        one_hot_padded = torch.zeros((1, max_n_nodes, one_hot.shape[1]), dtype=torch.float32)
        one_hot_padded[:,:num_atoms,:] = one_hot
        charges_padded = torch.zeros((1, max_n_nodes, charges.shape[1]), dtype=torch.float32)
        charges_padded[:,:num_atoms,:] = charges

        fix_noise = {"x": mmff_pos_padded.cuda(),
                     "h_categorical": one_hot_padded.cuda(),
                     "h_integer": charges_padded.cuda()}

        diffused_one_hot, diffused_charges, diffused_x, diffused_node_mask, chain = \
            sample(args, args.device, model, dataset_info,
                   nodesxsample=torch.tensor([num_atoms]), fix_noise=fix_noise, start_T=50)

        #chain = torch.flip(chain, dims=[0])

        # returned as a batch with padding. Get first out of batch & remove padding
        diffused_one_hot = diffused_one_hot[diffused_node_mask[:,:,0] == 1].cpu()
        diffused_charges = diffused_charges[diffused_node_mask[:,:,0] == 1].view(-1).cpu()
        diffused_x = diffused_x[diffused_node_mask[:,:,0] == 1].cpu()

     
        atom_types = [dataset_info["atom_decoder"][a] for a in diffused_one_hot.argmax(dim=1)]
        atomic_nums = [{"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}[a] for a in atom_types]
        assert (diffused_charges - torch.tensor(atomic_nums)).abs().sum() == 0


        mask = diffused_node_mask[:,:,0].cpu().view(-1) == 1
        numbers = [{"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}[dataset_info["atom_decoder"][a]]
                   for a in chain[0,:,3:8].float().argmax(dim=1)]
        numbers = torch.tensor(numbers)
        numbers = numbers[mask]

        mmff_atoms = Atoms(numbers=numbers, positions=mmff_pos)
        diffused_atoms = Atoms(numbers=numbers, positions=diffused_x)

        start_time = time.time()
        relax(diffused_atoms)
        print("time for diffused:", time.time() - start_time)

        start_time = time.time()
        relax(mmff_atoms)
        print("time for MMFF:", time.time() - start_time)


if __name__ == "__main__":
    main(3250)
