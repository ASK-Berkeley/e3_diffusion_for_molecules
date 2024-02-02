from rdkit import Chem
from rdkit.Chem import AllChem
import os
import io
import sys
import time
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace
from qm9.models import get_model
from qm9.sampling import sample
from qm9.data.args import init_argparse
from qm9.data.utils import initialize_datasets
from qm9.data.collate import PreprocessQM9
from qm9.rdkit_functions import mol2smiles, build_molecule
from qm9 import dataset
from configs.datasets_config import get_dataset_info
import ase
from ase.visualize import view
from ase.optimize import BFGS
from ase import Atoms
import psi4
from psi4_chain import get_ef, relax
from qcelemental.exceptions import ValidationError
from openbabel import openbabel as ob
import matgl
from matgl.ext.ase import M3GNetCalculator
import random
from multiprocessing import Pool

NUM_THREADS = 24

#query = torch.tensor([[ 0.1836,  1.4657,  0.2675],
#        [-0.1137,  0.0701, -0.0703],
#        [ 0.5879, -0.4521, -1.1837],
#        [ 0.2378, -0.2818, -2.4923],
#        [-0.8980,  0.4187, -3.1417],
#        [ 1.1484, -0.9052, -3.2955],
#        [ 2.0695, -1.4938, -2.4831],
#        [ 1.7937, -1.2524, -1.1760],
#        [ 2.5528, -1.7201,  0.0257],
#        [ 1.2485,  1.6630,  0.4714],
#        [-0.1217,  2.1137, -0.5589],
#        [-0.3975,  1.7514,  1.1497],
#        [ 0.0345, -0.5163,  0.7423],
#        [-0.5980,  1.3621, -3.6154],
#        [-1.3536, -0.2057, -3.9177],
#        [-1.6542,  0.6415, -2.3845],
#        [ 2.8507, -2.0453, -2.9810],
#        [ 2.8760, -0.8818,  0.6544],
#        [ 1.9490, -2.3817,  0.6596],
#        [ 3.4466, -2.2782, -0.2663]])

def do_ml_relaxations(model, args, xyz, symbols, charges, dataset_info, start_T):
    num_atoms = xyz.shape[0]
    atom_idx = torch.tensor([dataset_info["atom_encoder"][s] for s in symbols])
    one_hot = F.one_hot(atom_idx, len(dataset_info["atom_decoder"])).type(torch.float32)

    if dataset_info["name"] == "qm9":
        idx_to_num = [1, 6, 7, 8, 9]
    elif dataset_info["name"] == "geom":
        idx_to_num = dataset_info["atomic_nb"]
    atomic_nums = torch.tensor(idx_to_num)[atom_idx]

    # reject structures with nonzero charge
    if args.include_charges:
        assert (charges.view(-1) - atomic_nums).abs().sum() == 0

    # add padding & put in one-element batch
    max_n_nodes = dataset_info["max_n_nodes"]
    xyz_padded = F.pad(xyz, (0, 0, 0, max_n_nodes - num_atoms))[None,:,:].type(torch.float32)
    one_hot_padded = F.pad(one_hot, (0, 0, 0, max_n_nodes - num_atoms))[None,:,:]
    charges_padded = F.pad(charges, (0, 0, 0, max_n_nodes - num_atoms))[None,:,:]

    # run diffused on the input positions
    fix_noise = {"x": xyz_padded.cuda(),
                 "h_categorical": one_hot_padded.cuda()}
    if args.include_charges:
         fix_noise["h_integer"] = charges_padded.type(torch.float32).cuda()

    diffused_one_hot, diffused_charges, diffused_xyz, diffused_node_mask, chain = \
        sample(args, args.device, model, dataset_info,
               nodesxsample=torch.tensor([num_atoms]), fix_noise=fix_noise, start_T=start_T)

    chain = torch.flip(chain, dims=[0])

    # returned as a batch with padding. Get first out of batch & remove padding
    mask = diffused_node_mask[:,:,0] == 1
    diffused_one_hot = diffused_one_hot[mask].cpu()
    diffused_charges = diffused_charges[mask].view(-1).cpu()
    diffused_xyz = diffused_xyz[mask].cpu()

    mask = mask.cpu().view(-1)
    chain = chain[:,mask,:]

    return diffused_xyz, chain

    # make sure diffused didn't add charge to any of the atoms or change atom type
    if (diffused_one_hot.argmax(dim=1) - atom_idx).abs().sum() != 0:
        print("WARNING: Diffusion changed atomic nums. Ignoring new nums.")
    if args.include_charges:
        if (diffused_charges - atomic_nums).abs().sum() != 0:
            print("WARNING: Diffusion changed charges. Ignoring new charges.")

    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    calc = M3GNetCalculator(pot)
    atoms = Atoms(symbols=symbols, positions=xyz)
    atoms.set_cell(np.eye(3) * 100)
    atoms.calc = calc
    traj_fn = "tmp{}.traj".format(np.random.randint(1000))
    opt = BFGS(atoms, trajectory="test3.traj")
    opt.run(fmax=0.03)
    m3g_xyz = atoms.positions

    m3g_traj = ase.io.Trajectory(traj_fn)
    os.remove(traj_fn)

    return diffused_xyz, chain, m3g_xyz, m3g_traj


def get_relax_results(atom_symbols, xyz, method="psi4", basis="6-31G_2df_p_", norelax=False):
    atoms = Atoms(symbols=atom_symbols, positions=xyz)
    initial_e, initial_f = get_ef(atoms, method=method, basis=basis, num_threads=NUM_THREADS)
    start_time = time.time()
    if norelax:
        return initial_e, initial_f, None, None, None, None, None

    success, relaxed, relax_num_steps = relax(atoms, method=method, basis=basis, num_threads=NUM_THREADS)
    if not success:
        return initial_e, initial_f, None, None, None, None, None
    relax_time = time.time() - start_time
    relaxed_xyz = relaxed.positions
    relaxed_e, relaxed_f = get_ef(relaxed, method=method, basis=basis, num_threads=NUM_THREADS)

    return initial_e, initial_f, relaxed_xyz, relaxed_e, relaxed_f, relax_time, relax_num_steps

def xyz_to_rdkit(symbols, xyz):
    atoms = Atoms(symbols=symbols, positions=xyz)
    bytesio = io.BytesIO(b"")
    txtwrapper = io.TextIOWrapper(bytesio, write_through=True, errors="replace")
    ase.io.write(txtwrapper, atoms, format="xyz")
    bytesio.seek(0)
    xyz_string = bytesio.read().decode("utf-8")
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")
    mol = ob.OBMol()
    obConversion.ReadString(mol, xyz_string)

    # Convert the Open Babel molecule to an RDKit molecule
    mol_block = obConversion.WriteString(mol)
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    return mol


def main(model_path, epoch, start_T, dataset_name="qm9"):
    torch.manual_seed(4)
    np.random.seed(1)
    random.seed(42)

    global_start_time = time.time()

    psi4.set_memory("32 GB")

    if dataset_name == "qm9":
        method = "psi4"
        basis = "6-31G_2df_p_"
    elif dataset_name == "geom":
        method = "xtb"
        basis = "6-31G"
        #basis = "cc-pVDZ"

    # Load the model from disk based on `model_path` and `epoch`
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
    dataloaders, charge_scale = retrieve_dataloaders(args)
    #first_datum = next(iter(dataloaders["valid"]))

    all_symbols = []
    initial_xyzs = []
    initial_es = []
    initial_fs = []
    initial_relaxed_xyzs = []
    initial_relaxed_es = []
    initial_relaxed_fs = []
    initial_relax_times = []
    initial_nsteps = []
    guess_xyzs = []
    guess_es = []
    guess_fs = []
    guess_relaxed_xyzs = []
    guess_relaxed_es = []
    guess_relaxed_fs = []
    guess_relax_times = []
    guess_nsteps = []
    mmff_xyzs = []
    mmff_es = []
    mmff_fs = []
    mmff_relaxed_xyzs = []
    mmff_relaxed_es = []
    mmff_relaxed_fs = []
    mmff_relax_times = []
    mmff_nsteps = []
    diffused_xyzs = []
    diffused_es = []
    diffused_fs = []
    diffused_relaxed_xyzs = []
    diffused_relaxed_es = []
    diffused_relaxed_fs = []
    diffused_relax_times = []
    diffused_nsteps = []

    chain_xyzs = []
    chain_es = []
    chain_fs = []

    # Calculate diffused trajectories for first N validation-set data points
    if "valid" in dataloaders:
        val_dataloader = dataloaders["valid"]
    elif "val" in dataloaders:
        val_dataloader = dataloaders["val"]
    else:
        raise RuntimeError("No validation set dataloader available")

    #breakpoint()
    for data_idx, datum in enumerate(val_dataloader):
        if data_idx > (500 if dataset_name == "geom" else 50):
            break

        """
        cur = datum["positions"][0]
        if cur.shape[0] != query.shape[0]:
            continue
        delta = (cur - query).norm(dim=1).mean()
        if delta < 0.01:
            print(delta)
            print(datum["index"])
            print([atom_decoder[a] for a in datum["one_hot"][0].float().argmax(dim=1)])

        continue
        """

        print("[{}] Starting data_idx {}".format(time.time() - global_start_time, data_idx))

        # structures we care about for each input:
        # initial
        # initial-DFT
        # guess
        # guess-DFT
        # MMFF
        # MMFF-DFT
        # MMFF-diffused
        # MMFF-diffused-DFT

        # Re-relax the molecule at our level of theory

        initial_xyz = datum["positions"][0] # (datum is actually a batch of size 1)
        one_hot = datum["one_hot"][0]
        atom_types = one_hot.float().argmax(dim=1)

        atom_symbols = [atom_decoder[a] for a in atom_types]
        all_symbols.append(atom_symbols)

        initial_e, initial_f, initial_relaxed_xyz, initial_relaxed_e, initial_relaxed_f, initial_relax_time, initial_nstep = \
                get_relax_results(atom_symbols, initial_xyz, method=method, basis=basis, norelax=False)

        initial_xyzs.append(initial_xyz)
        initial_es.append(initial_e)
        initial_fs.append(initial_f)
        initial_relaxed_xyzs.append(initial_relaxed_xyz)
        initial_relaxed_es.append(initial_relaxed_e)
        initial_relaxed_fs.append(initial_relaxed_f)
        initial_relax_times.append(initial_relax_time)
        initial_nsteps.append(initial_nstep)

        # Find an approximate starting geometry using RDKit's heuristics

        # first, turn the XYZ coordinates into an rdkit mol with bonds

        # this only works for simple molecules like in QM9
        #mol = build_molecule(torch.tensor(initial_relaxed_xyz), atom_types, dataset_info)

        # this works for more complex molecules
        mol = xyz_to_rdkit(atom_symbols, initial_xyz)
        failed = False
        try:
            Chem.SanitizeMol(mol)
        except:
            print("WARNING: Chem.SanitizeMol errored")
            failed = True

        if not failed:
            success = AllChem.EmbedMolecule(mol, randomSeed=42)
            if success == -1:
                print("WARNING: Could not embed molecule")
                failed = True

        if failed:
            guess_xyzs.append(None), guess_es.append(None), guess_fs.append(None)
            guess_relaxed_xyzs.append(None), guess_relaxed_es.append(None)
            guess_relaxed_fs.append(None), guess_relax_times.append(None)
            guess_nsteps.append(None)

            mmff_xyzs.append(None), mmff_es.append(None), mmff_fs.append(None)
            mmff_relaxed_xyzs.append(None), mmff_relaxed_es.append(None)
            mmff_relaxed_fs.append(None), mmff_relax_times.append(None)
            mmff_nsteps.append(None)

            diffused_xyzs.append(None), diffused_es.append(None), diffused_fs.append(None)
            diffused_relaxed_xyzs.append(None), diffused_relaxed_es.append(None)
            diffused_relaxed_fs.append(None), diffused_relax_times.append(None)
            diffused_nsteps.append(None)

            continue

        guess_xyz = mol.GetConformer().GetPositions()
        guess_e, guess_f, guess_relaxed_xyz, guess_relaxed_e, guess_relaxed_f, guess_relax_time, guess_nstep = \
                get_relax_results(atom_symbols, guess_xyz, method=method, basis=basis, norelax=True)

        guess_xyzs.append(guess_xyz)
        guess_es.append(guess_e)
        guess_fs.append(guess_f)
        guess_relaxed_xyzs.append(guess_relaxed_xyz)
        guess_relaxed_es.append(guess_relaxed_e)
        guess_relaxed_fs.append(guess_relaxed_f)
        guess_relax_times.append(guess_relax_time)
        guess_nsteps.append(guess_nstep)

        # Optimize this starting geometry using the MMFF force field
        AllChem.MMFFOptimizeMolecule(mol)

        # get the DFT-computed energy of the MMFF-optimized structure
        mmff_xyz = mol.GetConformer().GetPositions()
        mmff_e, mmff_f, mmff_relaxed_xyz, mmff_relaxed_e, mmff_relaxed_f, mmff_relax_time, mmff_nstep = \
                get_relax_results(atom_symbols, mmff_xyz, method=method, basis=basis, norelax=epoch != "0")

        mmff_xyzs.append(mmff_xyz)
        mmff_es.append(mmff_e)
        mmff_fs.append(mmff_f)
        mmff_relaxed_xyzs.append(mmff_relaxed_xyz)
        mmff_relaxed_es.append(mmff_relaxed_e)
        mmff_relaxed_fs.append(mmff_relaxed_f)
        mmff_relax_times.append(mmff_relax_time)
        mmff_nsteps.append(mmff_nstep)

        # prepare the resulting structure to pass to the diffused model
        mmff_xyz = torch.tensor(mmff_xyz)
        #mmff_xyz = torch.tensor(guess_xyz)
        mmff_xyz -= mmff_xyz.mean(dim=0)
        num_atoms = mol.GetNumAtoms()
        charges = [mol.GetAtoms()[i].GetAtomicNum() for i in range(num_atoms)]
        charges = torch.tensor(charges).view(-1, 1)

        diffused_xyz, chain = do_ml_relaxations(model, args, mmff_xyz, atom_symbols,
                                                charges, dataset_info, start_T)

        diffused_e, diffused_f, diffused_relaxed_xyz, diffused_relaxed_e, diffused_relaxed_f, diffused_relax_time, diffused_nstep = \
                get_relax_results(atom_symbols, diffused_xyz, method=method, basis=basis)

        diffused_xyzs.append(diffused_xyz)
        diffused_es.append(diffused_e)
        diffused_fs.append(diffused_f)
        diffused_relaxed_xyzs.append(diffused_relaxed_xyz)
        diffused_relaxed_es.append(diffused_relaxed_e)
        diffused_relaxed_fs.append(diffused_relaxed_f)
        diffused_relax_times.append(diffused_relax_time)
        diffused_nsteps.append(diffused_nstep)

        if dataset_info["name"] == "qm9":
            idx_to_num = [1, 6, 7, 8, 9]
        elif dataset_info["name"] == "geom":
            idx_to_num = dataset_info["atomic_nb"]

        # compute DFT energies for each structure along the diffused path
        n_species = len(dataset_info["atom_decoder"])
        numbers = [[idx_to_num[a]
                    for a in chain[i,:,3:3+n_species].float().argmax(dim=1)]
                   for i in range(chain.shape[0])]
        numbers = torch.tensor(numbers).cpu()
        es = []
        fs = []
        xyzs = []
        print("Computing chain ef...")
        for i in range(0, numbers.shape[0]):
            print("{} / {}".format(i, numbers.shape[0] - 1), end="\r")
            xyz = chain[i,:,:3].cpu()
            e, f = get_ef(Atoms(numbers=numbers[i,:], positions=xyz),
                          method=method,
                          basis=basis,
                          num_threads=NUM_THREADS)
            xyzs.append(xyz)
            es.append(e - initial_relaxed_e)
            fs.append(f)
        print()
        chain_xyzs.append(xyzs)
        chain_es.append(es)
        chain_fs.append(fs)

        print("********************")
        print("i", initial_nsteps[-1], initial_relaxed_es[-1])
        print("g", guess_nsteps[-1], guess_relaxed_es[-1])
        print("m", mmff_nsteps[-1], mmff_relaxed_es[-1])
        print("d", diffused_nsteps[-1], diffused_relaxed_es[-1])
        print("********************")

    results = {
        "all_symbols": all_symbols,
        "initial_xyzs": initial_xyzs,
        "initial_es": initial_es,
        "initial_fs": initial_fs,
        "initial_relaxed_xyzs": initial_relaxed_xyzs,
        "initial_relaxed_es": initial_relaxed_es,
        "initial_relaxed_fs": initial_relaxed_fs,
        "initial_relax_times": initial_relax_times,
        "initial_nsteps": initial_nsteps,
        "guess_xyzs": guess_xyzs,
        "guess_es": guess_es,
        "guess_fs": guess_fs,
        "guess_relaxed_xyzs": guess_relaxed_xyzs,
        "guess_relaxed_es": guess_relaxed_es,
        "guess_relaxed_fs": guess_relaxed_fs,
        "guess_relax_times": guess_relax_times,
        "guess_nsteps": guess_nsteps,
        "mmff_xyzs": mmff_xyzs,
        "mmff_es": mmff_es,
        "mmff_fs": mmff_fs,
        "mmff_relaxed_xyzs": mmff_relaxed_xyzs,
        "mmff_relaxed_es": mmff_relaxed_es,
        "mmff_relaxed_fs": mmff_relaxed_fs,
        "mmff_relax_times": mmff_relax_times,
        "mmff_nsteps": mmff_nsteps,
        "diffused_xyzs": diffused_xyzs,
        "diffused_es": diffused_es,
        "diffused_fs": diffused_fs,
        "diffused_relaxed_xyzs": diffused_relaxed_xyzs,
        "diffused_relaxed_es": diffused_relaxed_es,
        "diffused_relaxed_fs": diffused_relaxed_fs,
        "diffused_relax_times": diffused_relax_times,
        "diffused_nsteps": diffused_nsteps,
        "chain_xyzs": chain_xyzs,
        "chain_es": chain_es,
        "chain_fs": chain_fs
    }
    with open("denoising_results/denoising_results_{}{}_sT{}.pkl".format(dataset_name, epoch, start_T), "wb") as f:
        pickle.dump(results, f)


# copied from qm9/dataset.py but with shuffle=True
def retrieve_dataloaders(cfg):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = True

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale



if __name__ == "__main__":
    dataset_name = sys.argv[1]
    start_T = int(sys.argv[2])
    if dataset_name == "qm9":
        model_path = "outputs/edm_qm9/"
        #epoch = 5150
        # epoch = 3250
        epoch = int(sys.argv[3])
        #epochs = [0, 250, 500, 750, 950, 1450, 2000, 2500, 3250, 5150]
        main(model_path, str(epoch), start_T, "qm9")
    elif dataset_name == "geom":
        model_path = "outputs/edm_geom_drugs_resume"
        epoch = "84"
        for epoch in range(190, 251, 10):
            main(model_path, str(epoch), start_T, dataset_name=dataset_name)
    print()
