import copy
from numpy import average, load, argmax
import numpy as np
from qm9.bond_analyze import get_bond_order
from qm9.analyze import check_stability
from matplotlib import pyplot as plt

import ase
import ase.io

from openbabel import openbabel as ob

import matplotlib
matplotlib.use("qtagg")

atom_key = 'HCNOF'

DO_ATOM_ID_CALCS = True
DO_BOND_ORDER_CALCS = True
DO_BOND_STABILITY_CALCS = True
DO_BOND_DISTANCE_CALCS = False  # keep this one false, it's not helpful
DO_BOND_DISTANCE_RMSD = True
DO_STEP_SIZES = True
DEBUGGING = False

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}

def get_atom_type(one_hot_one_atom):
    max_index = argmax(one_hot_one_atom)
    return atom_key[max_index]

def make_atom_types_string(one_hot_arr):
    result = ''
    for atom in one_hot_arr:
        result += get_atom_type(atom)
    return result

def find_atom_finalized_iters(one_hot):
    num_atoms = len(one_hot[0])
    atom_finalized_iter = np.zeros(num_atoms)
    atom_final_identities = [argmax(atom) for atom in one_hot[-1]]
    for i in range(len(one_hot)):
        one_hot_iter = one_hot[i]
        for j in range(len(one_hot_iter)):
            if argmax(one_hot_iter[j]) != atom_final_identities[j]:
                atom_finalized_iter[j] = i
    return atom_finalized_iter

def find_atom_finalized_iters_filter_H(one_hot):
    num_atoms = len(one_hot[0])
    atom_finalized_iter = np.zeros(num_atoms)
    atom_final_identities = [argmax(atom) for atom in one_hot[-1]]
    for i in range(len(one_hot)):
        one_hot_iter = one_hot[i]
        for j in range(len(one_hot_iter)):
            if argmax(one_hot_iter[j]) != atom_final_identities[j]:
                atom_finalized_iter[j] = i
    filtered_atom_finalized_iter = []
    for i in range(len(atom_finalized_iter)):
        if atom_final_identities[i] == 0:
            filtered_atom_finalized_iter.append(atom_finalized_iter[i])
    return filtered_atom_finalized_iter

def calc_all_bond_orders(one_hot, positions):
    all_bond_orders = []    # num_iters by num_atoms. e.g. 1000 x 19
    num_iters = len(one_hot)
    for i in range(num_iters):
        all_bond_orders.append(calc_one_iter_bond_orders_atomwise(one_hot[i], positions[i]))
    return all_bond_orders

def calc_one_iter_bond_orders_atomwise(one_hot, positions):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    nr_bonds = np.zeros(len(x), dtype='int')
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1 = get_atom_type(one_hot[i])
            atom2 = get_atom_type(one_hot[j])
            order = get_bond_order(atom1, atom2, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    return nr_bonds

def calc_one_iter_bond_orders_pairwise(one_hot, positions):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    num_atoms = len(x)
    nr_bonds = np.zeros(int(num_atoms * (num_atoms - 1) / 2), dtype='int')
    index = 0
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1 = get_atom_type(one_hot[i])
            atom2 = get_atom_type(one_hot[j])
            order = get_bond_order(atom1, atom2, dist)
            nr_bonds[index] = order
            index += 1
    return nr_bonds

# DON'T use this
def calc_all_bond_orders_from_final_identities(final_one_hot, positions):
    # DON'T USE THIS I don't think it's correct and also it is super slow
    all_bond_orders_from_final = []
    # use the identities of the FINAL iteration to calc this
    all_bond_orders = []    # num_iters by num_atoms. e.g. 1000 x 19
    num_iters = len(positions)
    for i in range(num_iters):
        all_bond_orders.append(calc_one_iter_bond_orders_atomwise(final_one_hot, positions[i]))
    return all_bond_orders


def find_atom_finalized_bond_count_iters(one_hot, positions):
    num_iters = len(one_hot)
    num_atoms = len(one_hot[0])
    atom_finalized_iter = np.zeros(num_atoms)                                                   # each atom's iter where it is finalized
    # all_bond_orders_from_final = calc_all_bond_orders_from_final_identities(one_hot[-1], x)     # bond orders based on the identities of the final iteration
    # atom_final_bond_count = all_bond_orders_from_final[-1]                                      # the bond count of the final iteration
    atom_final_bond_count = calc_one_iter_bond_orders_atomwise(one_hot[-1], positions[-1])
    atom_prev_bond_order = copy.deepcopy(atom_final_bond_count)
    for i in reversed(range(num_iters)):
        if 0 not in atom_finalized_iter:
            return atom_finalized_iter
        cur_bond_orders = calc_one_iter_bond_orders_atomwise(one_hot[i], positions[i])
        for j in range(num_atoms):
            if atom_finalized_iter[j] == 0:
                if cur_bond_orders[j] != atom_prev_bond_order[j]:
                    atom_finalized_iter[j] = i + 1
                atom_prev_bond_order[j] = cur_bond_orders[j]
    return atom_finalized_iter

# Params: ONE iter of position data
def calc_one_iter_bond_distances(positions):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    num_atoms = len(x)
    num_pairs = int(num_atoms * (num_atoms - 1) / 2)
    bond_dists = np.zeros(num_pairs, dtype='int')
    counter = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            bond_dists[counter] = dist
            counter += 1
    return bond_dists

def find_pairwise_finalized_bond_dist_iters(one_hot, positions):
    cutoff = 0.05                   # 5% deviation from final bond distance allowed
    num_iters = len(one_hot)
    num_atoms = len(one_hot[0])
    num_bonds = int(num_atoms * (num_atoms - 1) / 2)
    bond_dist_finalized_iter = np.zeros(num_bonds)                                  # each bond's iter where it is finalized
    atom_final_bond_dist = calc_one_iter_bond_distances(positions[-1])
    atom_prev_bond_dist = copy.deepcopy(atom_final_bond_dist)
    for i in reversed(range(num_iters)):
        if 0 not in bond_dist_finalized_iter:
            return bond_dist_finalized_iter
        cur_bond_dists = calc_one_iter_bond_distances(positions[i])
        for j in range(num_bonds):
            if bond_dist_finalized_iter[j] == 0:
                # check if these two are outside of the error range
                if cur_bond_dists[j] < atom_prev_bond_dist[j] * (1 - cutoff) or cur_bond_dists[j] > atom_prev_bond_dist[j] * (1 + cutoff) :
                    bond_dist_finalized_iter[j] = i + 1
                atom_prev_bond_dist[j] = cur_bond_dists[j]
    return bond_dist_finalized_iter

def calc_one_iter_rmsd_bond_distances(positions, reference_bond_lengths, bond_filter, num_filtered_bonds):
    num_atoms = len(positions)
    num_bonds = len(reference_bond_lengths)
    cur_bond_distances = calc_one_iter_bond_distances(positions)
    cumulative_differences_squared = 0.0
    for i in range(num_bonds):
        if bond_filter[i]:
            cumulative_differences_squared += (cur_bond_distances[i] - reference_bond_lengths[i]) ** 2
    return np.sqrt(cumulative_differences_squared / num_filtered_bonds)

# returns an array that is [1 x num_iters]
def get_all_iters_bond_distance_rmsd(one_hot, positions):
    num_iters = len(one_hot)
    num_atoms = len(one_hot[0])
    num_bonds = int(num_atoms * (num_atoms - 1) / 2)
    reference_bond_lengths = calc_one_iter_bond_distances(positions[-1])
    final_bond_order = calc_one_iter_bond_orders_pairwise(one_hot[-1], positions[-1])
    bond_filter = []
    num_filtered_bonds = 0
    for b in final_bond_order:
        if b > 0:
            bond_filter.append(True)
            num_filtered_bonds += 1
        else:
            bond_filter.append(False)
    rmsd_bond_length_all_iters = []
    for i in range(num_iters):
        rmsd_bond_length_all_iters.append(calc_one_iter_rmsd_bond_distances(positions[i], reference_bond_lengths, bond_filter, num_filtered_bonds))
    return rmsd_bond_length_all_iters

def get_all_iters_step_size_rmsd(one_hot, positions):
    # loop through each iteration
    #   loop through each atom
    #       add up the rmsd step size from the previous iteration
    #   average it
    num_iters = len(positions)
    num_atoms = len(positions[0])
    step_size_rmsd = []
    for i in range(num_iters):
        cur_positions = positions[i]
        if i == 0:
            step_size_rmsd.append(0.0)
        else:
            sum_step_size = 0.0
            for j in range(num_atoms):
                sum_step_size += np.sqrt((cur_positions[j][0] - prev_positions[j][0]) ** 2 \
                                        + (cur_positions[j][1] - prev_positions[j][1]) ** 2 \
                                        + (cur_positions[j][2] - prev_positions[j][2]) ** 2)
            step_size_rmsd.append(sum_step_size / num_atoms)
        prev_positions = cur_positions
    return step_size_rmsd

def calc_one_iter_atom_stability(one_hot, positions):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    num_atoms = len(x)

    nr_bonds = np.zeros(num_atoms, dtype='int')

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = get_atom_type(one_hot[i]), get_atom_type(one_hot[j])
            order = get_bond_order(atom1, atom2, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for one_hot_i, nr_bonds_i in zip(one_hot, nr_bonds):
        possible_bonds = allowed_bonds[get_atom_type(one_hot_i)]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        nr_stable_bonds += int(is_stable)

    return 100 * nr_stable_bonds / num_atoms     # integer representing percentage of atoms with a stable number of bonds in this iteration

def calc_all_iters_atom_stability(one_hot, positions):
    num_iters = len(positions)
    atom_stability = []
    for i in range(num_iters):
        atom_stability.append(calc_one_iter_atom_stability(one_hot[i], positions[i]))   # array of shape num_iters
    return atom_stability

# Just study one chain for testing
with load('eval/qm9_5150/chain_000.npz') as data:
    # 1000 iters, 19 atoms
    # Use bond_order function from e3_diffusion_for_molecules/qm9/bond_analyze.py
    one_hot_000 = data['one_hot']      # num_iters by num_atoms by num_possible_identities = 1000 x 19 x 5
    charges_000 = data['charges']
    x_000 = data['x']

    bond_orders_finalized_iters = find_atom_finalized_bond_count_iters(one_hot_000, x_000)
    print(f"000 finalized: {bond_orders_finalized_iters}")


    if (False):
        # TODO: try averaging for each atom type to see if different types are doing different things
        average_charges = [average(elem) for elem in charges_000]
        # average_charges = [[average(elem_per_atom) for elem_per_atom in elem] for elem in charges_000]
        plt.plot(average_charges)
        plt.show()

    # Atom Stability
    # Generate list of numbers which indicate the number iteration at which the atom no longer changes identity
    # set up list of atoms, default value 0 corresponding to iteration 0
    # set up list of atom identities corresponding to their final identity
    # loop through iterations from end to beginning
    #   loop through atoms
    #       if atom_finalized_iter is nonzero:
    #           if current atom identity is different from final identity
    #               set atom_finalized_iter
    print(find_atom_finalized_iters(one_hot_000))

# all chains
all_simulation_files = [f'eval/qm9_5150/chain_{i:03}.npz' for i in range(100)]
# atom ID
all_atom_finalized_iters = []
all_mol_finalized_iters = []
all_atom_id_means = []
all_atom_id_stdev = []
all_cumulative_atom_id_stability = []
all_cumulative_mol_id_stability = []
# bond order
all_atom_bond_order_finalized_iters = []
all_mol_bond_order_finalized_iters = []
all_atom_bo_means = []
all_atom_bo_stdev = []
all_cumulative_atom_bo_stability = []
all_cumulative_mol_bo_stability = []
# bond stability
all_iters_bond_stability = []
# bond distance finalization
all_atom_bond_distance_finalized_iters = []
all_mol_bond_distance_finalized_iters = []
all_atom_bd_means = []
all_atom_bd_stdev = []
# bond rmsds
all_files_all_iters_bond_distance_rmsds = []    # will become [num_files x num_iters]
# step size rmsds
all_files_all_iters_step_size_rmsds = []
counter = 0
for simulation_file in all_simulation_files:
    with load(simulation_file) as data:
        if(DEBUGGING):
            counter += 1
            if(counter > 5):
                continue
        print(f"Calculating data for {simulation_file}")
        one_hot_data = data['one_hot']
        x_data = data['x']

        # atom identities
        if(DO_ATOM_ID_CALCS):
            filter_H = False
            if(filter_H):
                atom_finalized_iters = find_atom_finalized_iters_filter_H(one_hot_data)
            else:
                atom_finalized_iters = find_atom_finalized_iters(one_hot_data)

            # Mean and stdev for each file
            # to see if some chains are far faster than others
            all_atom_id_means.append(np.mean(atom_finalized_iters))
            all_atom_id_stdev.append(np.std(atom_finalized_iters))

            all_atom_finalized_iters.extend(atom_finalized_iters)
            mol_finalized_iter = max(atom_finalized_iters)
            all_mol_finalized_iters.append(mol_finalized_iter)

            num_iters = 1000
            num_atoms = 19
            cumulative_atom_id_stability = np.zeros(num_iters)    # cumulate the atom stability. shape is num_iters. should start at 0 and go up to num_atoms
            for i in range(num_atoms):
                this_atom_stable_iter = int(atom_finalized_iters[i])
                this_atom_stability = 100 * np.append(np.full(this_atom_stable_iter, 0), np.full(num_iters - this_atom_stable_iter, 1)) / num_atoms
                cumulative_atom_id_stability += this_atom_stability
            all_cumulative_atom_id_stability.append(cumulative_atom_id_stability)
            cumulative_mol_id_stability = 100 * np.append(np.full(int(mol_finalized_iter), 0), np.full(num_iters - int(mol_finalized_iter), 1))
            all_cumulative_mol_id_stability.append(cumulative_mol_id_stability)

        # bond orders
        if(DO_BOND_ORDER_CALCS):
            bond_orders_from_final = find_atom_finalized_bond_count_iters(one_hot_data, x_data)

            all_atom_bo_means.append(np.mean(bond_orders_from_final))
            all_atom_bo_stdev.append(np.std(bond_orders_from_final))

            all_atom_bond_order_finalized_iters.extend(bond_orders_from_final)
            mol_bo_finalized_iter = max(bond_orders_from_final)
            all_mol_bond_order_finalized_iters.append(mol_bo_finalized_iter)

            num_iters = 1000
            num_atoms = 19
            cumulative_atom_bo_stability = np.zeros(num_iters)    # cumulate the atom stability. shape is num_iters. should start at 0 and go up to num_atoms
            for i in range(num_atoms):
                this_atom_bo_stable_iter = int(bond_orders_from_final[i])
                this_atom_bo_stability = 100 * np.append(np.full(this_atom_bo_stable_iter, 0), np.full(num_iters - this_atom_bo_stable_iter, 1)) / num_atoms
                cumulative_atom_bo_stability += this_atom_bo_stability
            all_cumulative_atom_bo_stability.append(cumulative_atom_bo_stability)
            cumulative_mol_bo_stability = 100 * np.append(np.full(int(mol_bo_finalized_iter), 0), np.full(num_iters - int(mol_bo_finalized_iter), 1))
            all_cumulative_mol_bo_stability.append(cumulative_mol_bo_stability)

        # fraction valid bond lengths
        if(DO_BOND_STABILITY_CALCS):
            all_iters_bond_stability.append(calc_all_iters_atom_stability(one_hot_data, x_data))  # shape is num_files x num_iters. each has the number of atoms that are stable

        # bond distance: is it within an error bar of the final bond distance? (Use percentage so it adapts to different types of atoms)
        if(DO_BOND_DISTANCE_CALCS):
            # last_iter_bond_distances = calc_one_iter_bond_distances(x_data[-1])
            finalized_bond_dists_iters = find_pairwise_finalized_bond_dist_iters(one_hot_data, x_data)

            all_atom_bd_means.append(np.mean(finalized_bond_dists_iters))
            all_atom_bd_stdev.append(np.std(finalized_bond_dists_iters))

            all_atom_bond_distance_finalized_iters.extend(finalized_bond_dists_iters)
            mol_bd_finalized_iter = max(finalized_bond_dists_iters)
            all_mol_bond_distance_finalized_iters.append(mol_bd_finalized_iter)

        # bond distance: average RMSD at each timestep
        if(DO_BOND_DISTANCE_RMSD):
            all_files_all_iters_bond_distance_rmsds.append(get_all_iters_bond_distance_rmsd(one_hot_data, x_data))

        # step size between iterations
        if(DO_STEP_SIZES):
            all_files_all_iters_step_size_rmsds.append(get_all_iters_step_size_rmsd(one_hot_data, x_data))
        

# MEETING NOTES
# ONLY grab ones that are nearby each other aka have bonds? No, instead: are close within a cutoff radius
# THIS PLOT WILL BE DIFFERENT. Y axis is angstroms. Show the average distance from final
# would want to see same trajectory as we see for gradient descent, 1/t or something
# torch.cdist(pos1, pos2) gets all pairwise distances!!
# np.linalg.norm(pos1[:,None] - pos2[None,:]) same with numpy. the None thing does broadcasting to make it a matrix instead of vector
# that is probably slightly wrong on the numpy one
# axis=2
# try filters like "where one of the atoms is H vs. the other bonds" or "has F" bc fewer fluorines in the dataset
# hopefully at iteration 960 there are reasonable sized deviations from final, and after that it is getting smaller. 
# use a larger fig size to get a bigger text size
# 2 plots: one with the full 1 to 1000, one with like 940 to 1000. indicate the zoomed section on the first one. 
# RMSD of the dists
# shaded error bars??
# https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
# if that looks bad, could plot a horizontal error bar showing the variation in the 50% point
# right hand y axis will be %

################
# CALCULATIONS #
################

# atom identities
if(DO_ATOM_ID_CALCS):
    # Average over the correct axis and also get stdev
    atom_stability_avg_to_plot = np.average(all_cumulative_atom_id_stability, axis=0)
    atom_stability_stdev_to_plot = np.std(all_cumulative_atom_id_stability, axis=0)
    mol_stability_avg_to_plot = np.average(all_cumulative_mol_id_stability, axis=0)
    mol_stability_stdev_to_plot = np.std(all_cumulative_mol_id_stability, axis=0)
    print("ATOM IDENTITIES")
    print("Mean atom finalized iteration for each simulation")
    print(all_atom_id_means)
    print("Stdev atom finalized iteration for each simulation")
    print(all_atom_id_stdev)
    print("Finalized iteration mean and stdev across all simulations:")
    print(f'{np.mean(all_atom_finalized_iters)} +/- {np.std(all_atom_finalized_iters)}')
    print("Molecule stability iterations")
    print(all_mol_finalized_iters)
    print("Max value")
    print(max(all_mol_finalized_iters))
    print("Mean and stdev")
    print(f'{np.mean(all_mol_finalized_iters)} +/- {np.std(all_mol_finalized_iters)}')

# bond order
if(DO_BOND_ORDER_CALCS):
    # Average over the correct axis and also get stdev
    atom_bo_stability_avg_to_plot = np.average(all_cumulative_atom_bo_stability, axis=0)
    atom_bo_stability_stdev_to_plot = np.std(all_cumulative_atom_bo_stability, axis=0)
    mol_bo_stability_avg_to_plot = np.average(all_cumulative_mol_bo_stability, axis=0)
    mol_bo_stability_stdev_to_plot = np.std(all_cumulative_mol_bo_stability, axis=0)
    print("BOND ORDER")
    print("Mean atom bond order finalized iteration for each simulation")
    print(all_atom_bo_means)
    print("Stdev atom bond order finalized iteration for each simulation")
    print(all_atom_bo_stdev)
    print("Finalized iteration mean and stdev across all simulations:")
    print(f'{np.mean(all_atom_bond_order_finalized_iters)} +/- {np.std(all_atom_bond_order_finalized_iters)}')
    print("Bond order stability iters")
    print(all_mol_bond_order_finalized_iters)
    print("Max value")
    print(max(all_mol_bond_order_finalized_iters))
    print("Mean and stdev")
    print(f'{np.mean(all_mol_bond_order_finalized_iters)} +/- {np.std(all_mol_bond_order_finalized_iters)}')

# bond stability (percent of atoms with valid bond number)
if(DO_BOND_STABILITY_CALCS):
    mol_bo_valid_avg_to_plot = np.average(all_iters_bond_stability, axis=0)
    mol_bo_valid_stdev_to_plot = np.std(all_iters_bond_stability, axis=0)

    # num_files = len(all_iters_bond_stability)
    # num_iters = len(all_iters_bond_stability[0])
    # sum_all_iters_bond_stability = np.zeros(num_iters)
    # for i in range(num_files):
    #     for j in range(num_iters):
    #         sum_all_iters_bond_stability[j] += all_iters_bond_stability[i][j]
    # num_atoms = 19
    # percentage_bond_stability = 100. * sum_all_iters_bond_stability / (num_files)

# bond distance finalization
if(DO_BOND_DISTANCE_CALCS):
    print("BOND DISTANCE within 5 percent of final")
    print("Mean atom bond distance finalized iteration for each simulation")
    print(all_atom_bd_means)
    print("Stdev atom bond distance finalized iteration for each simulation")
    print(all_atom_bd_stdev)
    print("Finalized iteration mean and stdev for atoms across all simulations:")
    print(f'{np.mean(all_atom_bond_distance_finalized_iters)} +/- {np.std(all_atom_bond_distance_finalized_iters)}')
    print("Bond distance molecule stability iters")
    print(all_mol_bond_distance_finalized_iters)
    print("Max value molecule stabilityu")
    print(max(all_mol_bond_distance_finalized_iters))
    print("Mean and stdev for molecule stability")
    print(f'{np.mean(all_mol_bond_distance_finalized_iters)} +/- {np.std(all_mol_bond_distance_finalized_iters)}')

# bond distance rmsd
if(DO_BOND_DISTANCE_RMSD):
    all_iters_average_rmsd = []
    all_iters_stdev_rmsd = []
    num_files = len(all_files_all_iters_bond_distance_rmsds)
    num_iters = len(all_files_all_iters_bond_distance_rmsds[0])
    for iter in range(num_iters):
        all_files_one_iter_all_rmsd = [this_file_all_iters_rmsd[iter] for this_file_all_iters_rmsd in all_files_all_iters_bond_distance_rmsds]
        all_iters_average_rmsd.append(np.average(all_files_one_iter_all_rmsd))
        all_iters_stdev_rmsd.append(np.std(all_files_one_iter_all_rmsd))

# step sizes
if(DO_STEP_SIZES):
    all_iters_average_step_size_rmsd = []
    all_iters_stdev_step_size_rmsd = []
    num_files = len(all_files_all_iters_step_size_rmsds)
    num_iters = len(all_files_all_iters_step_size_rmsds[0])
    for iter in range(num_iters):
        all_files_one_iter_step_size_rmsd = [this_file_all_iters_step_size_rmsd[iter] for this_file_all_iters_step_size_rmsd in all_files_all_iters_step_size_rmsds]
        all_iters_average_step_size_rmsd.append(np.average(all_files_one_iter_step_size_rmsd))
        all_iters_stdev_step_size_rmsd.append(np.std(all_files_one_iter_step_size_rmsd))

############
# PLOTTING #
############

# TIME BETWEEN
if(DO_ATOM_ID_CALCS and DO_BOND_ORDER_CALCS):
    # time between atoms' identities being fully stable and the bond orders being fully stable
    all_time_between = [bo_final_iter - atom_final_iter for (bo_final_iter, atom_final_iter) in zip(all_mol_bond_order_finalized_iters, all_mol_finalized_iters)]

    cumulative_time_between_to_plot = np.zeros(1000)
    for final_iter_value in all_time_between:
        for i in range(1000 - int(final_iter_value)):
            cumulative_time_between_to_plot[999 - i] += 1

    plt.plot(cumulative_time_between_to_plot)
    plt.title("Cumulative plot: Number of iterations between atom identities being stable and bond order being stable")
    plt.show()

# ATOM ID
if(DO_ATOM_ID_CALCS):
    atom_id_error_above = [atom_stability_avg_to_plot[i] + atom_stability_stdev_to_plot[i] for i in range(len(atom_stability_avg_to_plot))]
    atom_id_error_below = [atom_stability_avg_to_plot[i] - atom_stability_stdev_to_plot[i] for i in range(len(atom_stability_avg_to_plot))]
    plt.plot(atom_stability_avg_to_plot)
    plt.fill_between(range(1000), atom_id_error_below, y2=atom_id_error_above, alpha=0.2)
    plt.title("Percentage of atoms per molecule fully atom-identity-stable after N iterations")
    plt.show()

    mol_id_error_above = [mol_stability_avg_to_plot[i] + mol_stability_stdev_to_plot[i] for i in range(len(mol_stability_avg_to_plot))]
    mol_id_error_below = [mol_stability_avg_to_plot[i] - mol_stability_stdev_to_plot[i] for i in range(len(mol_stability_avg_to_plot))]
    plt.plot(mol_stability_avg_to_plot)
    plt.fill_between(range(1000), mol_id_error_below, y2=mol_id_error_above, alpha=0.2)
    plt.title("Percentage of molecules fully atom-identity-stable after N iterations")
    plt.show()
    # OLD WAY
    if(False):
        cumulative_mol_to_plot = np.zeros(1000)
        for final_iter_value in all_mol_finalized_iters:
            for i in range(1000 - int(final_iter_value)):
                cumulative_mol_to_plot[999 - i] += 1

        plt.plot(cumulative_mol_to_plot)
        plt.title("Number of molecules fully atom-identity-stable after N iterations")
        plt.show()

        cumulative_mol_to_plot_percent = 100. * cumulative_mol_to_plot / cumulative_mol_to_plot[-1]

        cumulative_atom_to_plot = np.zeros(1000)
        for final_iter_value in all_atom_finalized_iters:
            for i in range(1000 - int(final_iter_value)):
                cumulative_atom_to_plot[999 - i] += 1

        plt.plot(cumulative_atom_to_plot)
        plt.title("Number of atoms fully atom-identity-stable after N iterations")
        plt.show()

        cumulative_atom_to_plot_percent = 100. * cumulative_atom_to_plot / 1900.
        plt.plot(cumulative_atom_to_plot_percent)
        plt.title("Percent of atoms fully atom-identity-stable after N iterations")
        plt.show()

# BOND ORDER
if(DO_BOND_ORDER_CALCS):
    atom_bo_error_above = [atom_bo_stability_avg_to_plot[i] + atom_bo_stability_stdev_to_plot[i] for i in range(len(atom_bo_stability_avg_to_plot))]
    atom_bo_error_below = [atom_bo_stability_avg_to_plot[i] - atom_bo_stability_stdev_to_plot[i] for i in range(len(atom_bo_stability_avg_to_plot))]
    plt.plot(atom_bo_stability_avg_to_plot)
    plt.fill_between(range(1000), atom_bo_error_below, y2=atom_bo_error_above, alpha=0.2)
    plt.title("Percentage of atoms per molecule fully bond-order-stable after N iterations")
    plt.show()

    mol_bo_error_above = [mol_bo_stability_avg_to_plot[i] + mol_bo_stability_stdev_to_plot[i] for i in range(len(mol_bo_stability_avg_to_plot))]
    mol_bo_error_below = [mol_bo_stability_avg_to_plot[i] - mol_bo_stability_stdev_to_plot[i] for i in range(len(mol_bo_stability_avg_to_plot))]
    plt.plot(mol_bo_stability_avg_to_plot)
    plt.fill_between(range(1000), mol_bo_error_below, y2=mol_bo_error_above, alpha=0.2)
    plt.title("Percentage of molecules fully bond-order-stable after N iterations")
    plt.show()
    # OLD WAY
    if(False):
        cumulative_mol_bond_order_to_plot = np.zeros(1000)
        for final_iter_value in all_mol_bond_order_finalized_iters:
            for i in range(1000 - int(final_iter_value)):
                cumulative_mol_bond_order_to_plot[999 - i] += 1

        plt.plot(cumulative_mol_bond_order_to_plot)
        plt.title("Number of molecules fully bond-order-stable after N iterations")
        plt.show()

        cumulative_atom_bond_order_to_plot = np.zeros(1000)
        for final_iter_value in all_atom_bond_order_finalized_iters:
            for i in range(1000 - int(final_iter_value)):
                cumulative_atom_bond_order_to_plot[999 - i] += 1

        plt.plot(cumulative_atom_bond_order_to_plot)
        plt.title("Number of atoms fully bond-order-stable after N iterations")
        plt.show()

        # plt.plot(atom_stability_avg_to_plot, label='atom identity stable')
        # plt.plot(cumulative_mol_bond_order_to_plot, label='bond order stable')
        # plt.legend()
        # plt.show()

        percentage_atom_bo_stable = 100. * cumulative_atom_bond_order_to_plot / cumulative_atom_bond_order_to_plot[-1]
        percentage_mol_bo_stable = 100. * cumulative_mol_bond_order_to_plot / cumulative_mol_bond_order_to_plot[-1]

# BOND STABILITY
if(DO_BOND_STABILITY_CALCS):
    mol_bo_valid_error_above = [mol_bo_valid_avg_to_plot[i] + mol_bo_valid_stdev_to_plot[i] for i in range(len(mol_bo_valid_avg_to_plot))]
    mol_bo_valid_error_below = [mol_bo_valid_avg_to_plot[i] - mol_bo_valid_stdev_to_plot[i] for i in range(len(mol_bo_valid_avg_to_plot))]
    plt.plot(mol_bo_valid_avg_to_plot, label='valid atom bond orders')
    plt.fill_between(range(1000), mol_bo_valid_error_below, y2=mol_bo_valid_error_above, alpha=0.2)
    plt.legend()
    plt.show()


# BOND DISTANCE FINALIZATION
if(DO_BOND_DISTANCE_CALCS):
    cumulative_mol_bond_distance_to_plot = np.zeros(1000)
    for final_iter_value in all_mol_bond_distance_finalized_iters:
        for i in range(1000 - int(final_iter_value)):
            cumulative_mol_bond_distance_to_plot[999 - i] += 1

    plt.plot(cumulative_mol_bond_distance_to_plot)
    plt.title("Number of molecules fully bond-distance-stable after N iterations")
    plt.show()

    cumulative_atom_bond_distance_to_plot = np.zeros(1000)
    for final_iter_value in all_atom_bond_distance_finalized_iters:
        for i in range(1000 - int(final_iter_value)):
            cumulative_atom_bond_distance_to_plot[999 - i] += 1

    plt.plot(cumulative_atom_bond_distance_to_plot)
    plt.title("Number of atoms fully bond-distance-stable after N iterations")
    plt.show()

    plt.plot(cumulative_mol_to_plot, label='atom identity stable')
    plt.plot(cumulative_mol_bond_distance_to_plot, label='bond distance stable')
    plt.legend()
    plt.show()



    # argmax, then the function that splits an array where there is a change
    # OR if that doesn't exist, then compare pairwise by subtracting, then do numpy nonzero


    # Goal data to get:
    # 1000 by 1 array   each entry represents the fraction of atoms for which the bond order matches the bond identity

# BOND DISTANCE RMSD
if(DO_BOND_DISTANCE_RMSD):
    rmsd_error_above = [all_iters_average_rmsd[i] + all_iters_stdev_rmsd[i] for i in range(len(all_iters_average_rmsd))]
    rmsd_error_below = [all_iters_average_rmsd[i] - all_iters_stdev_rmsd[i] for i in range(len(all_iters_average_rmsd))]
    plt.plot(all_iters_average_rmsd)
    plt.fill_between(range(1000), rmsd_error_below, y2=rmsd_error_above, alpha=0.2)
    plt.title('Average RMS distance from final bond length for bonded pairs at each iteration')
    plt.show()

# STEP SIZES
if(DO_STEP_SIZES):
    all_iters_average_step_size_rmsd[0] = all_iters_average_step_size_rmsd[1]
    all_iters_stdev_step_size_rmsd[0] = 0
    step_size_rmsd_error_above = [all_iters_average_step_size_rmsd[i] + all_iters_stdev_step_size_rmsd[i] for i in range(len(all_iters_average_step_size_rmsd))]
    step_size_rmsd_error_below = [all_iters_average_step_size_rmsd[i] - all_iters_stdev_step_size_rmsd[i] for i in range(len(all_iters_average_step_size_rmsd))]
    plt.plot(all_iters_average_step_size_rmsd)
    plt.fill_between(range(1000), step_size_rmsd_error_below, y2=step_size_rmsd_error_above, alpha=0.2)
    plt.title('Average RMS distance from previous position for all atoms at each iteration')
    plt.show()

if(DO_BOND_DISTANCE_RMSD and DO_STEP_SIZES):
    plt.plot(all_iters_average_rmsd, label='avg dist from final')
    plt.plot(all_iters_average_step_size_rmsd, label='avg step size')
    plt.title('distance from final position and step sizes')
    plt.legend()
    plt.show()

PLOT_EVERYTHING = DO_ATOM_ID_CALCS and DO_BOND_ORDER_CALCS and DO_BOND_STABILITY_CALCS and DO_BOND_DISTANCE_RMSD and DO_STEP_SIZES

if(PLOT_EVERYTHING):
    fig, ax1 = plt.subplots(figsize=(4.2,3.5))
    ax1.invert_xaxis()

    x = list(reversed(range(1000)))

    perc_y_max = 102
    perc_y_min = -2
    angs_y_max = 1.75
    angs_y_min = angs_y_max * perc_y_min / perc_y_max

    # LEFT Y AXIS: percentages
    ax1.set_xlabel('Diffusion Step')

    # ax1.set_ylabel('Percentage Stabilized (%)')
    ax1.set_ylabel('Angstroms')
    ax1.set_ylim(angs_y_min, angs_y_max)
    ax1.tick_params(labelright=False)
    ax1.plot(x, all_iters_average_step_size_rmsd, label='Step Size', color='C0')
    ax1.fill_between(x, step_size_rmsd_error_below, y2=step_size_rmsd_error_above, alpha=0.2, color='C0')
    ax1.plot(x, all_iters_average_rmsd, label='Bond Len. RMSD', color='C1')
    ax1.fill_between(x, rmsd_error_below, y2=rmsd_error_above, alpha=0.2, color='C1')

    # RIGHT Y AXIS: angstroms
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #ax2.invert_xaxis()
    
    ax2.set_ylabel('Percent')
    ax2.set_ylim(perc_y_min, perc_y_max)

    # ax2.plot(cumulative_atom_to_plot_percent, label='element finalized (atoms)', color='C2')
    # ax2.plot(cumulative_mol_to_plot_percent, label='element finalized (molecules)', color='C3')
    ax2.plot(x, atom_stability_avg_to_plot, label='Atom Elem. Final', color='C2')
    ax2.fill_between(x, atom_id_error_below, y2=atom_id_error_above, alpha=0.2, color='C2')
    ax2.plot(x, mol_stability_avg_to_plot, label='Mol. Elem. Final', color='C3')
    ax2.fill_between(x, mol_id_error_below, y2=mol_id_error_above, alpha=0.2, color='C3')
    # ax2.plot(percentage_atom_bo_stable, label='bond order finalized (atoms)', color='C4')
    # ax2.plot(percentage_mol_bo_stable, label='bond order finalized (molecules)', color='C5')
    ax2.plot(x, atom_bo_stability_avg_to_plot, label='Atom BO Final', color='C4')
    ax2.fill_between(x, atom_bo_error_below, y2=atom_bo_error_above, alpha=0.2, color='C4')
    ax2.plot(x, mol_bo_stability_avg_to_plot, label='Mol BO Final', color='C5')
    ax2.fill_between(x, mol_bo_error_below, y2=mol_bo_error_above, alpha=0.2, color='C5')
    # ax2.plot(percentage_bond_stability, label='valid atoms', color='C6')
    ax2.plot(x, mol_bo_valid_avg_to_plot, label='Atom Valid BO', color='C6')
    ax2.fill_between(x, mol_bo_valid_error_below, y2=mol_bo_valid_error_above, alpha=0.2, color='C6')

    fig.legend(loc="upper center", bbox_to_anchor=(0.50, 0.97))
    #plt.show()
    plt.tight_layout()
    plt.savefig("writeup/figures/fig1_me.pdf")

    # SECOND PLOT FOR ZOOM
    fig, ax3 = plt.subplots(figsize=(4.2,3.5))
    ax3.invert_xaxis()
    xlims = [75, 0]
    perc_y_max = 102
    perc_y_min = -2
    angs_y_max = 0.5
    angs_y_min = angs_y_max * perc_y_min / perc_y_max
    # angstroms
    ax3.set_xlabel('Diffusion Step')
    ax3.set_ylabel('Angstroms')
    ax3.set_xlim(xlims)
    ax3.set_ylim(angs_y_min, angs_y_max)

    ax3.plot(x, all_iters_average_step_size_rmsd, label='step size', color='C0')
    ax3.fill_between(x, step_size_rmsd_error_below, y2=step_size_rmsd_error_above, alpha=0.2, color='C0')
    ax3.plot(x, all_iters_average_rmsd, label='delta from final bond length', color='C1')
    ax3.fill_between(x, rmsd_error_below, y2=rmsd_error_above, alpha=0.2, color='C1')
    # percentages
    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    ax4.invert_xaxis()
    ax4.set_ylabel("Percent")
    ax4.set_xlim(xlims)
    ax4.set_ylim(perc_y_min, perc_y_max)
    # ax4.plot(cumulative_atom_to_plot_percent, label='element finalized (atoms)', color='C2')
    # ax4.plot(cumulative_mol_to_plot_percent, label='element finalized (molecules)', color='C3')
    ax4.plot(x, atom_stability_avg_to_plot, label='element finalized (atom)', color='C2')
    ax4.fill_between(x, atom_id_error_below, y2=atom_id_error_above, alpha=0.2, color='C2')
    ax4.plot(x, mol_stability_avg_to_plot, label='elements finalized (molecule)', color='C3')
    ax4.fill_between(x, mol_id_error_below, y2=mol_id_error_above, alpha=0.2, color='C3')
    # ax4.plot(percentage_atom_bo_stable, label='bond order finalized (atoms)', color='C4')
    # ax4.plot(percentage_mol_bo_stable, label='bond order finalized (molecules)', color='C5')
    ax4.plot(x, atom_bo_stability_avg_to_plot, label='bond order finalized (atom)', color='C4')
    ax4.fill_between(x, atom_bo_error_below, y2=atom_bo_error_above, alpha=0.2, color='C4')
    ax4.plot(x, mol_bo_stability_avg_to_plot, label='bond order finalized (molecule)', color='C5')
    ax4.fill_between(x, mol_bo_error_below, y2=mol_bo_error_above, alpha=0.2, color='C5')
    # ax4.plot(percentage_bond_stability, label='valid atoms', color='C6')
    ax4.plot(x, mol_bo_valid_avg_to_plot, label='valid atom bond orders', color='C6')
    ax4.fill_between(x, mol_bo_valid_error_below, y2=mol_bo_valid_error_above, alpha=0.2, color='C6')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #fig.legend(loc='upper left')
    # fig.legend(loc='upper center')
    #plt.show()
    plt.savefig("writeup/figures/fig1_me_zoom.pdf")
