import os.path
import time

import numpy as np
from tqdm import tqdm

import clusters
import constants
import dynamics
import initialize
import utils


def run_with_densities(N: int = constants.MATRIX_SIDE_LENGTH, densities=[constants.INITIAL_ZERO_DENSITY],
                       T: float = constants.INITIAL_TEMPERATURE, steps: int = constants.NUMBER_OF_STEPS,
                       gif_name: str = "visual/vis.gif"):
    """
    Run a Monte-Carlo simulation of Kawasaki dynamics with slowly increasing
    densities of the minority spin type and return the spin lattice after the simulation.
    :param N: the side of the system, the spin lattice is of size (N, N)
    :param densities: list of densities to use, each density is used for the same
    number of steps.
    :param T: the temperature of the system.
    :param steps: total number of simulation steps.
    :param gif_name: old parameter, default "visual/vis.gif"
    :return: (N, N) matrix representing the final state of the lattice.
    """
    matrix = initialize.initialize_random_binary_matrix(N, densities[0])
    neighbors_matrix = initialize.create_neighbors_matrix(matrix)
    different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    densities_number = len(densities)
    steps_per_density = steps // densities_number

    for i in tqdm(range(steps_per_density * (densities_number) - 1)):
        matrix, neighbors_matrix, different_neighbor_numbers = dynamics.kawasaki_time_step(matrix, T, neighbors_matrix,
                                                                                           different_neighbor_numbers)
        # if (i * constants.NUMBER_OF_FRAMES) % steps == 0:
        #     energy.append(dynamics.calculate_energy(matrix, different_neighbor_numbers))
        #     imgs.append(Image.fromarray((matrix == constants.MINORITY_VALUE) * 255))
        if i % steps_per_density == 0:
            if i == 0:
                continue
            matrix, neighbors_matrix = dynamics.increase_density(matrix, densities[i // steps_per_density])
            different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    matrix, neighbors_matrix, different_neighbor_numbers = dynamics.kawasaki_time_step(matrix, T, neighbors_matrix,
                                                                                       different_neighbor_numbers)
    # energy.append(dynamics.calculate_energy(matrix, different_neighbor_numbers))
    # imgs.append(Image.fromarray((matrix == constants.MINORITY_VALUE) * 255))
    # imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=1, loop=1)  #
    return matrix


def multiple_identical_runs_mean_and_std(initial_den, final_den, total_change_steps, steps_at_final_den,
                                         times_to_repeat):
    """
    run specific scheduling and calculates the mead and std of the clusters number
    :param initial_den: initial MINORITY density
    :param final_den: final MINORITY density
    :param total_change_steps: how many steps between initial and final density
    :param steps_at_final_den: after all the changes, how many step we want in the final density?
    :param times_to_repeat: how many times to repeat this scheduling
    :return: mean cluster num, clusters number standard deviation
    """
    dens = np.concatenate([np.linspace(initial_den, final_den, total_change_steps), [final_den] * steps_at_final_den])
    clusters_num = []
    for i in range(times_to_repeat):
        matrix = run_with_densities(T=0.7, densities=dens, gif_name="visual/densities.gif")
        clusters_num.append(clusters.get_clusters_number(matrix, constants.MINORITY_VALUE, constants.MIN_CLUSTER_SIZE))
    return np.mean(clusters_num), np.std(clusters_num)


def clusters_time_dependence(final_den, spins_per_change, T: float = constants.INITIAL_TEMPERATURE,
                             steps: int = constants.NUMBER_OF_STEPS, try_until_change_is_made=True,
                             number_of_snapshots_for_constant_run=50):
    """
    run a simulation and saves the state of the matrix before and after each
     density change
    :param try_until_change_is_made:
    :param final_den:
    :param spins_per_change:
    :param T:
    :param steps:
    :return: matrices_before_change
             matrices_after_change
             change_times - note that changes times is for the matrices before change
                             the times after changes will be [0]+change_times[:-1]
             tries_change_times - density change times, as measured with steps inside kawasaki time step
    """
    if spins_per_change == 0:
        densities = np.ones(number_of_snapshots_for_constant_run) * final_den
    else:
        densities = np.arange(3, final_den * constants.N_SPINS, spins_per_change) / constants.N_SPINS
    matrices_after_change, matrices_before_change, change_times = [], [], []
    matrix = initialize.initialize_random_binary_matrix(constants.MATRIX_SIDE_LENGTH, densities[0])
    neighbors_matrix = initialize.create_neighbors_matrix(matrix)
    different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    densities_number = len(densities)
    steps_per_density = steps // densities_number
    matrices_after_change.append(matrix + 0)
    tries_time = 0
    tries_change_times = []
    for i in tqdm(range(steps_per_density * densities_number)):
        matrix, neighbors_matrix, different_neighbor_numbers, _, tries_till_accept = \
            dynamics.kawasaki_time_step_with_energy(
                matrix, T, neighbors_matrix, different_neighbor_numbers,
                try_until_change_is_made=try_until_change_is_made)
        tries_time += tries_till_accept
        if i % steps_per_density == 0:
            if i == 0:
                continue
            change_times.append(i)
            matrices_before_change.append(matrix + 0)
            matrix, neighbors_matrix = dynamics.increase_density(matrix, densities[i // steps_per_density])
            matrices_after_change.append(matrix + 0)
            different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
            tries_change_times.append(tries_time)
    matrices_before_change.append(matrix + 0)
    change_times.append(steps_per_density * densities_number)
    return matrices_before_change, matrices_after_change, change_times, tries_change_times


def clusters_time_dependence_with_energy(final_den, spins_per_change, T: float = constants.INITIAL_TEMPERATURE,
                                         steps: int = constants.NUMBER_OF_STEPS,
                                         try_until_change_is_made=True, number_of_snapshots_for_constant_run=50):
    """
    run a simulation and saves the state of the matrix before and after each
     density change
    :param try_until_change_is_made:
    :param final_den:
    :param spins_per_change:
    :param T:
    :param steps:
    :param number_of_snapshots_for_constant_run:
    :return: matrices_before_change
             matrices_after_change
             change_times - note that changes times is for the matrices before change
                             the times after changes will be [0]+change_times[:-1]
             tries_change_times - density change times, as measured with steps inside kawasaki time step
             energies - energies time evolution
             energies_times - the times in which the energies were recorded
    """
    if spins_per_change == 0:
        densities = np.ones(number_of_snapshots_for_constant_run) * final_den
    else:
        densities = np.arange(3, final_den * constants.N_SPINS, spins_per_change) / constants.N_SPINS
    matrices_after_change, matrices_before_change, change_times = [], [], []
    matrix = initialize.initialize_random_binary_matrix(constants.MATRIX_SIDE_LENGTH, densities[0])
    neighbors_matrix = initialize.create_neighbors_matrix(matrix)
    different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    densities_number = len(densities)
    steps_per_density = steps // densities_number
    matrices_after_change.append(matrix + 0)
    tries_time = 0
    tries_change_times = []
    energies = [dynamics.calculate_energy(matrix, different_neighbor_numbers)]
    energy = energies[0]
    energies_times = [0]
    for i in tqdm(range(steps_per_density * densities_number)):
        matrix, neighbors_matrix, different_neighbor_numbers, delta_E, tries_till_accept = \
            dynamics.kawasaki_time_step_with_energy(
                matrix, T, neighbors_matrix, different_neighbor_numbers,
                try_until_change_is_made=try_until_change_is_made)
        tries_time += tries_till_accept
        if delta_E != 0:
            energy += delta_E
            energies.append(energy)
            energies_times.append(tries_time)
        if i % steps_per_density == 0:
            if i == 0:
                continue
            change_times.append(i)
            matrices_before_change.append(matrix + 0)
            matrix, neighbors_matrix = dynamics.increase_density(matrix, densities[i // steps_per_density])
            matrices_after_change.append(matrix + 0)
            different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
            tries_change_times.append(tries_time)
            energy = dynamics.calculate_energy(matrix, different_neighbor_numbers)
            energies.append(energy)
            energies_times.append(tries_time)

    matrices_before_change.append(matrix + 0)
    change_times.append(steps_per_density * densities_number)
    return matrices_before_change, matrices_after_change, change_times, tries_change_times, energies, energies_times


def run_configuration_to_saturation(matrix, temperature, one_flip_every_n_tries, save_every_n_tries):
    neighbors_matrix = initialize.create_neighbors_matrix(matrix)
    different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    tries = 0
    switch_times = [tries]
    cur_E = dynamics.calculate_energy(matrix, different_neighbor_numbers)
    Energies = [cur_E]
    times = [0]
    matrices = [matrix + 0]
    t = 0
    change_occurred = True
    while change_occurred and t < 1e12:
        if t % 1e7 == 0 and t != 0:
            print(t)
            if np.std(Energies) < 1e-3 or len(Energies) < t / 1e5 or abs(
                    Energies[-1] - Energies[max(0, t - int(1e7))]) < 10:
                break
        change_occurred = False
        for i in range(one_flip_every_n_tries):
            t += 1
            matrix, neighbors_matrix, different_neighbor_numbers, delta_E, tries_now = dynamics.kawasaki_time_step_with_energy(
                matrix,
                temperature,
                neighbors_matrix,
                different_neighbor_numbers,
                try_until_change_is_made=False)
            tries += tries_now
            if t % save_every_n_tries == 0:
                matrices.append(matrix + 0)
            if delta_E != 0:
                print(t, delta_E)
                times.append(t)
                cur_E += delta_E
                Energies.append(cur_E)
                switch_times.append(tries)
                change_occurred = True

    matrices.append(matrix + 0)
    return matrix, matrices, times, Energies, switch_times


def main(final_den, rep_num):
    mean_path = f"npy_files/mean_clusters_num_{final_den}_{constants.NUMBER_OF_STEPS}_steps rep_num_{rep_num}.npy"
    std_path = f"npy_files/clusters_std_map_{final_den}_{constants.NUMBER_OF_STEPS}_steps rep_num_{rep_num}.npy"
    initial_den = 3 / 64 ** 2
    changes_steps_nums_list = [int(((64 ** 2 - 3) / 4) // i) for i in range(1, 10)] + list(
        np.linspace(65, 0, 5, dtype=int))
    steps_at_final_density = [1, 2, 5, 10, 15, 20, 25, 30]
    if os.path.exists(mean_path):
        print("mean_file exists")
        mean_cluster_num_map = np.load(mean_path)
    else:
        print("mean_file does not exist, new table created")
        mean_cluster_num_map = np.zeros((len(steps_at_final_density), len(changes_steps_nums_list)))
    if os.path.exists(std_path):
        print("std_file exists")
        cluster_std_map = np.load(std_path)
    else:
        print("std_file does not exist, new table created")
        cluster_std_map = np.zeros((len(steps_at_final_density), len(changes_steps_nums_list)))
    try:
        for i, steps_at_final_den in enumerate(steps_at_final_density):
            for j, total_changes_steps_num in enumerate(changes_steps_nums_list):
                print(f"i, j = {i}, {j}")
                if mean_cluster_num_map[i, j] == 0:
                    mean_cluster_num_map[i, j], cluster_std_map[i, j] = multiple_identical_runs_mean_and_std(
                        initial_den, final_den,
                        total_changes_steps_num,
                        steps_at_final_den, rep_num)
                else:
                    print("skipped")
                print(f"mean clusters num: {mean_cluster_num_map[i, j]}")
    finally:
        np.save(mean_path, mean_cluster_num_map)
        np.save(std_path, cluster_std_map)
        print(f"{final_den} density heat map:")
        print(mean_cluster_num_map)
        print("cluster_std map:")
        print(cluster_std_map, end="\n\n")


def cluster_time_dependence_full_experiment(spin_number_per_change, experiment_num, final_density=0.1,
                                            folder="saved_matrices/", try_until_change_is_made=True,
                                            total_steps=constants.NUMBER_OF_STEPS, with_energies=False,
                                            costume_final_to_all_files="", temperature=constants.INITIAL_TEMPERATURE,
                                            number_of_snapshots_for_constant_run=50):
    """
    Run the experiment and saves the data to files.
    :param spin_number_per_change: how many spins to add in each density change.
    :param experiment_num: ordinal number of this experiment, used only for files names
    :param final_density: max density to reach in the experiment, defaults to 0.1
    :param folder: folder path to save the files, defaults to "saved_matrices/"
    :param try_until_change_is_made: whether to try each stem until some
    accepted step, defaults to True
    :param total_steps: the length of the simulation, defaults to constants.NUMBER_OF_STEPS
    :param with_energies: boolean flag for whether to save the energies of the
    system during the simulation, defaults to False
    :param costume_final_to_all_files: optional string to add at the end of the
    saved files, defaults to empty string
    :param temperature: temperature to do the experiment with, defaults to
    constants.INITIAL_TEMPERATURE
    :param number_of_snapshots_for_constant_run: how many times to save the
    state of the matrix in runs without density changes, defaults to 50
    """
    check_num = 0
    if with_energies:
        matrices_before_change, matrices_after_change, times, tries_change_times, energies, energies_times = \
            clusters_time_dependence_with_energy(
                final_density, spin_number_per_change, T=temperature, steps=total_steps,
                try_until_change_is_made=try_until_change_is_made,
                number_of_snapshots_for_constant_run=number_of_snapshots_for_constant_run)
        np.save(
            folder + f"energies, {spin_number_per_change} jumps, experiment {experiment_num} {costume_final_to_all_files}.npy",
            np.array(energies))
        np.save(
            folder + f"energies times, {spin_number_per_change} jumps, experiment {experiment_num} "
                     f"{costume_final_to_all_files}.npy",
            np.array(energies_times))
    else:
        matrices_before_change, matrices_after_change, times, tries_change_times = clusters_time_dependence(
            final_density, spin_number_per_change, T=temperature, steps=total_steps,
            try_until_change_is_made=try_until_change_is_made,
            number_of_snapshots_for_constant_run=number_of_snapshots_for_constant_run)
    matrices_before_change = np.array(matrices_before_change)
    matrices_after_change = np.array(matrices_after_change)
    utils.save_grid(utils.pack_grid(matrices_before_change),
                    folder + f"states_before_changes {check_num},"
                             f" {final_density} final density, "
                             f"{spin_number_per_change} every change, "
                             f"{experiment_num}"
                    + constants.WITHOUT_CHANGE_DEPENDENT_SCHEDULE
                    + costume_final_to_all_files + constants.MATRIX_EXTENSION)
    utils.save_grid(utils.pack_grid(matrices_after_change),
                    folder + f"states_after_changes {check_num},"
                             f" {final_density} final density, "
                             f"{spin_number_per_change} every change, "
                             f"{experiment_num}"
                    + constants.WITHOUT_CHANGE_DEPENDENT_SCHEDULE
                    + costume_final_to_all_files + constants.MATRIX_EXTENSION)
    np.save(folder + f"time, {spin_number_per_change}" +
            costume_final_to_all_files + constants.MATRIX_EXTENSION,
            np.array(times))
    np.save(folder + f"density change times, calculated by swap tries, {spin_number_per_change} jumps, "
                     f"experiment {experiment_num}" + costume_final_to_all_files + constants.MATRIX_EXTENSION,
            np.array(tries_change_times))
    print(f"cluster time dependence with {spin_number_per_change} every jump, number {experiment_num} done")


def saturation_wrapper(final_density, folder_path="saved_matrices/"):
    check_num = 0
    save_matrix_every = 3000
    more_then_one_flip_every_n_steps = 10000
    for spin_number_per_change in [0] + list(range(3, 12, 2)):
        for i in range(11):
            matrix_file_name = f"states_before_changes {check_num}," + \
                               f" {final_density} final density, " + \
                               f"{spin_number_per_change} every change, {i}" + constants.MATRIX_EXTENSION
            mat = utils.read_min_grid(folder_path + matrix_file_name,
                                      constants.MATRIX_SIDE_LENGTH)[-1]
            start_time = time.time()
            matrix, matrices, times, Energies, switch_times = run_configuration_to_saturation(mat,
                                                                                              constants.INITIAL_TEMPERATURE,
                                                                                              more_then_one_flip_every_n_steps,
                                                                                              save_matrix_every)
            end_time = time.time()
            if len(times) == 1:
                print("already saturated")
            utils.save_grid(utils.pack_grid(np.array(matrices)),
                            f"{folder_path}saturation for {matrix_file_name}, save every {save_matrix_every} tries" + constants.MATRIX_EXTENSION)
            np.save(folder_path + f"energies for {spin_number_per_change} jump, experiment {i}", Energies)
            np.save(folder_path + f"energies times for {spin_number_per_change} jump, experiment {i}", times)
            print(
                f"saturation for {spin_number_per_change} jumps, experiment {i} done in {end_time - start_time}s and ~{times[-1]} steps")


if __name__ == '__main__':
    # np.random.seed(1)
    # main(0.1, 5)
    # main(0.25, 5)
    # folder_path = "saved_matrices/"
    # folder_path = "saved_runs_with_tries_schedule_ten_million_steps_with_energies/"
    final_density = 0.1
    check_num = 0

    save_matrix_every = 3000
    more_then_one_flip_every_n_steps = 10000
    j = 0
    # for i in range(30, 35):

    # different rates:
    # for i in range(5, 15):
    #     for spin_number_per_change in [0, 1, 13, 2, 15]:  # [0]+list(range(1,18,2))
    #         print(f'spin number: {spin_number_per_change}, experiment {i}')
    #         cluster_time_dependence_full_experiment(spin_number_per_change, i, final_density, folder=folder_path,
    #                                                 try_until_change_is_made=False,
    #                                                 total_steps=10 * constants.NUMBER_OF_STEPS, with_energies=True)
    #     j += 1

    # different temperatures, constant (and short) run:
    folder_path = "saved_runs_for_temperature_changes_200_thousand_steps/"
    for i in range(150, 200):
        break  # this loop is already used
        temperatures = [1e-2, 1e-3, 1e-4] + [0.1, 0.9, 1.3, 1.7, 2.5, 3.3, 4.4, 5.8, 10]
        for T in temperatures:
            print(f'T={T}, number {i}')
            cluster_time_dependence_full_experiment(0, i, final_density, temperature=T, folder=folder_path,
                                                    try_until_change_is_made=False,
                                                    total_steps=int(2e5), with_energies=True,
                                                    costume_final_to_all_files=f', T={T}',
                                                    number_of_snapshots_for_constant_run=1000)

    # long runs with different temperatures and jumps:
    folder_path = "saved_runs_for_temperature_changes_10 million_steps/"
    for i in range(38, 40):
        for T in [0.1, 0.9, 1.3, 1.7, 2.5, 3.3, 4.4, 5.8, 10]:
            for jumps in [0, 1, 13, 2, 15]:
                print(f'T={T}, jump={jumps}, number {i}, second part')
                cluster_time_dependence_full_experiment(jumps, i, final_density, temperature=T, folder=folder_path,
                                                        try_until_change_is_made=False,
                                                        total_steps=10 * constants.NUMBER_OF_STEPS, with_energies=True,
                                                        costume_final_to_all_files=f', T={T}',
                                                        number_of_snapshots_for_constant_run=1000)
