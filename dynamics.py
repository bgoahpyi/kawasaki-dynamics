import numpy as np
from numba import jit, njit

import constants
import initialize



@njit
def kawasaki_in_jit_mode(matrix, T, neighbors_matrix, different_neighbor_numbers, steps_num: int, exponent_dict):
    for i in range(steps_num):
        matrix, neighbors_matrix, different_neighbor_numbers = kawasaki_time_step(matrix, T, neighbors_matrix,
                                                                                           different_neighbor_numbers, exponent_dict)
    return matrix, neighbors_matrix, different_neighbor_numbers


@njit
def kawasaki_time_step(matrix, temperature, neighbors_matrix, different_neighbor_numbers):
    """
    preforms one kawasaki time step:
        pick random particle A
        pick random neighbor B
        check if they are the same:
            continue
        else
            calculate current system energy E and energy after the flip E'
            calculate the probabilities(boltzmann)
            flip with probability p'/(p+p')


    :param matrix:
    :param temperature:
    :return: updated cluster_matrix
    """
    N = matrix.shape[0]
    for time in range(constants.TIMEOUT):  # continue until a step is made
        # timeout logic for not finding any swap

        i, j, neighbor_i, neighbor_j = get_site_and_neighbor(N, matrix, neighbors_matrix, different_neighbor_numbers)
        # if (cluster_matrix[i, j] == cluster_matrix[neighbor_i, neighbor_j]):
        #     continue

        # calculate E, swap and calculate E'
        # E = calculate_energy(cluster_matrix, J)
        # cluster_matrix[i, j], cluster_matrix[neighbor_i, neighbor_j] = cluster_matrix[neighbor_i, neighbor_j], cluster_matrix[
        # 	i, j]
        # E_prime = calculate_energy(cluster_matrix, J)

        delta_E = calculate_energy_diff(matrix, (i, j), (neighbor_i, neighbor_j), different_neighbor_numbers)
        # might need to change probability calculation
        if delta_E <= 0:
            # cluster_matrix = swap_matrix
            matrix, neighbors_matrix, different_neighbor_numbers = swap_points(matrix, (i, j), (neighbor_i, neighbor_j),
                                                                               neighbors_matrix,
                                                                               different_neighbor_numbers)

            break
        else:
            probability = np.exp(-delta_E / temperature)
            # probability=exponents_dict[delta_E]
            if np.random.rand() < probability:
                # cluster_matrix = swap_matrix
                matrix, neighbors_matrix, different_neighbor_numbers = swap_points(matrix, (i, j),
                                                                                   (neighbor_i, neighbor_j),
                                                                                   neighbors_matrix,
                                                                                   different_neighbor_numbers)
                break
            # cluster_matrix[i, j], cluster_matrix[neighbor_i, neighbor_j] = cluster_matrix[neighbor_i, neighbor_j], cluster_matrix[
            # 	i, j]
            continue
    return matrix, neighbors_matrix, different_neighbor_numbers


@njit
def get_site_and_neighbor(N, matrix, neighbors_matrix, different_neighbor_numbers):
    # because we want to switch between values with different value, choosing
    # only from one type is the same
    minority_places = np.where((matrix == constants.MINORITY_VALUE) & (different_neighbor_numbers > 0))
    ind = np.random.randint(0, len(minority_places[0]))
    i, j = minority_places[0][ind], minority_places[1][ind]
    # i, j = np.random.randint(0, N, 2)
    neighbor = constants.NEIGHBOR[neighbors_matrix[i, j]][np.random.randint(0, different_neighbor_numbers[
        i, j])]  # problem with probability of picking an edge - where do you account for that?
    neighbor_i = (i + neighbor[0]) % N  # currently we do periodic edges
    neighbor_j = (j + neighbor[1]) % N
    return i, j, neighbor_i, neighbor_j


def increment_density(matrix, higher_density):
    """
    how to update the density? pick a 0 site at random and flip it to 1?
    :return: changed cluster_matrix, new neighbors cluster_matrix
    """
    N_spins = np.prod(matrix.shape)
    original_density = np.sum(matrix == constants.MINORITY_VALUE) / N_spins
    if original_density >= higher_density:
        return matrix, initialize.create_neighbors_matrix(matrix)
    spins_to_add = int(N_spins * (higher_density - original_density))
    majority_indexes = np.where(matrix == constants.MAJORITY_VALUE)
    change_indexes = np.random.choice(np.array(majority_indexes[0].shape[0], dtype=int), spins_to_add, replace=False)
    matrix[
        tuple((dimension_indexes[change_indexes] for dimension_indexes in majority_indexes))] = constants.MINORITY_VALUE
    return matrix, initialize.create_neighbors_matrix(matrix)


@njit
def calculate_energy_diff(matrix, point_A, point_B, different_neighbor_numbers):
    """
    calculate the energy difference for switching pointA and B, assuming they are nearest neighbors
    :param matrix:
    :param point_A:
    :param point_B:
    :return: energy difference from swapping the two points
    """
    i, j = point_A
    neighbor_i, neighbor_j = point_B
    return 2 * constants.CONSTANT_J * (
            4 * constants.DIMENSION + 2 - 2 * (different_neighbor_numbers[i, j] + different_neighbor_numbers[
        neighbor_i, neighbor_j]))
    # delta E=-2 * original E, couple E=-J*(4d+2-2*(diff_num(A)+diff_num(B)))
    # before = calc_energy_point(cluster_matrix, point_A) + calc_energy_point(cluster_matrix, point_B)
    # cluster_matrix = swap_points(cluster_matrix, point_A, point_B)
    # after = calc_energy_point(cluster_matrix, point_A) + calc_energy_point(cluster_matrix, point_B)
    # cluster_matrix = swap_points(cluster_matrix, point_A, point_B)  # needed!!!
    # return after - before


@njit
def swap_points(matrix, point_A, point_B, neighbors_matrix, different_neighbor_numbers):
    i, j = point_A
    neighbor_i, neighbor_j = point_B
    matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
        i, j]
    for neighbor_num, neighbor in enumerate(constants.NEIGHBOR):
        # original neighbor is the neighbor of i,j
        original_neighbor_i = (i - neighbor[0]) % constants.MATRIX_SIZE
        original_neighbor_j = (j - neighbor[1]) % constants.MATRIX_SIZE
        neighbors_matrix[original_neighbor_i, original_neighbor_j, neighbor_num] = ~neighbors_matrix[
            original_neighbor_i, original_neighbor_j, neighbor_num]
        different_neighbor_numbers[original_neighbor_i, original_neighbor_j] = np.sum(
            neighbors_matrix[original_neighbor_i, original_neighbor_j])

        neighbor_neighbor_i = (neighbor_i - neighbor[0]) % constants.MATRIX_SIZE
        neighbor_neighbor_j = (neighbor_j - neighbor[1]) % constants.MATRIX_SIZE
        neighbors_matrix[neighbor_neighbor_i, neighbor_neighbor_j, neighbor_num] = ~neighbors_matrix[
            neighbor_neighbor_i, neighbor_neighbor_j, neighbor_num]
        different_neighbor_numbers[neighbor_neighbor_i, neighbor_neighbor_j] = np.sum(
            neighbors_matrix[neighbor_neighbor_i, neighbor_neighbor_j])

    neighbors_matrix[i, j] = ~neighbors_matrix[i, j]
    different_neighbor_numbers[i, j] = np.sum(neighbors_matrix[i, j])
    neighbors_matrix[neighbor_i, neighbor_j] = ~neighbors_matrix[neighbor_i, neighbor_j]
    different_neighbor_numbers[neighbor_i, neighbor_j] = np.sum(neighbors_matrix[neighbor_i, neighbor_j])
    return matrix, neighbors_matrix, different_neighbor_numbers


@njit
def calc_energy_point(matrix, point, different_neighbor_numbers):
    i, j = point
    N = matrix.shape[0]
    return -constants.CONSTANT_J * (2 * len(N) - 2 * different_neighbor_numbers[i, j])


@njit
def calculate_energy(matrix, different_neighbor_numbers):
    """
    calculate the energy of the system
    :param matrix:
    :return:
    """
    return -constants.CONSTANT_J * (
            constants.DIMENSION * constants.N_SPINS - np.sum(different_neighbor_numbers))
    # energy = -np.sum(cluster_matrix[1:] * cluster_matrix[:-1])
    # energy -= np.sum(cluster_matrix[0] * cluster_matrix[-1])
    # energy -= np.sum(cluster_matrix[:, 1:] * cluster_matrix[:, :-1])
    # energy -= np.sum(cluster_matrix[:, 0] * cluster_matrix[:, -1])
    # return constants.CONSTANT_J * energy
