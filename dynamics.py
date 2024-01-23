import numpy as np
from numba import njit

import constants
import initialize


@njit
def kawasaki_in_jit_mode(matrix, T, neighbors_matrix, different_neighbor_numbers, steps_num: int, exponent_dict):
    """
    Run Kawasaki dynamics for
    :param matrix:
    :param T:
    :param neighbors_matrix:
    :param different_neighbor_numbers:
    :param steps_num:
    :param exponent_dict:
    :return:
    """
    for i in range(steps_num):
        matrix, neighbors_matrix, different_neighbor_numbers = kawasaki_time_step(matrix, T, neighbors_matrix,
                                                                                  different_neighbor_numbers,
                                                                                  exponent_dict)
    return matrix, neighbors_matrix, different_neighbor_numbers


@njit
def kawasaki_time_step(matrix, temperature, neighbors_matrix, different_neighbor_numbers, try_until_change_is_made=True,
                       return_energy=False):
    """
    preforms one kawasaki time step:
        pick random particle A
        pick random different neighbor B
        check if they are the same:
            continue
        else
            calculate current system energy E and energy after the switch E'
            calculate the probabilities(boltzmann distribution)
            flip with probability p'/(p+p')



    :param matrix: matrix (N,N) representing the spin lattice
    :param temperature: the temperature of the system
    :param neighbors_matrix: neighbor matrix as defined in initialize.create_neighbors_matrix
    :param different_neighbor_numbers: the number of neighbors from the other spun type for every site
    :param try_until_change_is_made: whether to try for an accepted step or just try once
    :param return_energy: this variable is only for api reasons
    :return: updated: matrix, neighbors_matrix, different_neighbor_numbers
    """
    N = matrix.shape[0]
    for time in range(
            constants.TIMEOUT * try_until_change_is_made + 1 * ~try_until_change_is_made):  # continue until a step is made
        i, j, neighbor_i, neighbor_j = get_site_and_neighbor(N, matrix, neighbors_matrix, different_neighbor_numbers)
        delta_E = calculate_energy_diff(matrix, (i, j), (neighbor_i, neighbor_j), different_neighbor_numbers)
        # might need to change probability calculation
        if delta_E <= 0:
            # cluster_matrix = swap_matrix
            matrix, neighbors_matrix, different_neighbor_numbers = swap_neighbors(matrix, (i, j),
                                                                                  (neighbor_i, neighbor_j),
                                                                                  neighbors_matrix,
                                                                                  different_neighbor_numbers)
            break
        else:
            probability = np.exp(-delta_E / temperature)
            # probability=exponents_dict[delta_E]
            if np.random.rand() < probability:
                # cluster_matrix = swap_matrix
                matrix, neighbors_matrix, different_neighbor_numbers = swap_neighbors(matrix, (i, j),
                                                                                      (neighbor_i, neighbor_j),
                                                                                      neighbors_matrix,
                                                                                      different_neighbor_numbers)
                break
            continue
    return matrix, neighbors_matrix, different_neighbor_numbers


@njit
def kawasaki_time_step_with_energy(matrix, temperature, neighbors_matrix, different_neighbor_numbers,
                                   try_until_change_is_made=True):
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


    :param matrix: matrix (N,N) representing the spin lattice
    :param temperature: the temperature of the system
    :param neighbors_matrix: neighbor matrix as defined in initialize.create_neighbors_matrix
    :param different_neighbor_numbers: the number of neighbors from the other spin type for every site
    :param try_until_change_is_made: whether to try for an accepted step or just try once
    :return: updated:matrix, neighbors_matrix, different_neighbor_numbers.
             delta_E the energy difference, step number of tries till we made the change
    """
    N = matrix.shape[0]
    delta_E = 0
    for step in range(
            constants.TIMEOUT * try_until_change_is_made + 1 * (
            not try_until_change_is_made)):  # continue until a step is made
        i, j, neighbor_i, neighbor_j = get_site_and_neighbor(N, matrix, neighbors_matrix, different_neighbor_numbers)
        delta_E = calculate_energy_diff(matrix, (i, j), (neighbor_i, neighbor_j), different_neighbor_numbers)
        # might need to change probability calculation
        if delta_E <= 0:
            # cluster_matrix = swap_matrix
            matrix, neighbors_matrix, different_neighbor_numbers = swap_neighbors(matrix, (i, j),
                                                                                  (neighbor_i, neighbor_j),
                                                                                  neighbors_matrix,
                                                                                  different_neighbor_numbers)
            break
        else:
            probability = np.exp(-delta_E / temperature)
            # probability=exponents_dict[delta_E]
            if np.random.rand() < probability:
                # cluster_matrix = swap_matrix
                matrix, neighbors_matrix, different_neighbor_numbers = swap_neighbors(matrix, (i, j),
                                                                                      (neighbor_i, neighbor_j),
                                                                                      neighbors_matrix,
                                                                                      different_neighbor_numbers)
                break
        delta_E = 0
    return matrix, neighbors_matrix, different_neighbor_numbers, delta_E, step + 1


@njit
def get_site_and_neighbor(N, matrix, neighbors_matrix, different_neighbor_numbers):
    """
    get two neighboring sites with different state
    :param N: length of the matrix in each dimension
    :param matrix: matrix representing the spin lattice
    :param neighbors_matrix: a matrix with shape matrix_side*matrix_side*2d matrix
    that indicates which neighbors has different value for each site
    :param different_neighbor_numbers: a matrix containing the number of neighboring sites with different sign
    :return: row_1, col_1, row_2, col_2 for the two sites
    """
    # because we want to switch between values with different value, choosing
    # only from one type is the same
    minority_places = np.where((matrix == constants.MINORITY_VALUE) & (different_neighbor_numbers > 0))
    ind = np.random.randint(0, len(minority_places[0]))
    i, j = minority_places[0][ind], minority_places[1][ind]
    # i, j = np.random.randint(0, matrix_side, 2)
    neighbor = constants.NEIGHBOR[neighbors_matrix[i, j]][np.random.randint(0, different_neighbor_numbers[
        i, j])]  # problem with probability of picking an edge - where do you account for that?
    neighbor_i = (i + neighbor[0]) % N  # currently we do periodic edges
    neighbor_j = (j + neighbor[1]) % N
    return i, j, neighbor_i, neighbor_j


def increase_density(matrix, higher_density):
    """
    increase the density of the constants.MINORITY_VALUE in the matrix to higher density
    :param matrix: spin lattice matrix
    :param higher_density: new density of MINORITY_VALUE
    :return: updated matrix, new neighbors matrix
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
    calculate the energy difference for switching point A and B, assuming they are nearest neighbors
    :param matrix: spin lattice matrix
    :param point_A: first point row and column index, tuple(row,column)
    :param point_B: second point row and column index, tuple(row,column)
    :param different_neighbor_numbers: a matrix containing the number of neighboring sites with different sign
    :return: energy difference for swapping the two points
    """
    i, j = point_A
    neighbor_i, neighbor_j = point_B
    return 2 * constants.CONSTANT_J * (
            4 * constants.DIMENSION + 2 - 2 * (different_neighbor_numbers[i, j] + different_neighbor_numbers[
                neighbor_i, neighbor_j]))


@njit
def swap_neighbors(matrix, point_A, point_B, neighbors_matrix, different_neighbor_numbers):
    """
    swapping two neighboring spins with different value and changing the
    neighbors matrix and different_neighbor_numbers matrix to represent the new
    state.
    :param matrix: spin lattice matrix
    :param point_A: first point row and column index, tuple(row,column)
    :param point_B: second point row and column index, tuple(row,column)
    :param neighbors_matrix:
    :param different_neighbor_numbers: a matrix containing the number of neighboring sites with different sign
    :return: updated spin matrix, updated neighbor matrix, updated neighbors numbers matrix
    """
    i, j = point_A
    neighbor_i, neighbor_j = point_B
    matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
        i, j]
    for neighbor_num, neighbor in enumerate(constants.NEIGHBOR):
        # original neighbor is the neighbor of i,j
        original_neighbor_i = (i - neighbor[0]) % constants.MATRIX_SIDE_LENGTH
        original_neighbor_j = (j - neighbor[1]) % constants.MATRIX_SIDE_LENGTH
        neighbors_matrix[original_neighbor_i, original_neighbor_j, neighbor_num] = ~neighbors_matrix[
            original_neighbor_i, original_neighbor_j, neighbor_num]
        different_neighbor_numbers[original_neighbor_i, original_neighbor_j] = np.sum(
            neighbors_matrix[original_neighbor_i, original_neighbor_j])

        neighbor_neighbor_i = (neighbor_i - neighbor[0]) % constants.MATRIX_SIDE_LENGTH
        neighbor_neighbor_j = (neighbor_j - neighbor[1]) % constants.MATRIX_SIDE_LENGTH
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
    """
    Calculates the energy for all the bonds involving the given point
    :param matrix: spin lattice matrix
    :param point: (row, col) coordinates of the point
    :param different_neighbor_numbers: a matrix containing the number of neighboring sites with different sign
    :return: the energy from bonds containing the given point
    """
    i, j = point
    N = matrix.shape[0]
    return -constants.CONSTANT_J * (2 * len(N) - 2 * different_neighbor_numbers[i, j])


@njit
def calculate_energy(matrix, different_neighbor_numbers):
    """
    calculate the energy of the system
    :param matrix: spin lattice matrix
    :param different_neighbor_numbers: a matrix containing the number of neighboring sites with different sign
    :return: total energy of the system
    """
    return -constants.CONSTANT_J * (
            constants.DIMENSION * constants.N_SPINS - np.sum(different_neighbor_numbers))
