import numpy as np
from numba import jit

import constants


def kawasaki_time_step(matrix, J, temperature):
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
    :return: updated matrix
    """
    N = matrix.shape[0]
    time = 0
    while True:  # continue until a step is made
        # timeout logic for not finding any swap
        time += 1
        if time >= constants.TIMEOUT:
            print("timeout")
            return matrix

        i, j, neighbor_i, neighbor_j = get_site_and_neighbor(N, matrix)
        if (matrix[i, j] == matrix[neighbor_i, neighbor_j]):
            continue

        # calculate E, swap and calculate E'
        # E = calculate_energy(matrix, J)
        # matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
        # 	i, j]
        # E_prime = calculate_energy(matrix, J)

        delta_E = calculate_energy_diff(matrix, J, (i, j), (neighbor_i, neighbor_j))
        # might need to change probability calculation
        if delta_E <= 0:
            # matrix = swap_matrix
            matrix = swap_points(matrix, (i, j), (neighbor_i, neighbor_j))

            break
        else:
            probability = np.exp(-delta_E / temperature)
            if np.random.rand() < probability:
                # matrix = swap_matrix
                matrix = swap_points(matrix, (i, j), (neighbor_i, neighbor_j))
                break
            # matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
            # 	i, j]
            continue
    return matrix


def get_site_and_neighbor(N, matrix):
    # because we want to switch between values with different value, choosing
    # only from one type is the same
    minority_places=np.where(matrix==constants.MINORITY_VALUE)
    ind = np.random.randint(len(minority_places[0]))
    i,j = minority_places[0][ind], minority_places[1][ind]
    # i, j = np.random.randint(0, N, 2)  # todo: change to choice for ==1
    neighbor = constants.NEIGHBOR[
        np.random.randint(0, 4)]  # problem with probability of picking an edge - where do you account for that?
    neighbor_i = (i + neighbor[0]) % N  # currently we do periodic edges
    neighbor_j = (j + neighbor[1]) % N
    return i, j, neighbor_i, neighbor_j

def increment_density(matrix, higher_density):
    """
    how to update the density? pick a 0 site at random and flip it to 1?
    :return: changed matrix
    """
    N_spins = np.prod(matrix.shape)
    original_density = np.sum(matrix == constants.MINORITY_VALUE) / N_spins
    if original_density>=higher_density:
        return matrix
    spins_to_add = int(N_spins * (higher_density - original_density))
    majority_indexes = np.where(matrix == constants.MAJORITY_VALUE)
    change_indexes = np.random.choice(np.array(majority_indexes[0].shape[0], dtype=int), spins_to_add, replace=False)
    matrix[
        tuple((dimension_indexes[change_indexes] for dimension_indexes in majority_indexes))] = constants.MINORITY_VALUE
    return matrix


@jit()
def calculate_energy_diff(matrix, J, point_A, point_B):
    """
    :param matrix:
    :param point_A:
    :param point_B:
    :return: energy difference from swapping the two points
    """

    before = calc_energy_point(matrix, J, point_A) + calc_energy_point(matrix, J, point_B)
    matrix = swap_points(matrix, point_A, point_B)
    after = calc_energy_point(matrix, J, point_A) + calc_energy_point(matrix, J, point_B)
    matrix = swap_points(matrix, point_A, point_B)  # needed!!!
    return after - before


@jit()
def swap_points(matrix, point_A, point_B):
    i, j = point_A
    neighbor_i, neighbor_j = point_B
    matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
        i, j]
    return matrix


@jit()
def calc_energy_point(matrix, J, point):
    i, j = point
    N = matrix.shape[0]
    return -matrix[i, j] * (
            matrix[(i + 1) % N, j] * J[constants.VERTICAL, i, j]
            + matrix[(i - 1) % N, j] * J[constants.VERTICAL, (i - 1) % N, j]
            + matrix[i, (j + 1) % N] * J[constants.HORIZONTAL, i, j]
            + matrix[i, (j - 1) % N] * J[constants.HORIZONTAL, i, (j - 1) % N])


@jit()
def calculate_energy(matrix, J):
    """
    calculate the energy of the system
    :param matrix:
    :return:
    """
    energy = 0
    N = matrix.shape[0]
    for i in range(N):
        for j in range(N):
            energy -= matrix[i, j] * (
                    matrix[(i + 1) % N, j] * J[constants.VERTICAL, i, j]
                    + matrix[(i - 1) % N, j] * J[constants.VERTICAL, (i - 1) % N, j]
                    + matrix[i, (j + 1) % N] * J[constants.HORIZONTAL, i, j]
                    + matrix[i, (j - 1) % N] * J[constants.HORIZONTAL, i, (j - 1) % N])
    return energy / 2
