import numpy as np

import constants


def initialize_matrix(N, density):
    matrix = np.ones((N, N), dtype=type(constants.MAJORITY_VALUE))
    matrix[:, :] = constants.MAJORITY_VALUE
    minority_num = max((int(density * N ** 2), 1))
    majority_places = np.where(matrix == constants.MAJORITY_VALUE)
    inds = np.random.choice(np.arange(0, len(majority_places[0]), dtype=int), size=minority_num, replace=False)
    matrix[majority_places[0][inds], majority_places[1][inds]] = constants.MINORITY_VALUE
    return matrix


def create_neighbors_matrix(matrix: np.ndarray):
    """

    :param matrix: spin cluster_matrix
    :return: neighbors cluster_matrix: True for different neighbor, False otherwise
    """
    neighbors = np.zeros(matrix.shape + (len(constants.NEIGHBOR),), dtype=bool)
    for i, neighbor in enumerate(constants.NEIGHBOR):
        axis = 0
        for j in neighbor:
            if j != 0:
                break
            axis += 1
        neighbors[:, :, i] = np.roll(matrix, -neighbor[axis], axis=axis) != matrix
    return neighbors
