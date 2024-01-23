import numpy as np

import constants


def initialize_random_binary_matrix(N, density):
    """
    initializing random 2d binary matrix with values from {constant.MAJORITY_VALUE, constant.MINORITY_VALUE
    :param N: length of each dimension
    :param density: what part of the matrix should have the MINORITY value
    :return: random binary (MAJORITY and MINORITY) matrix of shape (N,N)
    """
    matrix = np.ones((N, N), dtype=type(constants.MAJORITY_VALUE))
    matrix[:, :] = constants.MAJORITY_VALUE
    minority_num = max((int(density * N ** 2), 1))
    majority_places = np.where(matrix == constants.MAJORITY_VALUE)
    inds = np.random.choice(np.arange(0, len(majority_places[0]), dtype=int), size=minority_num, replace=False)
    matrix[majority_places[0][inds], majority_places[1][inds]] = constants.MINORITY_VALUE
    return matrix


def create_neighbors_matrix(matrix: np.ndarray):
    """
    return neighbors matrix, in the last dimension there is a list of booleans 
    indicating if the neighbor at this direction
    (according to constants.NEIGHBOR) is with different value.
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
