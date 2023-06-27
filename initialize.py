import numpy as np

import constants


def initialize_matrix(N,density):
	matrix = np.random.choice([constants.MAJORITY_VALUE, constants.MINORITY_VALUE], size=(N, N), p=[1-density, density])
	if np.count_nonzero(matrix==constants.MINORITY_VALUE)==0:
		matrix[0,0]=constants.MINORITY_VALUE
	return matrix

