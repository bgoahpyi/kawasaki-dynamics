import numpy as np

import constants


def initialize_matrix(N,density):
	return np.random.choice([constants.MAJORITY_VALUE, constants.MINORITY_VALUE], size=(N, N), p=[1-density, density])

