import numpy as np
def initialize_matrix(N,density):
	return np.random.choice([0, 1], size=(N, N), p=[density, 1 - density])


