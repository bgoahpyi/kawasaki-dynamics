import constants
import initialize
import numpy as np

if __name__ == '__main__':
	N = constants.MATRIX_SIZE
	density = constants.INITIAL_ZERO_DENSITY
	initial_state = initialize.initialize_matrix(N,density)
	print()