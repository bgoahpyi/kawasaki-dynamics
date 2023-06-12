import constants
import initialize
import dynamics
import numpy as np

if __name__ == '__main__':
	N = constants.MATRIX_SIZE
	density = constants.INITIAL_ZERO_DENSITY
	temperature = constants.INITIAL_TEMPERATURE  # currently constant. can be changed
	matrix = initialize.initialize_matrix(N, density)

	for i in range(constants.NUMBER_OF_STEPS):
		matrix = dynamics.kawasaki_time_step(matrix,
											 temperature)  # bad practice copying the matrix every time. can be changed.
