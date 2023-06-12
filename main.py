import constants
import initialize
import dynamics
import numpy as np
import matplotlib.pyplot as plt


def main(N=constants.MATRIX_SIZE, density=constants.INITIAL_ZERO_DENSITY, T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS):
	matrix = initialize.initialize_matrix(N, density)


	for i in range(steps):
		matrix = dynamics.kawasaki_time_step(matrix, J, T)


if __name__ == '__main__':
	main()
