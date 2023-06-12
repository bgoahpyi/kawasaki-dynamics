import constants
import initialize
import dynamics
import numpy as np
import matplotlib.pyplot as plt


def main(N=constants.MATRIX_SIZE, density=constants.INITIAL_ZERO_DENSITY, T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS):
	matrix = initialize.initialize_matrix(N, density)

	energy = []
	for i in range(steps):
		matrix = dynamics.kawasaki_time_step(matrix, J, T)
		energy.append(dynamics.calculate_energy(matrix, J))
	plt.plot(energy)
	plt.show()

if __name__ == '__main__':
	main()
