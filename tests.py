import dynamics
import main
import constants
import initialize
import numpy as np
import time
def initial_density_check():
	"""
	check that the initial density is close to the desired density
	"""
	N = constants.MATRIX_SIZE
	density = constants.INITIAL_ZERO_DENSITY
	initial_state = initialize.initialize_matrix(N,density)
	# print(initial_state)
	# print(initial_state.shape)
	# print(initial_state.size)
	# print(np.count_nonzero(initial_state==0))
	assert abs((np.count_nonzero(initial_state==0) / initial_state.size) - density) < 0.05


def time_benchmark():
	"""
	check that the time for the simulation is reasonable
	"""
	print("starting time benchmark")
	print("cluster_matrix size is", constants.MATRIX_SIZE)
	print("number of steps is", constants.NUMBER_OF_STEPS)
	start = time.time()
	main.main()
	end = time.time()
	print(end-start, "seconds")


def zero_density_check():
	"""
	check that the density is zero at the end of the simulation

	"""
	N = constants.MATRIX_SIZE
	density = 0
	initial_state = initialize.initialize_matrix(N,density)
	assert np.count_nonzero(initial_state==0) == 0
	final_state = main.main(N=constants.MATRIX_SIZE, density=0, T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS)
	assert np.count_nonzero(final_state==0) == 0


def calculate_energy_test():
	state=initialize.initialize_matrix(constants.MATRIX_SIZE, constants.INITIAL_ZERO_DENSITY)
	assert dynamics.calculate_energy(
		state, np.sum(initialize.create_neighbors_matrix(state), len(state.shape))
	)==-constants.CONSTANT_J*np.sum(state*(np.roll(state, 1, axis=0)+np.roll(state, 1, axis=1)))

if __name__ == '__main__':
	# initial_density_check()
	# zero_density_check()
	time_benchmark()
	calculate_energy_test()
	# print("initial_density_check passed")

