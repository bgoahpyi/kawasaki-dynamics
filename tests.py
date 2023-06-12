import main
import constants
import initialize
import numpy as np
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
	print(np.count_nonzero(initial_state==0))
	assert abs((np.count_nonzero(initial_state==0) / initial_state.size) - density) < 0.05


if __name__ == '__main__':
	initial_density_check()
	print("initial_density_check passed")