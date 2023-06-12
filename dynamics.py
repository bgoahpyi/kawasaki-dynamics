import numpy as np
import constants

def kawasaki_time_step(matrix, J, temperature):
	"""
	preforms one kawasaki time step:
		pick random particle A
		pick random neighbor B
		check if they are the same:
			continue
		else
			calculate current system energy E and energy after the flip E'
			calculate the probabilities(boltzmann)
			flip with probability p'/(p+p')


	:param matrix:
	:param temperature:
	:return: updated matrix
	"""
	N = matrix.shape[0]
	while True:  # continue until a step is made
		i, j = np.random.randint(0, N, 2)
		neighbor = constants.NEIGHBOR[
			np.random.randint(0, 4)]  # problem with probability of picking an edge - where do you account for that?
		neighbor_i = (i + neighbor[0]) % N  # currently we do periodic edges
		neighbor_j = (j + neighbor[1]) % N
		if (matrix[i, j] == matrix[neighbor_i, neighbor_j]):
			continue
		#calculate E, swap and calculate E'
		E = calculate_energy(matrix, J)
		swap_matrix = np.copy(matrix)
		swap_matrix[i, j], swap_matrix[neighbor_i, neighbor_j] = swap_matrix[neighbor_i, neighbor_j], swap_matrix[
			i, j]
		E_prime = calculate_energy(swap_matrix, J)

		#might need to change probability calculation
		p = np.exp(-E/temperature)
		p_prime = np.exp(-E_prime/temperature)
		if np.random.rand() < p_prime/(p+p_prime):
			matrix = swap_matrix
			break
		continue
	return matrix
def increment_density():
	"""
	how to update the density? pick a 0 site at random and flip it to 1?
	:return:
	"""
	pass




def calculate_energy(matrix, J):
	"""
	calculate the energy of the system
	:param matrix:
	:return:
	"""
	energy = 0
	N = matrix.shape[0]
	for i in range(N):
		for j in range(N):
			energy += matrix[i, j] * (
					matrix[(i + 1) % N, j] * J[constants.VERTICAL, i, j]
					+ matrix[(i - 1) % N, j] * J[constants.VERTICAL, (i - 1) % N, j]
					+ matrix[i, (j + 1) % N] * J[constants.HORIZONTAL, i, j]
					+ matrix[i, (j - 1) % N] * J[constants.HORIZONTAL, i, (j - 1) % N])
	return energy / 2

