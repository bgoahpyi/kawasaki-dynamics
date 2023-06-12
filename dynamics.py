import numpy as np
import constants
from numba import jit

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
	time = 0
	while True:  # continue until a step is made
		#timeout logic for not finding any swap
		time += 1
		if time >= constants.TIMEOUT:
			print("timeout")
			return matrix

		i, j, neighbor_i, neighbor_j = get_site_and_neighbor(N)
		if (matrix[i, j] == matrix[neighbor_i, neighbor_j]):
			continue

		#calculate E, swap and calculate E'
		E = calculate_energy(matrix, J)
		matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
			i, j]
		E_prime = calculate_energy(matrix, J)

		delta_E = E_prime - E
		#might need to change probability calculation
		if delta_E < 0:
			# matrix = swap_matrix
			break
		else:
			probability = np.exp(-delta_E / temperature)
			if np.random.rand() < probability:
				# matrix = swap_matrix
				break
			matrix[i, j], matrix[neighbor_i, neighbor_j] = matrix[neighbor_i, neighbor_j], matrix[
				i, j]
			continue
	return matrix


def get_site_and_neighbor(N):
	i, j = np.random.randint(0, N, 2) #todo: change to choice for ==1
	neighbor = constants.NEIGHBOR[
		np.random.randint(0, 4)]  # problem with probability of picking an edge - where do you account for that?
	neighbor_i = (i + neighbor[0]) % N  # currently we do periodic edges
	neighbor_j = (j + neighbor[1]) % N
	return i, j, neighbor_i, neighbor_j


def increment_density():
	"""
	how to update the density? pick a 0 site at random and flip it to 1?
	:return:
	"""
	pass



@jit()
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

