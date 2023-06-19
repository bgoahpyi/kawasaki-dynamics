import constants
import initialize
import dynamics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def run_with_densities(N=constants.MATRIX_SIZE, densities=[constants.INITIAL_ZERO_DENSITY], T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS, gif_name="visual/vis.gif"):
	matrix = initialize.initialize_matrix(N, densities[0])
	densities_number=len(densities)
	steps_per_density=steps//densities_number
	energy = []
	imgs = []

	for i in tqdm(range(steps_per_density*densities_number-1)):
		matrix = dynamics.kawasaki_time_step(matrix, J, T)
		if (i*constants.NUMBER_OF_FRAMES) % steps == 0:
			energy.append(dynamics.calculate_energy(matrix, J))
			imgs.append(Image.fromarray((matrix==constants.MINORITY_VALUE) * 127.5))
		if i%steps_per_density == 0:
			if i==0:
				continue
			matrix = dynamics.increment_density(matrix, densities[i//steps_per_density])

	# print(int(constants.GIF_LENGTH*10**3/len(imgs)))
	# print(len(imgs))
	imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=1,loop=1) #

	plt.plot(energy)
	plt.show()

def main(N=constants.MATRIX_SIZE, density=constants.INITIAL_ZERO_DENSITY, T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS, gif_name="visual/vis.gif"):
	matrix = initialize.initialize_matrix(N, density)

	energy = []
	imgs = []
	for i in tqdm(range(steps)):
		matrix = dynamics.kawasaki_time_step(matrix, J, T)
		if (i*constants.NUMBER_OF_FRAMES) % steps == 0:
			energy.append(dynamics.calculate_energy(matrix, J))
			imgs.append(Image.fromarray((matrix + 1) * 127.5))

	# print(int(constants.GIF_LENGTH*10**3/len(imgs)))
	# print(len(imgs))
	imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=1,loop=1) #

	plt.plot(energy)
	plt.show()

if __name__ == '__main__':
	run_with_densities(T=0.7,densities=np.concatenate([np.arange(2,64,1)/64**2, np.ones(5)/64,np.linspace(3/128, 0.5, 15), 0.5*np.ones(3)]), gif_name="visual/densities.gif")
	# main(T=0.1)
	# for t in np.linspace(1e-3, 2,15):
	# 	main(T=t, gif_name=f"visual/T={round(t,3)}.gif")
