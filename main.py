import constants
import initialize
import dynamics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm




def main(N=constants.MATRIX_SIZE, density=constants.INITIAL_ZERO_DENSITY, T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS):
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
	imgs[0].save("visual/vis.gif", save_all=True, append_images=imgs[1:], duration=1,loop=1) #

	plt.plot(energy)
	plt.show()

if __name__ == '__main__':
	main()
