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
	# imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=1,loop=1) #

	# plt.plot(energy)
	# plt.show()
	return energy[-1]

def main(N=constants.MATRIX_SIZE, density=constants.INITIAL_ZERO_DENSITY, T=constants.INITIAL_TEMPERATURE,
		 J=constants.CONSTANT_J, steps=constants.NUMBER_OF_STEPS, gif_name="visual/vis.gif"):
	matrix = initialize.initialize_matrix(N, density)

	energy = []
	imgs = []
	for i in tqdm(range(steps)):
		matrix = dynamics.kawasaki_time_step(matrix, J, T)
		if (i*constants.NUMBER_OF_FRAMES) % steps == 0:
			energy.append(dynamics.calculate_energy(matrix, J))
			# imgs.append(Image.fromarray((matrix + 1) * 127.5))

	# print(int(constants.GIF_LENGTH*10**3/len(imgs)))
	# print(len(imgs))
	# imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=1,loop=1) #

	plt.plot(energy)
	plt.show()

def compare():
	"""
	T=0.7, 1e4 steps.
	"""
	#only 0.5 density
	constant_density_energies = [-4184.0, -4136.0, -4228.0, -4096.0, -4112.0, -4116.0, -4124.0, -4208.0, -4076.0, -4092.0]
	print(np.mean(constant_density_energies))
	# 0.1 ->0.5 in 10 steps, 2 steps that are 0.5
	linear_density_01_10 = [-3720.0, -3840.0, -3808.0, -3700.0, -3800.0, -3744.0, -3808.0, -3704.0, -3660.0, -3756.0]
	print(np.mean(linear_density_01_10))
	# 0.1 ->0.5 in 10 steps, 10 steps that are 0.5
	linear_density_01_10_2 = [-4168.0, -3996.0, -3956.0, -3940.0, -3964.0, -4092.0, -3936.0, -3972.0, -4000.0, -3984.0]
	print(np.mean(linear_density_01_10_2))
	# 0.1 ->0.5 in 10 steps, 5 steps that are 0.5
	linear_density_01_10_3 = [-3968.0, -3860.0, -3940.0, -3952.0, -3828.0, -3948.0, -3980.0, -3932.0, -3912.0, -3904.0]
	print(np.mean(linear_density_01_10_3))
	#0.1 ->0.5 in 10 steps, 20 step that are 0.5
	linear_density_01_10_4 = [-4000.0, -4000.0, -4024.0, -4100.0, -4052.0, -3912.0, -4112.0, -4072.0, -4048.0, -4160.0]
	print(np.mean(linear_density_01_10_4))

def compare2():
	"""
	1e5 steps
	T = 0.7
	max_density = 0.5
	"""

	constant_density_energies=[-5056.0, -4812.0, -5000.0, -4932.0, -4932.0]
	print(np.mean(constant_density_energies))
	#0.01->0.5 *10, 0.5
	lin_den_1 = [-4800.0, -4716.0, -4820.0, -4788.0, -4816.0]
	print(np.mean(lin_den_1))
	#0.1->0.5 * 10, 0.5
	lin_den_2 = [-4912.0, -4820.0, -4816.0, -4904.0, -4852.0]
	print(np.mean(lin_den_2))
	#0.05->0.5 * 20, 0.5
	lin_den_3 = [-4872.0, -4768.0, -4924.0, -4948.0, -4768.0]

def compare3():
	"""
		1e5 steps
		T = 0.7
		max_density = 0.25
	"""
	constant_density = [-5528.0, -5448.0, -5424.0, -5472.0, -5352.0]
	print(np.mean(constant_density))
	#0.05->0.25 * 10
	lin_den_1 = [-5264.0, -5360.0, -5312.0, -5324.0, -5452.0]
	print(np.mean(lin_den_1))
	#0.05->0.25 * 10, 0.25
	lin_den_2 = [-5408.0, -5396.0, -5332.0, -5436.0, -5428.0]
	print(np.mean(lin_den_2))
	#0.05->0.25 * 15, 0.25
	lin_den_3 = [-5488.0, -5484.0, -5464.0, -5348.0, -5368.0]
	print(np.mean(lin_den_3))

	# 0.05->0.25 * 5
	lin_den_4 = [-5300.0, -5360.0, -5308.0, -5248.0, -5292.0]
	print(np.mean(lin_den_4))

	# 0.05->0.25 * 15
	lin_den_5 = [-5300.0, -5440.0, -5528.0, -5396.0, -5404.0]
	print(np.mean(lin_den_5))

	# 0.05->0.25 * 20
	lin_den_6 = [-5380.0, -5436.0, -5380.0, -5340.0, -5480.0]
	print(np.mean(lin_den_6))

	# 0.05->0.25 * 15, 0.25 * 5
	lin_den_7 = [-5412.0, -5548.0, -5528.0, -5528.0, -5464.0]
	print(np.mean(lin_den_7))

	# 0.05->0.25 * 15, 0.25 * 15
	lin_den_8 = [-5476.0, -5532.0, -5444.0, -5444.0, -5472.0]
	print(np.mean(lin_den_8))

def check_values(initial_den,final_den,steps_between,steps_at_final_den, times_to_repeat):
	dens = np.concatenate([np.linspace(initial_den, final_den, steps_between), [final_den]*steps_at_final_den])
	final_energies = []
	for i in range(times_to_repeat):
		energy = run_with_densities(T=0.7,densities=dens, gif_name="visual/densities.gif")
		final_energies.append(energy)
	return np.mean(final_energies)



if __name__ == '__main__':
	step_between_array = [0,5,10,15,20,25,30,35,40,45,50]
	step_at_final_den_array = [1,2,5,10,15,20,25,30,35,40,45,50]
	heat_map = np.zeros((len(step_between_array), len(step_at_final_den_array)))
	for steps_between in range(len(step_between_array)):
		for steps_at_final_den in range(len(step_at_final_den_array)):
			# heat_map[steps_between, steps_at_final_den] = np.random.randint(0,100)
			heat_map[steps_between, steps_at_final_den] = check_values(3/(64**2),0.1,step_between_array[steps_between],step_at_final_den_array[steps_at_final_den], 10)
	print(heat_map)
	np.savetxt("heat_map_0.1_5e5_steps.csv", heat_map, delimiter=",")
	plt.imshow(heat_map, cmap='hot', interpolation='nearest')
	plt.xlabel("steps at final density")
	plt.ylabel("steps between densities")
	plt.xticks(range(len(step_at_final_den_array)), step_at_final_den_array)
	plt.yticks(range(len(step_between_array)), step_between_array)
	plt.show()


