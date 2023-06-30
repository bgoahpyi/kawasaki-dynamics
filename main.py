import os.path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import clusters
import constants
import dynamics
import initialize


def run_with_densities(N: int = constants.MATRIX_SIZE, densities = [constants.INITIAL_ZERO_DENSITY],
                       T: float = constants.INITIAL_TEMPERATURE, steps: int = constants.NUMBER_OF_STEPS,
                       gif_name: str = "visual/vis.gif"):
    matrix = initialize.initialize_matrix(N, densities[0])
    neighbors_matrix = initialize.create_neighbors_matrix(matrix)
    different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    densities_number = len(densities)
    steps_per_density = steps // densities_number
    # energy = []
    # imgs = []

    for i in tqdm(range(steps_per_density * (densities_number) - 1)):
        matrix, neighbors_matrix, different_neighbor_numbers = dynamics.kawasaki_time_step(matrix, T, neighbors_matrix,
                                                                                           different_neighbor_numbers)
        # if (i * constants.NUMBER_OF_FRAMES) % steps == 0:
        #     energy.append(dynamics.calculate_energy(matrix, different_neighbor_numbers))
        #     imgs.append(Image.fromarray((matrix == constants.MINORITY_VALUE) * 255))
        if i % steps_per_density == 0:
            if i == 0:
                continue
            matrix, neighbors_matrix = dynamics.increment_density(matrix, densities[i // steps_per_density])
            different_neighbor_numbers = np.sum(neighbors_matrix, axis=len(neighbors_matrix.shape) - 1)
    matrix, neighbors_matrix, different_neighbor_numbers = dynamics.kawasaki_time_step(matrix, T, neighbors_matrix,
                                                                                       different_neighbor_numbers)
    # energy.append(dynamics.calculate_energy(matrix, different_neighbor_numbers))
    # imgs.append(Image.fromarray((matrix == constants.MINORITY_VALUE) * 255))
    # imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=1, loop=1)  #
    return matrix





def check_values(initial_den, final_den, total_change_steps, steps_at_final_den, times_to_repeat):
    dens = np.concatenate([np.linspace(initial_den, final_den, total_change_steps), [final_den] * steps_at_final_den])
    clusters_num = []
    for i in range(times_to_repeat):
        matrix = run_with_densities(T=0.7, densities=dens, gif_name="visual/densities.gif")
        clusters_num.append(clusters.get_clusters_number(matrix, constants.MINORITY_VALUE, 8))
    return np.mean(clusters_num), np.std(clusters_num)


def main(final_den, rep_num):
    mean_path=f"npy_files/mean_clusters_num_{final_den}_{constants.NUMBER_OF_STEPS}_steps rep_num_{rep_num}"
    std_path=f"npy_files/clusters_std_map_{final_den}_{constants.NUMBER_OF_STEPS}_steps rep_num_{rep_num}"
    initial_den=3/64**2
    changes_steps_nums_list=[int(((64**2-3)/4)//i) for i in range(1,10)] + list(np.linspace(65, 0, 5, dtype=int))
    steps_at_final_density= [1,2,5,10,15,20,25,30]
    if os.path.exists(mean_path):
        mean_cluster_num_map=np.load(mean_path)
    else:
        mean_cluster_num_map=np.zeros((len(steps_at_final_density),len(changes_steps_nums_list)))
    if os.path.exists(std_path):
        cluster_std_map=np.load(std_path)
    else:
        cluster_std_map=np.zeros((len(steps_at_final_density),len(changes_steps_nums_list)))
    try:
        for i,steps_at_final_den in enumerate(steps_at_final_density):
            for j, total_changes_steps_num in enumerate(changes_steps_nums_list):
                if mean_cluster_num_map[i,j]==0:
                    print("i, j=",i,j)
                    mean_cluster_num_map[i, j],cluster_std_map[i,j] = check_values(initial_den, final_den,
                                                                               total_changes_steps_num,
                                                                               steps_at_final_den, rep_num)
    finally:
        np.save(mean_path, mean_cluster_num_map)
        np.save(std_path, cluster_std_map)
        print(f"{final_den} density heat map:")
        print(mean_cluster_num_map)
        print("std map:")
        print(cluster_std_map, end="\n\n")



if __name__ == '__main__':
    np.random.seed(1)
    main(0.1,5)
    main(0.25,5)
    # run_with_densities(N=constants.MATRIX_SIZE)
# main(T=0.1)
# for t in np.linspace(1e-3, 2,15):
# 	main(T=t, gif_name=f"visual/T={round(t,3)}.gif")
