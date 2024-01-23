import numpy as np

# constants file for our simulation

MATRIX_SIDE_LENGTH = 64
DIMENSION = 2
N_SPINS = MATRIX_SIDE_LENGTH ** DIMENSION
INITIAL_ZERO_DENSITY = 0.5
NUMBER_OF_STEPS = int(1e6)
NUMBER_OF_FRAMES = 300
MAJORITY_VALUE = -1
MINORITY_VALUE = 1

MIN_CLUSTER_SIZE=8

INITIAL_TEMPERATURE = 0.1  # how do we know what is the right temperature?
TIMEOUT = int(1e3)
CONSTANT_J = 1

# the neighbors cluster_matrix works like this cluster_matrix, aka: neighbor_mat[i,j]=[RIGHT_is_different (bool), LEFT_is_different...]
# don't change the order of the neighbors, it needs to be +-1 on some axis, one after the other
NEIGHBOR = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

VERTICAL = 0
HORIZONTAL = 1

MATRIX_EXTENSION=""
WITHOUT_CHANGE_DEPENDENT_SCHEDULE = ", without changes dependent schedule"
PLOTLY_TEMPLATE="plotly_white"



