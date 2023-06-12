import numpy as np

# constants file for our simulation

MATRIX_SIZE = 64
INITIAL_ZERO_DENSITY = 0.5
NUMBER_OF_STEPS = 1e6
INITIAL_TEMPERATURE = 1

CONSTANT_J = np.ones((2, MATRIX_SIZE, MATRIX_SIZE))

NEIGHBOR = [[0, 1], [0, -1], [1, 0], [-1, 0]]

VERTICAL = 0
HORIZONTAL = 1