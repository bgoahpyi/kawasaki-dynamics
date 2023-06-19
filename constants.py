import numpy as np

# constants file for our simulation

MATRIX_SIZE = 64
INITIAL_ZERO_DENSITY = 0.5
NUMBER_OF_STEPS = int(1e4)
NUMBER_OF_FRAMES = 100

INITIAL_TEMPERATURE = 0.1 #how do we know what is the right temperature?
TIMEOUT = 1e4
CONSTANT_J = -np.ones((2, MATRIX_SIZE, MATRIX_SIZE))

NEIGHBOR = [[0, 1], [0, -1], [1, 0], [-1, 0]]

VERTICAL = 0
HORIZONTAL = 1


