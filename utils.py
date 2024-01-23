import numpy as np

import constants


def pack_grid(grid: np.ndarray):
    """
    Pack binary (two valued) matrix to bytes
    :param grid: matrix to pack
    :return: packed matrix
    """
    return np.packbits(grid == constants.MINORITY_VALUE)


def save_grid(packed_grid: np.ndarray, path):
    """
    Saves packed matrix to file.
    :param packed_grid: bytes matrix to save
    :param path: file path to save into
    """
    with open(path, "wb") as f:
        f.write(bytes(packed_grid))


def read_min_grid(path, matrix_side):
    """
    Read binary matrix from file
    :param path: path to read from
    :param matrix_side: matrix side for the shape of the matrix.
    :return: binary matrix
    """
    with open(path, "rb") as f:
        lines = f.read()
    byte_arr = np.frombuffer(lines, dtype=np.uint8)
    read_arr = np.unpackbits(byte_arr).reshape((-1, matrix_side, matrix_side)).astype(bool)
    return read_arr * constants.MINORITY_VALUE + (~read_arr) * constants.MAJORITY_VALUE
