import numpy as np

import constants


def clusters_from_original_matrix(matrix, cluster_value):
    """
    creating a new cluster matrix from the original matrix, each cluster of
    cluster value will be associated with different >0 integer and the shape of
    the cluster will be the same as in the original matrix.
    :param matrix: spin lattice to be clustered
    :param cluster_value: the value we want to cluster
    :return: the cluster matrix
    """
    N_rows, N_cols = matrix.shape
    cluster_matrix = np.zeros((N_rows, N_cols))
    cluster_matrix[np.where(matrix == cluster_value)] = -1
    max_cluster_label = 1
    last_value_in_cluster = False
    for i in range(N_rows):
        for j in range(N_cols):
            if cluster_matrix[i, j] == 0:
                if last_value_in_cluster:
                    max_cluster_label += 1
                last_value_in_cluster = False
            elif cluster_matrix[i, j] == -1:
                cluster_matrix = check_for_clusters_connection(cluster_matrix, i, j, max_cluster_label)
                last_value_in_cluster = True
    return cluster_matrix


def check_for_clusters_connection(matrix, row, col, new_cluster_label):
    """
    checks if the site in cluster_matrix[row, col] connects 2 different cluster, and
    unite them if that happens
    :param matrix: the cluster matrix we work on
    :param row: the row of the site
    :param col: the column of the site
    :param new_cluster_label: the value we want to associate to the spin cluster
    :return: the cluster matrix after the needed changes (changing to the site
    and needed clusters if needed)
    """
    N_rows, N_cols = matrix.shape
    previous_row_value = matrix[(row - 1) % N_rows, col]
    previous_col_value = matrix[row, (col - 1) % N_cols]
    matrix[row, col] = new_cluster_label
    if previous_row_value == previous_col_value == 0:
        return matrix
    if previous_row_value > 0:
        matrix[row, col] = previous_row_value
    elif previous_row_value == -1:
        matrix[(row - 1) % N_rows, col] = previous_row_value = matrix[row, col]
    if previous_col_value > 0:
        matrix[row, col] = previous_col_value
    elif previous_col_value == -1:
        matrix[row, (col - 1) % N_cols] = previous_col_value = matrix[row, col]

    if previous_col_value != 0 and previous_row_value != 0 and previous_row_value != previous_col_value:
        value = min((previous_col_value, previous_row_value, new_cluster_label))
        matrix[np.where(np.isin(matrix, (previous_col_value, previous_row_value, new_cluster_label)))] = value
    return matrix


def get_clusters_number(matrix, cluster_value, min_cluster_size):
    """
    Returns the number of different clusters of size>=min cluster size in matrix
    :param matrix: the matrix to be clustered
    :param cluster_value: the value that is in the clusters
    :param min_cluster_size: minimum number of sites participating in each
     cluster in order to count this as a cluster
    :return: the number of clusters in matrix
    """
    cluster_matrix = clusters_from_original_matrix(matrix, cluster_value)
    values, sizes = np.unique(cluster_matrix, return_counts=True)
    clusters_num = len(values[sizes >= min_cluster_size]) - 1
    return clusters_num


def spin_counts_per_cluster(matrix, cluster_value=constants.MINORITY_VALUE):
    """
    Returns the sizes of all the clusters in matrix.
    :param matrix: matrix to cluster.
    :param cluster_value: the value that is in the clusters.
    :return: np.ndarray of all the clusters sizes in the matrix, ordered from
    largest to smallest.
    """
    cluster_matrix = clusters_from_original_matrix(matrix, cluster_value)
    value, sizes = np.unique(cluster_matrix, return_counts=True)
    return np.sort(sizes[value > 0])[::-1]
