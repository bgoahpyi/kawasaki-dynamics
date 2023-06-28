import numpy as np


def clusters_from_original_matrix(matrix, cluster_value):
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
                cluster_matrix= check_for_clusters_connection(cluster_matrix, i, j, max_cluster_label)
                last_value_in_cluster=True
    return cluster_matrix
            # elif cluster_matrix[i,j]==-1:
            #     if not last_value_in_cluster:
            #         if cluster_matrix[(i-1)%N_rows, j]!=0
            #         cluster_matrix[i,j]=new_cluster_label
            #
            #     elif cluster_matrix[(i-1)%N_rows, j]!=0 and cluster_matrix[i, (j-1)%N_cols]!=0:
            #         cluster_value=min((cluster_matrix[(i-1)%N_rows, j],cluster_matrix[i, (j-1)%N_cols]))
            #
            #         if cluster_matrix[(i-1)%N_rows, j]!=0:
            #             cluster_matrix[i,j]=cluster_matrix[(i-1)%N_rows, j]


def check_for_clusters_connection(matrix, row, col, new_cluster_label):
    """
    checks if the value in cluster_matrix[row, col] connects 2 different cluster, and
    unite them if that happens
    :param matrix:
    :param row:
    :param col:
    :param new_cluster_label:
    :return:
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

    if previous_col_value != 0 and previous_row_value != 0 and previous_row_value!=previous_col_value:
        value = min((previous_col_value, previous_row_value, new_cluster_label))
        matrix[np.where(np.isin(matrix, (previous_col_value, previous_row_value, new_cluster_label)))] = value
    return matrix


def get_cluster_number(matrix, cluster_value, min_cluster_size):
    cluster_matrix=clusters_from_original_matrix(matrix, cluster_value)
    values, sizes=np.unique(cluster_matrix, return_counts=True)
    clusters_num=len(values[sizes>=min_cluster_size])-1
    return clusters_num