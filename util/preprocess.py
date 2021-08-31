import os
from os.path import dirname, exists, join, abspath
import numpy as np
import pandas as pd
from scipy.io import loadmat
from jacob_temp_code.helper_functions import IUPAC_SEQ2IDX

def preprocess_observations(y_wildtype, y_wetlab, y_scaled):
    y = np.vstack([y_wildtype, y_wetlab, y_scaled])
    mean_y = np.mean(y)
    y -= mean_y
    max_y = np.max(np.abs(y))
    y /= max_y
    return mean_y, max_y, y[[0], :], y[1:y_wetlab.shape[0] + 1, :], y[1 + y_wetlab.shape[0]:, :]


def correct_sub_matrices(subsitution_matfile_abspath: str="C:\protein_regression\data\subMats.mat") -> None:
    mgp_alphabet = list("ARNDCQEGHILKMFPSTWYV") # See alphabet mgpfusion/letters2AA.m
    # reference matlab files that need to be sorted in accordance with the general alphabet
    data_dir = dirname(subsitution_matfile_abspath)
    print(data_dir)
    matrices = loadmat(subsitution_matfile_abspath).get("subMats")
    for sub_matrix, matrix_id, _ in matrices:
        df_mat = pd.DataFrame(data=sub_matrix, columns=mgp_alphabet, index=mgp_alphabet)
        # X is in label-encoding but NOT contained in mgpfusion matrices - correct:
        df_mat = df_mat.append(pd.Series(name='X'))
        df_mat["X"] = pd.Series()
        df_mat = df_mat.sort_index().sort_index(axis=1)
        encoding_array = np.array([k for k in IUPAC_SEQ2IDX.keys() if k in df_mat.index])
        np.testing.assert_array_equal(df_mat.index, encoding_array)
        mat_filename = "{}/S_MATRIX_{}_INORDER.csv".format(data_dir, matrix_id[0])
        if not exists(mat_filename):
            df_mat.to_csv(mat_filename)
        else:
            print("{} exists...".format(matrix_id[0]))