import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from typing import List
from tqdm import tqdm


class KernelLoader:
    def __init__(self, sub_mat_ids: list = []):
        """
        Interface to MatrixKernel that encapsulates used substitution matrices. 
        Has list of kernels as class property.
        sub_mat_ids takes IDs from SubMat Matlab
        """
        s_mat, s_mat_id = self.load_sub_matrices(sub_mat_ids)
        self.kernels: list = [MatrixKernel(matrix=s, matrix_id=m_id) for s, m_id in zip(s_mat, s_mat_id)]
        self.sub_matrices_ids = s_mat_id
        assert len(self.kernels) == len(self.sub_matrices_ids)

    def load_sub_matrices(self, sub_mat_ids, matrix_dir=None):
        # load corrected substitution matrices from pandas CSVs
        matrix_dir = "C:\protein_regression\data\sub_matrices"
        s_mat = []
        s_mat_id = []
        # check for provided sub_matrices in data subMat
        for matrix_csv in os.listdir(matrix_dir):
            mat_id = matrix_csv.split("_")[2]
            if sub_mat_ids:
                if mat_id in sub_mat_ids:
                    s_mat_id.append(mat_id)
                    # eliminate indices from DF -> [:, 1:]
                    s_mat.append(pd.read_csv(os.path.join(matrix_dir, matrix_csv)).to_numpy()[:, 1:])
                else:
                    continue
            else:
                s_mat_id.append(mat_id)
                s_mat.append(pd.read_csv(os.path.join(matrix_dir, matrix_csv)).to_numpy()[:, 1:])
        return s_mat, s_mat_id


class MatrixKernel:
    def __init__(self, matrix: np.array, matrix_id: str, depth: int = 1):
        """
        Matrix Kernel class takes substitution matrix with which to compute the kernel.
        Takes sequences and list of adjacencies over which to compute the kernel value.
        """
        self.depth: int = depth  # not used downstream
        self.matrix = matrix
        self.matrix_id = matrix_id

    def k(self, x_p: np.ndarray, adjacencies: List[tuple]) -> np.ndarray:
        """
        Eq. 7
        Compute kernel value w.r.t. neighborhood normalized.
        N = num of mutational variants
        D = sequence length
        Input: sequences: NxD,
            adjacencies: NxD as list of tuples with residues and AA neighbors as integers

        return NxN Matrix
        """
        N = x_p.shape[0]
        k = np.zeros([N, N])
        temp_k = np.zeros([N, N])
        neighborhoods = adjacencies
        if isinstance(adjacencies[0], tuple):
            neighborhoods = np.array([contacts for res, contacts in adjacencies])
        neighborhood_iterator = tqdm(enumerate(neighborhoods))
        for idx, neighbors in neighborhood_iterator:
            neighborhood_iterator.set_description(f"Matrix: {self.matrix_id}")
            temp_k.fill(0.)
            for n in neighbors:
                # WARN: assumption is that neighborhood does NOT change
                # TODO use proper debugging tools here utilizing conda
                print(self.matrix[x_p[:, n], :][:, x_p[:, n]])
                print(type(self.matrix))
                temp_k += self.matrix[x_p[:, n], :][:, x_p[:, n]]
            temp_k *= self.matrix[x_p[:, idx], :][:, x_p[:, idx]]
            k += temp_k
        norm = np.sqrt(np.diag(k))[:, np.newaxis]
        k_hat = k / norm.dot(norm.T)
        return torch.Tensor(k_hat).type(torch.float64)