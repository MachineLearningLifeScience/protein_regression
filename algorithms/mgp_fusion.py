import torch
import numpy as np
from algorithms.mgp.kernels import KernelLoader
from algorithms.mgp.gp_regression import GPRegression
from algorithms.abstract_algorithm import AbstractAlgorithm
from algorithms.mgp.fusion_scaler import BayesScaler
from typing import List, Tuple
from util.preprocess import preprocess_observations

class GPfusion(AbstractAlgorithm):
    def __init__(self, kernel: KernelLoader, optimize=True, fusion=False) -> None:
        super().__init__()
        self.gp = None
        self.kernel = kernel # load MKL via KernelLoader
        self.optimize = optimize
        self.fusion = fusion
        self.__adjacencies = None # TODO compute/import contact map adjacencies
        self.__y_is: np.ndarray = np.array([])
        self.__x_is: np.ndarray = np.array([])
        self.__train_indices: np.ndarray = None
        self.__test_indices: np.ndarray = None

    @property
    def train_indices(self):
        return self.__train_indices

    @property
    def test_indices(self):
        return self.__test_indices

    @property
    def y_is(self):
        return self.__y_is

    @property
    def x_is(self):
        return self.__x_is

    @property
    def adjacencies(self):
        # TODO assertions here ...
        return self.__adjacencies

    @train_indices.setter
    def train_indices(self, x: np.ndarray):
        if self.gp:
            self.__train_indices = x
            self.gp.set_train_index(x)
        else:
            raise ValueError("Cannot set train indices GP not initialized")
    
    @test_indices.setter
    def test_indices(self, x: np.ndarray):
        if self.gp:
            self.__test_indices = x
            self.gp.set_test_index(x)
        else:
            raise ValueError("Cannot set test indices GP not initialized")

    @y_is.setter
    def y_is(self, x: np.ndarray):
        self.__y_is = x # TODO implement checks and assertion on input data here

    @x_is.setter
    def x_is(self, x: np.ndarray):
        self.__x_is = x # TODO implement checks and assertion on input data here

    @adjacencies.setter
    def adjacencies(self, x: List[Tuple[str, List[int]]]):
        self.__adjacencies = x # TODO implement adjacencies sanity check ??

    def get_name(self):
        name = "mGP"
        name = name + "fusion" if self.fusion else name
        name += "_" + "".join(self.kernel.sub_matrices_ids)
        return name

    def train(self, X, Y):
        X_wt = X[0, :][np.newaxis, :]
        X_exp = X[1:, :]
        y_wt = Y[0][:, np.newaxis]
        y_exp = Y[1:]
        y_scaled = np.array([])[:, np.newaxis]
        σ_transform = np.array([])[:, np.newaxis] # TODO check if these dimensions are sensible
        print(X_wt.shape)
        print(X_exp.shape)
        print(y_wt.shape)
        print(y_exp.shape)
        mean_y, max_y, y_wt, y_exp, _ = preprocess_observations(y_wildtype=y_wt, y_wetlab=y_exp, y_scaled=y_scaled)
        if self.fusion:
            # TODO set/get mutation IDs for training subset
            scaler = BayesScaler(is_mutations=mut_ids_is, ΔΔg=y_is, exp_mutations=np.array(mut_ids_exp)[holdout_idx],
                                experimentally_observed_ΔΔg=holdout_ΔΔg_exp, cached=True, holdout_idx=holdout_idx)
            y_scaled = scaler.transform(self.y_is)[:, np.newaxis]
            σ_transform = scaler.σ_T
            mean_y, max_y, y_wt, y_exp, y_scaled = preprocess_observations(y_wt, y_exp, y_scaled)
        self.gp = GPRegression(X_wt=X_wt, X_exp=X_exp, X_is=self.x_is, y_wt=y_wt, y_exp=y_exp, y_is=y_scaled, 
                        adjacencies=self.__adjacencies, σ_T=σ_transform, y_max=max_y, y_mean=mean_y, 
                        cached=True, fusion=self.fusion, kernel_loader=self.kernel)
        assert(Y.shape[1] == 1)
        self._optimize() # TODO verify that optimization is done with correct split (indices) !
        # self.gp = GPR(data=(tf.constant(X), tf.constant(Y)),
        #               kernel=self.kernel, mean_function=Constant())

    def predict(self, X):
        return self.gp.predict(torch.Tensor(X))

    def _optimize(self) -> None:
        if self.optimize and self.gp:
            self.gp._optimize()
        return