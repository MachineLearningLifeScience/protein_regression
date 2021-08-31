import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
from typing import Tuple
from util.variable import Variable
from algorithms.mgp.kernels import KernelLoader

# for reproducability:
torch.manual_seed(42)
np.random.seed(42)


class GPRegression:
    def __init__(self, X_wt: np.ndarray, X_exp: np.ndarray, X_is: np.ndarray, 
                y_wt: np.ndarray, y_exp: np.ndarray, y_is: np.ndarray, kernel_loader: KernelLoader,
                y_max: float, y_mean: float, adjacencies: np.ndarray, σ_T: float, n_optimization=15, 
                fusion=True, cached=True):
        self.X_wt = X_wt
        self.X_exp = X_exp
        self.X_is = X_is
        self.y_wt = y_wt
        self.y_exp = y_exp
        self.y_is = y_is
        self.y_max, self.y_mean = y_max, y_mean
        self.adjacencies = adjacencies
        self.fusion: bool = fusion
        self.cached: bool = cached
        self.cache_dir: str = os.path.join("./cache/")
        # set hyperparameters - see Appendix mGPfusion 
        α_E=2.5
        β_E=1/0.02
        α_S=50.
        β_S=1/0.007
        self.init_t = 1.1 * torch.ones([1, 1], dtype=torch.float64)
        self.t = Variable(self.init_t, lower=0.001, upper=10)
        # init prior noise
        self.σ_E_prior = Gamma(torch.tensor(α_E), torch.tensor(β_E))
        self.σ_S_prior = Gamma(torch.tensor(α_S), torch.tensor(β_S))
        # init noise terms
        self.init_σ_E = 0.075 * torch.ones([1, 1], dtype=torch.float64)
        self.init_σ_S = 0.1 * torch.ones([1, 1], dtype=torch.float64)
        self.σ_E = Variable(self.init_σ_E, lower=0.001, upper=10)
        self.σ_S = Variable(self.init_σ_S, lower=0.001, upper=10)
        self.σ_0 = 1e-6 * torch.ones([1, 1], dtype=torch.float64)
        self.init_σ_T = torch.tensor(σ_T).type(torch.float64)
        self.σ_T = self.init_σ_T * torch.ones([1, 1], dtype=torch.float64)
        self.σ = self.set_noise_term()

        self.n_optimization = n_optimization
        self.X, self.y = self._input_to_tensor(X_wt, X_exp, X_is, y_wt, y_exp, y_is)

        self.μ, self.cov, self.lml, self.p_sample = [], [], [], []

        self._kernel_ids = kernel_loader.sub_matrices_ids
        self._kernels = kernel_loader.kernels
        # init weights 
        self.init_w = (0.9/len(self._kernels)) * torch.ones([len(self._kernels), 1], dtype=torch.float64)
        self.weights = Variable(self.init_w, lower=0, upper=1) 
        if cached:
            try:
                self.covariance_matrices = self.load_cov_matrices()
            except FileNotFoundError as e:
                print(f"Error: Matrix not found! - {e} \n Computing new matrix...")
        # TODO exception for out of bounds with fusion samples
        self.covariance_matrices = self.compute_matrices(X=self.X, adjacencies=self.adjacencies[:len(self.X)])
        # trainable parameters for testing
        self.trainable_parameters: list = [w for w in self.weights.get_value()] + [self.σ_E, self.σ_S, self.t]
        # # DEFAULT: train set to complete data to compute neg-ll correctly while testing
        # self.X_train, self.x_test, self.y_train, self.y_test = self.X, None, self.y, None
        # self.idx_train, self.idx_test = np.arange(0, self.X.shape[0]), None

    def load_cov_matrices(self) -> list:
        covariance_mats = []
        for k_id in tqdm(self._kernel_ids):
            kernel_name = "{k_id}{fusion}.pt".format(k_id=k_id, fusion="_fusion" if self.fusion else "")
            k = torch.load(os.path.join(self.cache_dir, kernel_name))
            covariance_mats.append(k)
        return covariance_mats 

    def reset_trainable_parameters(self) -> None:
        """
        reset trainable parameters to initial values
        """
        self.t = Variable(self.init_t, lower=0.001, upper=10)
        self.σ_E = Variable(self.init_σ_E, lower=0.001, upper=10)
        self.σ_S = Variable(self.init_σ_S, lower=0.001, upper=10)
        self.weights = Variable(self.init_w, lower=0, upper=1)
        return None

    def reset_GPR(self) -> None:
        """
        reset state of GPR object 
        """
        # check that cached covariance matrices are unaffected...
        assert np.all([mat.shape[0] == mat.shape[1] for mat in self.covariance_matrices])
        assert np.all([mat.shape[0] == self.X.shape[0] for mat in self.covariance_matrices])
        # reset computed covariances
        self.X_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.idx_train, self.idx_test = None, None
        self.reset_trainable_parameters()
        return None
    
    @staticmethod
    def check_and_add_axis(x: np.ndarray) -> np.ndarray:
        return x[:, np.newaxis] if len(x.shape) == 1 else x

    def _input_to_tensor(self, X_wt: np.ndarray, X_exp: np.ndarray, X_is: np.ndarray, 
                            y_wt: np.ndarray, y_exp: np.ndarray, y_is: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to initialize X and y torch Tensors from given input data
        :return: combined sequences NxM and ddg Nx1 vector
        """
        y_wt = self.check_and_add_axis(y_wt)
        X_wt = self.check_and_add_axis(X_wt)
        X_exp, X_is = self.check_and_add_axis(X_exp), self.check_and_add_axis(X_is)
        y_exp, y_is = self.check_and_add_axis(y_exp), self.check_and_add_axis(y_is)
        print("X exp shape {}".format(X_exp.shape))
        print("X wt shape {}".format(X_wt.shape))
        assert X_exp.shape[-1] == X_wt.shape[-1]
        assert y_exp.shape[-1] == y_wt.shape[-1]
        if self.fusion:
            X = torch.Tensor(np.vstack([X_wt, X_exp, X_is])).type(dtype=torch.float64)
            y = torch.Tensor(np.vstack([y_wt, y_exp, y_is])).type(dtype=torch.float64)
            assert X.shape[1] == len(self.protein.sequence)
            assert X.shape[0] == y.shape[0]
           # assert y.shape[0] == self.protein.ΔΔg.shape[0] # check against original reference
        else:
            X = torch.Tensor(np.vstack([X_wt, X_exp])).type(dtype=torch.float64)
            y = torch.Tensor(np.vstack([y_wt, y_exp])).type(dtype=torch.float64)
        return X, y
    
    def set_noise_term(self):
        σ_E = self.σ_E.get_value()
        σ_S = self.σ_S.get_value()
        t = self.t.get_value()
        σ = torch.cat((self.σ_0, 
                    (σ_E/self.y_max) * torch.ones([self.X_exp.shape[0], 1], dtype=torch.float64), 
                    ((σ_E + σ_S) / self.y_max) * torch.ones([len(self.X_is), 1], dtype=torch.float64) + t*(self.σ_T/self.y_max)))
        return torch.square(σ).type(torch.float64)

    @torch.no_grad()
    def compute_matrices(self, X: torch.Tensor, adjacencies: List[tuple]) -> list:
        X = X.detach().numpy().astype(np.int64)
        n = X.shape[0]
        covariance_mats = []
        for i, (kernel, k_id) in tqdm(enumerate(zip(self._kernels, self._kernel_ids))):
            k = torch.zeros([n, n], dtype=torch.float64)
            print(k)
            print(type(k))
            print(kernel.k(x_p=X, adjacencies=adjacencies))
            print(type(kernel.k(x_p=X, adjacencies=adjacencies)))
            k += kernel.k(x_p=X, adjacencies=adjacencies)
            if self.cached:
                kernel_name = "{k_id}{fusion}.pt".format(k_id=k_id, fusion="_fusion" if self.fusion_flag else "")
                torch.save(k, os.path.join(self.cache_dir, kernel_name))
            covariance_mats.append(k)
        return covariance_mats

    def mWDK(self, X: torch.Tensor, covariance_matrices:list, x: torch.Tensor=None) -> torch.Tensor:
        """
        compute weighted kernel value from existing covariance matrix
        """
        n = X.shape[0]
        m = X.shape[0] if x is None else x.shape[0]
        k = torch.zeros([n, m], dtype=torch.float64)
        assert np.all(n == mat.shape[0] for mat in self.covariance_matrices)
        for i, mat in enumerate(covariance_matrices):
            # WARN: This operation is of type double, but torch doesnt complain
            k += self.weights.get_value()[i].type(torch.float64) * mat
        return k

    def neg_ll(self):
        n = self.X_train.shape[0]
        zero_μ = torch.zeros(n, dtype=torch.float64)
        cov_mats = [cov[self.idx_train, :][:, self.idx_train] for cov in self.covariance_matrices]
        K_XX = self.mWDK(X=self.X_train, covariance_matrices=cov_mats)
        # get noise on relevant data by index
        noise = self.set_noise_term().squeeze()[self.idx_train] 
        K_XX = K_XX + torch.diag(noise)
        # zero mean is consistent due to prior assumption
        nll = -(MultivariateNormal(zero_μ, covariance_matrix=K_XX).log_prob(torch.flatten(self.y_train)).sum() \
            + self.σ_E_prior.log_prob(self.σ_E.get_value()) + self.σ_S_prior.log_prob(self.σ_S.get_value()))
        nll.type(torch.float64).requires_grad_(True)
        return nll

    def _optimize(self) -> None:
        # TODO handle exceptions for LBFGS optimization - catch optim iteration idx
        optimizer = torch.optim.LBFGS([self.weights.unconstrained, self.σ_E.unconstrained, 
                                    self.σ_S.unconstrained, self.t.unconstrained], lr=0.99)
        def closure():
            optimizer.zero_grad()
            loss = self.neg_ll()
            loss.backward()
            return loss
        for n in range(self.n_optimization):
            optimizer.step(closure)
        return 
    
    def set_test_index(self, index: np.ndarray) -> None:
        self.idx_test = index
        self.x_test = self.X[index]
        self.y_test = self.y[index]

    def set_train_index(self, index: np.ndarray) -> None:
        self.idx_train = index
        self.X_train = self.X[index]
        self.y_train = self.y[index]

    def derive_Xx(self):
        """
        select n x m covariance matrices from existing ones
        """
        matrices = []
        n, m = self.X_train.shape[0], self.x_test.shape[0]
        for mat in self.covariance_matrices:
            k = torch.zeros([n, m], dtype=torch.float64)
            n_selected = mat[self.idx_train, :]
            m_selected = n_selected[:, self.idx_test]
            if m_selected.shape[0] != m:
                m_selected = m_selected.reshape(n, m)
            k += m_selected
            matrices.append(k)
        return matrices

    def predict(self, ref=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alg. 2.1 Rasmussen *GPs in ML* """
        n = self.X_train.shape[0]
        m = self.x_test.shape[0]
        train_mats = [cov[self.idx_train, :][:, self.idx_train] for cov in self.covariance_matrices]
        test_mats = [cov[self.idx_test, :][:,self.idx_test] for cov in self.covariance_matrices]
        K_XX = self.mWDK(X=self.X_train, covariance_matrices=train_mats)
        K_xx = self.mWDK(X=self.x_test, covariance_matrices=test_mats)
        cov_mats = self.derive_Xx()
        K_Xx = self.mWDK(X=self.X_train, x=self.x_test, covariance_matrices=cov_mats)
        σ = self.set_noise_term()[self.idx_train]
        if ref:
            σ += σ
        A = K_XX + σ * torch.eye(n)
        L = cholesky(A)
        α = cholesky_solve(self.y_train, L)
        # compute disttribution and lml
        f_μ = self.y_max * torch.matmul(K_Xx.T, α) + self.y_mean
        v = cholesky_solve(K_Xx, L)
        cov = (self.y_max*self.y_max) * (K_xx - torch.matmul(K_Xx.T, v))
        return f_μ, cov