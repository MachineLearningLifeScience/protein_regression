import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from utility import get_mutation_idx
import torch
from torch import cholesky, cholesky_solve
from torch.distributions import MultivariateNormal, Gamma
from typing import Tuple
from protein_representation import ProteinCollection
from graphkernel import KernelLoader
from util import Variable, compute_rmse, compute_ρ

# for reproducability:
torch.manual_seed(42)
np.random.seed(42)


class GPRegression:
    def __init__(self, protein_representation: ProteinCollection, X_wt: np.ndarray, 
                X_exp: np.ndarray, X_is: np.ndarray, y_wt: np.ndarray, y_exp: np.ndarray, y_is: np.ndarray,
                y_max: float, y_mean: float, adjacencies: np.ndarray, σ_T: float, n_optimization=15, 
                fusion=True, sub_matrices=None, cached=False, kernel_vae=None):
        self.X_wt = X_wt
        self.X_exp = X_exp
        self.X_is = X_is
        self.y_wt = y_wt
        self.y_exp = y_exp
        self.y_is = y_is
        self.y_max, self.y_mean = y_max, y_mean
        self.protein = protein_representation
        self.adjacencies = adjacencies
        self.cv_flag: str = None
        self.fusion_flag: bool = fusion
        self.cached: bool = cached
        #self.cache_dir: str = os.path.join(os.path.dirname(__file__), "/cache/")
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
        self.init_σ_T = σ_T
        self.σ_T = self.init_σ_T * torch.ones([1, 1], dtype=torch.float64)
        self.σ = self.set_noise_term()

        self.n_optimization = n_optimization
        self.id = protein_representation.pdb_ID
        self.X, self.y = self._combine_observations(X_wt, X_exp, X_is, y_wt, y_exp, y_is)

        self.μ, self.cov, self.lml, self.p_sample = [], [], [], []

        self._kernel_ids = protein_representation._kernels.sub_matrices_ids
        self._kernels = protein_representation._kernels.kernels
        # init weights 
        self.init_w = (0.9/len(self._kernels)) * torch.ones([len(self._kernels), 1], dtype=torch.float64)
        self.weights = Variable(self.init_w, lower=0, upper=1) 
        if cached:
            try:
                self.covariance_matrices = self.load_cov_matrices()
            except FileNotFoundError as e:
                print(f"Error: Matrix not found! - {e}")
                self.covariance_matrices = self.compute_matrices(X=self.X, adjacencies=self.adjacencies[:len(self.X)])
            # TODO exception for out of bounds with fusion samples
        else:
            self.covariance_matrices = self.compute_matrices(X=self.X, adjacencies=self.adjacencies[:len(self.X)])
        # trainable parameters for testing
        self.trainable_parameters: list = [w for w in self.weights.get_value()] + [self.σ_E, self.σ_S, self.t]
        # DEFAULT: train set to complete data to compute neg-ll correctly while testing
        self.X_train, self.x_test, self.y_train, self.y_test = self.X, None, self.y, None
        self.idx_train, self.idx_test = np.arange(0, self.X.shape[0]), None

    def load_cov_matrices(self) -> list:
        covariance_mats = []
        for k_id in tqdm(self._kernel_ids):
            kernel_name = "{pdb}_{k_id}{fusion}.pt".format(pdb=self.protein.pdb_ID, k_id=k_id,
                                                                         fusion="_fusion" if self.fusion_flag else "")
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

    def _combine_observations(self, X_wt: np.ndarray, X_exp: np.ndarray, X_is: np.ndarray, 
                            y_wt: np.ndarray, y_exp: np.ndarray, y_is: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to initialize X and y torch Tensors from given input data
        :return: combined sequences NxM and ddg Nx1 vector
        """
        y_wt = self.check_and_add_axis(y_wt)
        X_wt = self.check_and_add_axis(X_wt)
        X_exp, X_is = self.check_and_add_axis(X_exp), self.check_and_add_axis(X_is)
        y_exp, y_is = self.check_and_add_axis(y_exp), self.check_and_add_axis(y_is)
        assert X_exp.shape[1] == X_wt.shape[1]
        assert y_exp.shape[1] == y_wt.shape[1]
        if self.fusion_flag:
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
            k += kernel.k(x_p=X, adjacencies=adjacencies)
            if self.cached:
                kernel_name = "{pdb}_{k_id}{fusion}.pt".format(pdb=self.protein.pdb_ID, k_id=k_id, fusion="_fusion"
                                                                    if self.fusion_flag else "")
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

    def parameter_optimization(self) -> None:
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

    def position_level_CV(self, ref=False, optim=True) -> Dict[str, list]:
        self.cv_flag = "pos_lvl_CV"
        mutations = []
        optimization_parameters = []
        fit_parameters = []
        experimental_mutation_index = get_mutation_idx(self.protein.mut_ids_exp)
        for pos in tqdm(range(len(self.protein.sequence))):
            print("reset parameters ...")
            self.reset_GPR() # reset trainable parameters
            # gather all mutations at that position and assign train and test indices
            mutation_bool_mask = np.array([bool(pos in mut) for mut in experimental_mutation_index])
            test_mutation_idx = np.where(mutation_bool_mask)[0]
            not_test_mutation_idx = np.where(~mutation_bool_mask)[0]
            n_mutations = np.array([len(mut) for mut in experimental_mutation_index if bool(pos in mut)])
            # split into train and test
            self.set_test_index(1+test_mutation_idx) # offset with WT 
            # combine WT + not selected + in silico for training data
            train_index = np.concatenate([np.array([0]), 1+not_test_mutation_idx, 
                            np.arange(start=len(self.X_exp)+1, stop=self.X.shape[0])]) # all simulated data are training data
            self.set_train_index(train_index)
            if self.x_test.shape[0] == 0:
                print(f"No Mutation at pos:{pos} - skipping...")
                continue
            # optimize
            nll_init = self.neg_ll()
            if optim:
                try:
                    self.parameter_optimization()
                except RuntimeError as _:
                    print("Optimization broke.")
                    self.reset_trainable_parameters()
            nll_end = self.neg_ll()
            f_μ, cov = self._fit(ref=ref)
            # write optimization results
            optimization_parameters.append({"w": self.weights.get_value(),
                                        "sigma_S": self.σ_S.get_value(),
                                        "sigma_E": self.σ_E.get_value(),
                                        "t": self.t.get_value(),
                                        "nll": (nll_init, nll_end)})
            mutations.append(n_mutations)
            fit_parameters.append({'mu': f_μ.squeeze().detach().numpy(),
                                    'cov': cov.squeeze().detach().numpy(),
                                    'y_exp': (self.y_test.detach().numpy() * self.y_max) + self.y_mean
                                    })
        predictions = np.concatenate([np.atleast_1d(x) for x in [elem.get('mu') for elem in fit_parameters]])
        experimental = np.concatenate([x for sub in [elem.get('y_exp') for elem in fit_parameters] for x in sub])
        rho = compute_ρ(y_vec=experimental, y_pred_μ=predictions)
        rmse = compute_rmse(y=experimental, y_pred_μ=predictions)
        results = {"optimization": optimization_parameters, 
                    "regression": fit_parameters, 
                    "mutations": mutations,
                    "rho": rho,
                    "rmse": rmse}
        return results

    def mutation_level_CV(self, ref=False, optim=True) -> Tuple[List[dict], List[dict]]:
        """
        iteratively sets train and test splits, where one mutation is in the test-set
        has side-effects
        This trains on N-1 data and includes the excluded for test - LOO CV
        TODO not optimal - different approaches needed
        This has been adapted to only use experimental mutations for efficiency
        """
        self.cv_flag = "mut_lvl_CV"
        optimization_parameters = []
        fit_parameters = []
        n_mutations = []
        # get all experimental mutations incl WT
        pbar = tqdm(range(self.X_exp.shape[0] + 1))
        for idx in pbar:
            pbar.set_description(f"Pos: {idx}")
            if idx == 0: # exclude WT from CV
                continue 
            self.reset_GPR()
            # set train and testing indices
            self.set_train_index(np.delete(np.arange(0, self.X.shape[0]), idx))
            self.set_test_index(np.array([idx]))
            n_mutations.append(self.protein.mutation_ids[idx].count(")")) # get mutations by closing brackets on tuple
            nll_init = self.neg_ll()
            if optim:
                try:
                    self.parameter_optimization()
                except RuntimeError as _:
                    print("Optimization broke.")
                    self.reset_trainable_parameters()
            nll_end = self.neg_ll()
            optimization_parameters.append({"nll": (nll_init, nll_end),
                                            "w": self.weights.get_value(),
                                            "sigma_S": self.σ_S.get_value(),
                                            "sigma_E": self.σ_E.get_value(),
                                            "t": self.t.get_value()})
            f_μ, cov = self._fit(ref=ref)
            fit_parameters.append({"mu": f_μ.squeeze().detach().numpy(),
                                "cov": cov.squeeze().detach().numpy(),
                                "y_exp": (self.y_test.detach().numpy() * self.y_max) + self.y_mean
                                })
        predictions = np.concatenate([np.atleast_1d(x) for x in [elem.get('mu') for elem in fit_parameters]])
        experimental = np.concatenate([x for sub in [elem.get('y_exp') for elem in fit_parameters] for x in sub])
        results = { "optimization": optimization_parameters, 
                    "regression": fit_parameters,
                    "rho": compute_ρ(y_vec=experimental, y_pred_μ=predictions),
                    "rmse": compute_rmse(y=experimental, y_pred_μ=predictions),
                    "mutations": n_mutations}
        return results

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

    def _fit(self, ref=False) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def predict(self, f_μ, cov, n_samples=100):
        mN = MultivariateNormal(f_μ, cov)
        p_sample = mN.sample((n_samples,))
        return p_sample