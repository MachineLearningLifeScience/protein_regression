import numpy as np
import random
import pandas as pd
#import torch
#import pyro
#from pyro.infer import MCMC, NUTS
#import pyro.distributions as dist
import pickle
from os import path
from typing import List
from algorithms.abstract_algorithm import AbstractAlgorithm


class BayesScaler(AbstractAlgorithm):
    """
    Bayesian regression scaling using MCMC (w/ NUTS sampler).
    On joint subset of mutations given simulated experimental values
    and experimental observations.

    Input:
    is_mutations -- List[str]: in-silico mutation names
    ΔΔg -- np.ndarray: in-silico experimental observations
    exp_mutations -- List[str]: experimental mutation names
    experimentally_observed_ΔΔg -- np.ndarray: experimental observations
    """
    def __init__(self,
                α_a=2., β_a=1.5, α_b=1.3, β_b=2., α_c=2, β_c=5., σ_d=0.15, σ_n=0.5,
                samples_N=10000, warmup_N=500, train_test_split=0.2):
        pyro.set_rng_seed(42)
        pyro.clear_param_store()
        self.model = None
        self.mcmc = None
        self.train_test_split = train_test_split
        self.samples_N = samples_N
        self.warmup_N = warmup_N
        self.chains_N = 1   
        self.α_a, self.β_a = α_b, β_b
        self.α_b, self.β_b = α_a, β_a
        self.α_c, self.β_c = α_c, β_c
        self.σ_d, self.σ_n = σ_d, σ_n

    def get_name(self):
        return "BS"

    def train(self, X, Y):
        self.x_range = (min(Y)-2, max(Y)+2)
        Y = np.random.shuffle(Y)
        holdout_idx = int(self.train_test_split*Y.shape[0])
        y_train, y_test = torch.Tensor(Y[:holdout_idx]), torch.Tensor(Y[holdout_idx:])
        self.mcmc = self.run_mcmc(y_exp=y_test, y_is=y_train)

    def predict(self, X):
        mcmc = {k: v.detach().cpu().numpy() for k, v in self.mcmc.get_samples().items()}
        a_samples = mcmc.get("a")
        b_samples = mcmc.get("b")
        c_samples = mcmc.get("c")
        d_samples = mcmc.get("d")
        a = a_samples.mean(0)
        b = b_samples.mean(0)
        c = c_samples.mean(0)
        d = d_samples.mean(0)

        # put samples in range context:
        xx = np.arange(self.x_range[0], self.x_range[1], 0.01)
        # theta for all sampled points
        θ_samples = np.array([a_samples[i] * np.exp(np.dot(c_samples[i], 
                X)) + np.dot(b_samples[i], X) + d_samples[i] for i in range(self.samples_N)])
        θ_xx_samples = np.array([a_samples[i] * np.exp(np.dot(c_samples[i], 
                xx)) + np.dot(b_samples[i], xx) + d_samples[i] for i in range(self.samples_N)])
        θ = a * np.exp(np.dot(c, X)) + np.dot(b, X) + d
        θ_xx = np.array(list(map(self.predict, xx)))
        σ_T = np.sum(np.square(θ_samples - θ)) / self.samples_N
        σ_T_samples = np.sum(np.square(θ_samples - θ), axis=0) / self.samples_N
        # compute sigma over the whole range of possible inputs
        σ_T_xx = np.sum(np.square(θ_xx_samples - θ_xx), axis=0) / self.samples_N
        y_pred = a * np.exp(np.dot(c, X)) + np.dot(b, X) + d
        # TODO assert ∃ sigma per prediction
        return y_pred, σ_T
    
    def _check_observed(self) -> np.array:
        """
        NOTE: used in mgpfusion context - discontinued
        set of simulated mutations that have an experimental counterpart
        :Input: 
            ΔΔg = list of (Rosetta) simulated values
            experimentally_observed_ΔΔg = list of experimentally verified (Protherm) values
        :Output:
            experimental ΔΔg values that have been simulated 
        """
        assert(len(self.is_mutations) == len(self.ΔΔg_is) and len(self.exp_mutations) == len(self.experimentally_observed_ΔΔg))
        simulated_mutations = np.array(self.is_mutations)
        observed_ΔΔg = []
        simulated_ΔΔg = []
        for val, mut in zip(self.experimentally_observed_ΔΔg, self.exp_mutations):
            if mut in self.is_mutations:
                s_idx = np.where(simulated_mutations==mut)[0][0]
                observed_ΔΔg.append(val)
                simulated_ΔΔg.append(self.ΔΔg_is[s_idx])
        # subsample intersection for VAEs, otherwise total overfit        
        if self.vae and len(observed_ΔΔg) >= 0.5*len(self.experimentally_observed_ΔΔg): 
            n_subsamples = int(0.2*len(self.experimentally_observed_ΔΔg))
            subsample = random.sample(list(zip(observed_ΔΔg, simulated_ΔΔg)), k=n_subsamples)
            observed_ΔΔg, simulated_ΔΔg = list(zip(*subsample))
        return list(observed_ΔΔg), list(simulated_ΔΔg)

    def _model(self, y_is, y_exp):
        a = pyro.sample('a', dist.Gamma(self.α_a, self.β_a))
        b = 0.5 * pyro.sample('b', dist.Beta(self.α_b, self.β_b))
        c = 3.33 * pyro.sample('c', dist.Beta(self.α_c, self.β_c))
        d = pyro.sample('d', dist.Normal(-a, self.σ_d))
        # theta conditioned from in silico data
        θ = a * torch.exp(c*y_is) + b*y_is + d
        with pyro.plate("data", y_exp.shape[0]):
            # observations are experimental data
            pyro.sample('obs', dist.Normal(θ, self.σ_n), obs=y_exp)
    
    def run_mcmc(self, y_is, y_exp):
        nuts_kernel = NUTS(self._model, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=self.samples_N, warmup_steps=self.warmup_N)
        mcmc.run(y_is, y_exp)
        return mcmc

    @staticmethod
    def summary(samples: dict) -> dict:
        """
        Utility function to print latent sites' quantile information.
        """
        site_stats = {}
        for site_name, values in samples.items():
            marginal_site = pd.DataFrame(values)
            describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
            site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats

    def print_summary(self):
        for site, values in self.summary(self.mcmc).items():
            print("Site: {}".format(site))
            print(values, "\n")
       
