import numpy as np
import random
import pandas as pd
import torch
import pyro
from pyro.infer import MCMC, NUTS
import pyro.distributions as dist
import pickle
from os import path


class BayesScaler:
    """
    Bayesian In Silico Scaling for Rosetta simulated data.
    Using MCMC (w/ NUTS sampler)
    """
    def __init__(self, is_mutations: list, ΔΔg, exp_mutations, experimentally_observed_ΔΔg,
                α_a=2., β_a=1.5, α_b=1.3, β_b=2., α_c=2, β_c=5., σ_d=0.15, σ_n=0.5,
                samples_N=10000, warmup_N=500, pdb_ID=None, cached=False, vae=False, holdout_idx=None):
        pyro.set_rng_seed(42)
        pyro.clear_param_store()
        self.pdb_ID = pdb_ID
        self.cached = cached
        self.vae = vae
        self.cached_filename = path.join("./cache/", f"{self.pdb_ID}_scaler_vae{vae}.pkl")
        self.samples_N = samples_N
        self.warmup_N = warmup_N
        self.chains_N = 1
        self.x_range = (min(ΔΔg)-2, max(ΔΔg)+2)
        self.is_mutations = is_mutations
        self.exp_mutations = exp_mutations
        self.ΔΔg_is = ΔΔg
        self.experimentally_observed_ΔΔg = experimentally_observed_ΔΔg
        # persist samples that fitted transformation - exclude downstream
        self.holdout_idx = holdout_idx 

        ΔΔg_exp, ΔΔg_is = self._check_observed()

        self.ΔΔg_exp = torch.Tensor(ΔΔg_exp)
        self.ΔΔg_is = torch.Tensor(ΔΔg_is)
        self.α_a, self.β_a = α_b, β_b
        self.α_b, self.β_b = α_a, β_a
        self.α_c, self.β_c = α_c, β_c
        self.σ_d, self.σ_n = σ_d, σ_n

        self.mcmc = self.get_mcmc_results()

        self.a_samples = self.mcmc.get("a")
        self.b_samples = self.mcmc.get("b")
        self.c_samples = self.mcmc.get("c")
        self.d_samples = self.mcmc.get("d")
        self.a = self.a_samples.mean(0)
        self.b = self.b_samples.mean(0)
        self.c = self.c_samples.mean(0)
        self.d = self.d_samples.mean(0)

        # put samples in range context:
        self.xx = np.arange(self.x_range[0], self.x_range[1], 0.01)
        # theta for all sampled points
        self.θ_samples = np.array([self.a_samples[i] * np.exp(np.dot(self.c_samples[i], 
                self.ΔΔg_is)) + np.dot(self.b_samples[i], self.ΔΔg_is) + self.d_samples[i] for i in range(self.samples_N)])
        self.θ_xx_samples = np.array([self.a_samples[i] * np.exp(np.dot(self.c_samples[i], 
                self.xx)) + np.dot(self.b_samples[i], self.xx) + self.d_samples[i] for i in range(self.samples_N)])
        self.θ = self.a * np.exp(np.dot(self.c, self.ΔΔg_is)) + np.dot(self.b, self.ΔΔg_is) + self.d
        self.θ_xx = np.array(list(map(self.transform, self.xx)))
        self.σ_T = np.sum(np.square(self.θ_samples - self.θ)) / self.samples_N
        self.σ_T_samples = np.sum(np.square(self.θ_samples - self.θ), axis=0) / self.samples_N
        # compute sigma over the whole range of possible inputs
        self.σ_T_xx = np.sum(np.square(self.θ_xx_samples - self.θ_xx), axis=0) / self.samples_N

    def get_mcmc_results(self):
        """
        Checks if cached MCMC samples exist and loads them.
        If not MCMC sampling is conducted on specified models.
        """
        if self.cached and path.isfile(self.cached_filename):
            print(f"Loading saved MCMC run from {self.cached_filename}")
            with open(self.cached_filename, "rb") as infile:
                mcmc = pickle.load(infile)
        else:
            mcmc = self.run_mcmc()
            mcmc = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
            if self.cached:
                with open(self.cached_filename, "wb") as outfile:
                    pickle.dump(mcmc, outfile)
        return mcmc

    def transform(self, x: float) -> float:
        return self.a * np.exp(np.dot(self.c, x)) + np.dot(self.b, x) + self.d
    
    def _check_observed(self) -> np.array:
        """
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

    def _model(self, sim_ΔΔg, obs_ΔΔg):
        a = pyro.sample('a', dist.Gamma(self.α_a, self.β_a))
        b = 0.5 * pyro.sample('b', dist.Beta(self.α_b, self.β_b))
        c = 3.33 * pyro.sample('c', dist.Beta(self.α_c, self.β_c))
        d = pyro.sample('d', dist.Normal(-a, self.σ_d))
        # theta conditioned from in silico data
        θ = a * torch.exp(c*sim_ΔΔg) + b*sim_ΔΔg + d
        with pyro.plate("data", obs_ΔΔg.shape[0]):
            # observations are experimental data
            pyro.sample('obs', dist.Normal(θ, self.σ_n), obs=obs_ΔΔg)
    
    def run_mcmc(self):
        nuts_kernel = NUTS(self._model, jit_compile=True)
        mcmc = MCMC(nuts_kernel, num_samples=self.samples_N, warmup_steps=self.warmup_N)
        mcmc.run(self.ΔΔg_is, self.ΔΔg_exp)
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
       
