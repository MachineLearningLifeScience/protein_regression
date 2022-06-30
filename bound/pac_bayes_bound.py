import numpy as np
import tensorflow_probability as tfp
from typing import Tuple
from algorithms.gp_on_real_space import GPonRealSpace


def alquier_bounded_regression(post_mu, post_cov, prior_mu, prior_cov, n: int, delta: float, loss_bound: Tuple[float, float]) -> float:
    """
    Alquiers bound extended to bounded regression.
    See Theorem 3, equation (14) in 
    P Germain, F Bach, A Lacoste, S Lacoste-Julien "PAC-Bayesian Theory Meets Bayesian Inference", NIPS 2016.
    Input: GP-posterior, GP-prior, number of samples, likelihood delta, loss-bound.
    Returns epsilon
    , as in empirical_loss +/- epsilon
    """
    assert delta > 0. and delta <= 1
    assert type(n) == int
    a, b = loss_bound
    posterior = tfp.distributions.Normal(post_mu[:, np.newaxis], post_cov)
    prior = tfp.distributions.Normal(prior_mu[:, np.newaxis], prior_cov)
    # we use expected KLD for multivariate normal
    epsilon = np.mean(tfp.distributions.kl_divergence(posterior, prior).numpy()) + np.log(1/delta) + np.power((b-a),2)/2
    epsilon = (1/np.sqrt(n)) * epsilon
    return epsilon