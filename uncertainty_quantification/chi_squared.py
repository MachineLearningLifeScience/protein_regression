import numpy as np
import scipy


def chi_squared_stat(y: np.array, y_pred: np.array, var_pred: np.array) -> float:
    """
    NOTE: 
    (If fit accurately predicts the means, then the variance estimate should be in agreement and the ration close to one)
    """
    chi_squ = np.sum(np.square((y - y_pred)) / var_pred)
    return chi_squ


def compute_dof(y: np.array, y_pred: np.array) -> float:
    """
    For Definition of >> dof << see (1.2) in "Degrees of Freedom and Model Search" by R.J.Tibishirani in Statistica Sinica
    See: https://www.jstor.org/stable/24721231 (lasst accessed 28.04.2023 10:00AM)
    Assumptions: predictive error has mean=0 and marginal variance=sigma , errors are independent.
    """
    marginal_var = np.var(y - y_pred)
    dof = np.sum(np.cov(y_pred, y)) / marginal_var # TODO: compute COV correctly!
    return dof


def reduced_chi_squared_stat(y: np.array, y_pred: np.array, var_pred: np.array, dof: float=None) -> float:
    """
    Good fit should be in interval [1., 1.5]. 
    Above or below indicate overconfidence or underconfidence.
    """
    chi = chi_squared_stat(y, y_pred, var_pred)
    N = y.shape[0]
    p = dof if dof else compute_dof(y, y_pred)
    v = N - p - 1
    return chi / v


def chi_squared_anees(y: np.array, y_pred: np.ndarray, var_pred: np.ndarray, eps=0.0001) -> float:
    """
    See: https://probnum-evaluation.readthedocs.io/en/latest/_modules/probnumeval/multivariate/_calibration_measures.html#anees
    """
    cov = np.diag(var_pred) # NOTE: not all off-diagonal / cov. matrices are available for all methods. Assumption: independence.
    centered_mean = y_pred - y
    try:
        L, lower = scipy.linalg.cho_factor(cov, lower=True)
        normalized_discrepancies = centered_mean @ scipy.linalg.cho_solve((L, lower), centered_mean)
    except np.linalg.LinAlgError:
        cov = cov + np.diag(np.repeat(eps, cov.shape[0])) # add epsilon for p.d.
        L, lower = scipy.linalg.cho_factor(cov, lower=True)
        normalized_discrepancies = centered_mean @ scipy.linalg.cho_solve((L, lower), centered_mean)
    return np.mean(normalized_discrepancies)
