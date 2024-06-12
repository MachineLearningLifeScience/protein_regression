import numpy as np
import scipy


def chi_squared_stat(y: np.array, y_pred: np.array, var_pred: np.array) -> float:
    """
    Compute chi squared statistics of predicted labels respective true labels.
    NOTE:
    If fit accurately predicts the means, then the variance estimate should be in agreement and the ration close to one.
    Parameters:
        y (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        var_pred (np.ndarray): predictive variances
    Returns:
        chi_squ (float): chi-squared computed value
    """
    chi_squ = np.sum(np.square((y - y_pred)) / var_pred)
    return chi_squ


def compute_dof(y: np.array, y_pred: np.array) -> float:
    """
    Computes DOF from observations by variance estimates.
    For Definition of >> dof << see (1.2) in "Degrees of Freedom and Model Search" by R.J.Tibishirani in Statistica Sinica
    See: https://www.jstor.org/stable/24721231 (lasst accessed 28.04.2023 10:00AM)
    Assumptions: predictive error has mean=0 and marginal variance=sigma , errors are independent.
    Parameters:
        y (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
    Returns:
        dof (float): problem degree of freedom estimate.
    """
    marginal_var = np.var(y - y_pred)
    dof = np.sum(np.cov(y_pred, y)) / marginal_var  # TODO: compute COV correctly!
    return dof


def reduced_chi_squared_stat(
    y: np.array, y_pred: np.array, var_pred: np.array, dof: float = None
) -> float:
    """
    Computes reduced Chi-squared statistic.
    Computes chi estimate, normalizes by methods DOF.
    Good fit should be in interval [1., 1.5].
    Above or below indicate overconfidence or underconfidence.
    Parameters:
        y (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        var_pred (np.ndarray): predictive variances
        dof (float)[OPTIONAL]: if algorithm DOF available, input here.
    Returns:
        reduced_chi_squ (float): reduced chi-squared stats computed value
    """
    chi = chi_squared_stat(y, y_pred, var_pred)
    N = y.shape[0]
    p = dof if dof else compute_dof(y, y_pred)
    v = N - p - 1
    return chi / v


def chi_squared_anees(
    y: np.array, y_pred: np.ndarray, var_pred: np.ndarray, eps=0.0001
) -> float:
    """
    Lin.Alg. method to compute chi-squared statistics based on ProbNum package.
    Original implementation under MIT license: Copyright (c) 2021 Probabilistic Numerics.
    See: https://probnum-evaluation.readthedocs.io/en/latest/_modules/probnumeval/multivariate/_calibration_measures.html#anees
    Key assumption: independence for off-diagonal observations.
    Initially used for testing reduced Chi-squared stats computation.
    """
    cov = np.diag(
        var_pred
    )  # NOTE: not all off-diagonal / cov. matrices are available for all methods. Assumption: independence.
    centered_mean = y_pred - y
    try:
        L, lower = scipy.linalg.cho_factor(cov, lower=True)
        normalized_discrepancies = centered_mean @ scipy.linalg.cho_solve(
            (L, lower), centered_mean
        )
    except np.linalg.LinAlgError:
        cov = cov + np.diag(np.repeat(eps, cov.shape[0]))  # add epsilon for p.d.
        L, lower = scipy.linalg.cho_factor(cov, lower=True)
        normalized_discrepancies = centered_mean @ scipy.linalg.cho_solve(
            (L, lower), centered_mean
        )
    except ValueError:  # e.g. Nans in the diagonal
        return np.nan  # NOTE: case NaN predictions in the results
    return np.mean(normalized_discrepancies)
