import numpy as np


def chi_squared_stat(y: np.array, y_pred: np.array, var_pred: np.array) -> float:
    """
    NOTE: Good fit should be close to one. 
    (If fit accurately predicts the means, then the variance estimate should be in agreement and the ration close to one)
    """
    chi_squ = np.sum(np.square((y - y_pred), axis=1) / var_pred)
    return chi_squ


def reduced_chi_squared_stat(y: np.array, y_pred: np.array, var_pred: np.array, num_fitted_params: int) -> float:
    chi = chi_squared_stat(y, y_pred, var_pred)
    N = y.shape[0]
    p = num_fitted_params
    v = N - p - 1
    return chi / v