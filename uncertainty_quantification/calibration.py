import numpy as np
from typing import Tuple

def confidence_based_calibration(y_pred: np.array, uncertainties: np.array, y_ref_mean=0, quantiles=10) -> Tuple[np.array, np.array]:
    """
    See scalia et al. pg.2703 prior to Eq. (9), description of confidence based calibration.
    """
    assert len(y_pred) == len(uncertainties)
    N = len(y_pred)
    quantiles = np.arange(0, 1.1, 1/quantiles)
    fractions = []
    for sigma_q in quantiles:
        upper_bound, lower_bound = y_ref_mean + sigma_q, y_ref_mean - sigma_q
        interval_count = np.sum(((y_pred + uncertainties) <= upper_bound) & ((y_pred - uncertainties) >= lower_bound))
        fractions.append(interval_count / N)
    return np.array(fractions), np.array(quantiles)
    

def error_based_calibration(y_trues, y_pred, uncertainties):
    pass


def expected_calibration_error(loss_fractions: np.ndarray, conf_interval_values: np.ndarray) -> float:
    """
    Equation (10) ECE ratio of absolute difference between (loss_fractions in confidence interval) and confidence interval
    """
    assert len(loss_fractions) == len(conf_interval_values)
    return  np.sum(np.abs(loss_fractions - conf_interval_values)) / len(loss_fractions)


def max_calibration_error(loss_fractions: np.ndarray, conf_interval_values: np.ndarray) -> float:
    """
    Equation (10) MCE ratio of absolute difference  between loss-fraction in interval and interval
    """
    assert len(loss_fractions) == len(conf_interval_values)
    return np.max(np.abs(loss_fractions - conf_interval_values))

def expected_normalized_calibration_error(losses: np.ndarray, uncertainties: np.ndarray, n_quantiles=10) -> float:
    """
    Eq. (11) Expected normalized calibration error
    """
    unc_quantiles = np.arange(0.1, 1.1, 1/n_quantiles)
    m_vars = np.nan_to_num(np.array([np.sqrt(np.mean(np.array([u for u in uncertainties if u<=k]))) for k in unc_quantiles]))
    m_losses = np.nan_to_num(np.array([np.mean(np.array([l for l, unc in zip(losses, uncertainties) if unc<= k])) 
                for k in unc_quantiles]))
    ence = np.sum(np.abs(np.array(m_vars) - np.array(m_losses))) / len(unc_quantiles)
    return ence