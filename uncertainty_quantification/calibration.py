import numpy as np
from typing import Tuple
from scipy import stats
from sklearn.preprocessing import scale
from util.preprocess import scale_observations


## Uncertainty calibration
# confidence calibration
def prep_reliability_diagram(true, preds, uncertainties, number_quantiles):
    """
    AUTHOR: Jacob KH
    """
    true, preds, uncertainties = np.array(true), np.array(preds), np.array(uncertainties)

    # confidence intervals
    #four_sigma = 0.999936657516334
    #perc = np.concatenate([np.linspace(0.1,0.9,number_quantiles-1),[four_sigma]])
    perc = np.arange(0, 1.1, 1/number_quantiles)
    count_arr = np.vstack([np.abs(true-preds) <= stats.norm.interval(q, loc=np.zeros(len(preds)), scale=uncertainties)[1] for q in perc])
    count = np.mean(count_arr, axis=1)

    # ECE
    ECE = np.mean(np.abs(count - perc))

    # Sharpness
    Sharpness = np.std(uncertainties, ddof=1)/np.mean(uncertainties)

    return count, perc, ECE, Sharpness



def confidence_based_calibration(y_pred: np.array, uncertainties: np.array, y_ref_mean=0, quantiles=10) -> Tuple[np.array, np.array]:
    """
    AUTHOR: Richard M
    See scalia et al. pg.2703 prior to Eq. (9), description of confidence based calibration.

    Assumed standardized values... TODO
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
    raise NotImplementedError("Error Based Calibration is not yet implemented.")


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