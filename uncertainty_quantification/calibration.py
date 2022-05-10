from signal import Sigmasks
import numpy as np

def confidence_based_calibration(y_pred: np.array, uncertainties: np.array, y_ref_mean=0, quantiles=10) -> np.array:
    N = len(y_pred)
    quantiles = np.arange(0, 1, 1/quantiles)
    unc_quantiles = [np.quantile(uncertainties, q) for q in quantiles]
    fractions = []
    for sigma_q in unc_quantiles:
        upper_bound, lower_bound = y_ref_mean + sigma_q, y_ref_mean - sigma_q
        interval_count = np.sum(((y_pred + uncertainties) <= upper_bound) & ((y_pred - uncertainties) >= lower_bound))
        fractions.append(interval_count / N)
    return fractions
    

def error_based_calibration(y_trues, y_pred, uncertainties):
    pass