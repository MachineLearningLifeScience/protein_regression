import numpy as np
import pandas as pd


def quantile_and_oracle_errors(uncertainties, errors, number_quantiles):
    """
    AUTHOR: Jacob KH
    Based on a list of uncertainties, a list of errors 
    and a number of quantiles this function outputs
    oracle errors that is the mean of a given percentile when
    percentiles are sorted by the errors and quantile errors that is
    the mean error when sorted by their corresponding uncertainty.

    Input:
        uncertainties: list of uncertainties
        errors: list of prediction errors
        number_quantiles: number of percentile bins between 0 and 1
                          i.e bin size reciprocal of number_quantiles
    Output:
        quantile_errs: list of mean errors when percentiles are made
                        by sorting by uncertainty
        oracle_errs: list of mean errors when percentiles are made
                        by sorting by error
    """
    quantile_errs = []
    oracle_errs = []
    qs = np.linspace(0,1,1+number_quantiles)
    s = pd.DataFrame({'unc': uncertainties, 'err': errors})
    for q in qs:
        idx = (s.sort_values(by='unc',ascending=False).reset_index(drop=True) <= s.quantile(q)).values[:,0]

        quantile_errs.append(np.mean(s.sort_values(by='unc',ascending=False).reset_index(drop=True)[idx]['err'].values))

        idx = (s.sort_values(by='err',ascending=False).reset_index(drop=True) <= s.quantile(q)).values[:,1]
        oracle_errs.append(np.mean(s.sort_values(by='err',ascending=False).reset_index(drop=True)[idx]['err'].values))

    # normalize to the case where all datapoints are included
    quantile_errs = quantile_errs/quantile_errs[-1]
    oracle_errs = oracle_errs/oracle_errs[-1]

    return quantile_errs, oracle_errs



def ranking_confidence_curve(losses: np.array, uncertainties: np.array, quantiles=10):
    """
    AUTHOR: Richard M
    Compute confidence curve from provided losses and associated uncertainties.
    Partition uncertainties into losses, by number of quantiles.
    Assess loss after filtering uncertainties by quantiles iteratively.
    returns: 
    """
    assert len(losses) == len(uncertainties)
    quantiles = np.arange(0, 1.1, 1/quantiles)
    # sort by uncertainties descending
    sorted_tuples = np.array(sorted(zip(losses, uncertainties), key=lambda x: x[1], reverse=True))
    # quantile highest to lowest by uncertainty
    h_quantiles = [np.quantile(uncertainties, q) for q in quantiles][::-1]
    # oracle error in quantiles by error
    h_oracle = [np.quantile(losses, q) for q in quantiles][::-1]
    average_loss_in_quantile = np.array([np.average([t[0] for t in sorted_tuples if t[1] <= h_q]) for h_q in h_quantiles])
    average_loss_in_oracle_quantile = np.array([np.average([t[0] for t in sorted_tuples if t[0] <= h_o]) for h_o in h_oracle])
    assert len(quantiles) == len(h_quantiles) == len(average_loss_in_quantile)
    return average_loss_in_quantile, average_loss_in_oracle_quantile


def area_confidence_oracle_error(uncertainties, oracle_error, quantiles=10):
    """
    Equation (6) Scalia et al. area under the confidence orcale error
    """
    quantiles = np.arange(0, 1, 1/quantiles)
    uncertainties, oracle_error = sorted(uncertainties, reverse=True), sorted(oracle_error, reverse=True)
    quartile_diffs = np.array([unc - oracle for unc, oracle in zip(uncertainties, oracle_error)])
    auco = np.sum(quartile_diffs)
    return auco


def error_drop(h_quantiles: np.ndarray) -> float:
    """
    Equation (7) Scalia et al. difference between first uncertainty quantile and last
    """
    return h_quantiles[0] / h_quantiles[-1]


def decreasing_ratio(h_quantiles: np.ndarray) -> float:
    """
    Equation (8) fractions of uncertainties larger than the next quantiles uncertainties, to cover monotonicity
    """
    # compare uncertainty to next uncertainty and count
    larger_quantile_set = np.array([int(h_i >= next_h_i) for h_i, next_h_i in zip(h_quantiles[:-1], h_quantiles[1:])])
    return np.sum(larger_quantile_set) / (len(h_quantiles)-1)