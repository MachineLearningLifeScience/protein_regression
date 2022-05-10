import numpy as np

def ranking_confidence_curve(losses: np.array, uncertainties: np.array, quantiles=10):
    """
    Compute confidence curve from provided losses and associated uncertainties.
    Partition uncertainties into losses, by number of quantiles.
    Assess loss after filtering uncertainties by quantiles iteratively.
    returns: 
    """
    assert len(losses) == len(uncertainties)
    quantiles = np.arange(0, 1, 1/quantiles)
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


def error_drop(h_quantiles):
    """
    Equation (7) Scalia et al. difference between first uncertainty quantile and last
    """
    return h_quantiles[0] / h_quantiles[-1]