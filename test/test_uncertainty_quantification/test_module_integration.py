import numpy as np
import pytest
from uncertainty_quantification.calibration import (
    confidence_based_calibration,
    max_calibration_error,
)
from uncertainty_quantification.calibration import (
    error_based_calibration,
    expected_normalized_calibration_error,
)
from uncertainty_quantification.calibration import expected_calibration_error
from uncertainty_quantification.confidence import ranking_confidence_curve
from uncertainty_quantification.confidence import (
    area_confidence_oracle_error,
    decreasing_ratio,
)
from make_plot_uncertainties import prep_reliability_diagram, quantile_and_oracle_errors


np.random.seed(1234)
n = 1000
_test_losses_results = np.sort(np.random.normal(loc=1, scale=0.25, size=n))
_test_mean_predictions = np.sort(np.random.normal(loc=0, scale=0.2, size=n))
_uncalibrated_results_uncertainties = np.sort(
    np.random.normal(loc=0.2, scale=0.1, size=n)
)[::-1]
_calibrated_results_uncertainties = np.sort(
    np.random.normal(loc=0.2, scale=0.1, size=n)
)
quantiles = np.arange(0, 1, 1 / 10)


class TestModuleIntegration:
    def test_reliability_prep(self):
        true_values = _test_mean_predictions - np.ones(len(_test_mean_predictions))
        count, perc, ECE, Sharpness = prep_reliability_diagram(
            true=true_values,
            preds=_test_mean_predictions,
            uncertainties=_calibrated_results_uncertainties,
            number_quantiles=11,
        )
        fractions, uncertainties = confidence_based_calibration(
            _test_mean_predictions, _calibrated_results_uncertainties
        )
        # np.testing.assert_allclose(uncertainties, perc)
        np.testing.assert_equal(fractions, count)
        ece = expected_calibration_error(fractions, uncertainties)
        assert ECE == ece

    def test_confidence_curve(self):
        q_err, o_err = ranking_confidence_curve(
            losses=_test_losses_results, uncertainties=_calibrated_results_uncertainties
        )
        _q_err, _o_err = quantile_and_oracle_errors(
            uncertainties=_calibrated_results_uncertainties,
            errors=_test_losses_results,
            number_quantiles=10,
        )
        np.testing.assert_almost_equal(q_err, np.flip(_q_err), decimal=0.3)
        np.testing.assert_almost_equal(o_err, np.flip(_o_err), decimal=0.3)
