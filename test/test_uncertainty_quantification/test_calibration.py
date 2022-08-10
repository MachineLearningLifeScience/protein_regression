import numpy as np
import pytest
from uncertainty_quantification.calibration import confidence_based_calibration, max_calibration_error
from uncertainty_quantification.calibration import error_based_calibration, expected_normalized_calibration_error
from uncertainty_quantification.calibration import expected_calibration_error
from uncertainty_quantification.confidence import ranking_confidence_curve
from uncertainty_quantification.confidence import area_confidence_oracle_error, decreasing_ratio
from make_plot_uncertainties import prep_reliability_diagram, quantile_and_oracle_errors

np.random.seed(1234)

n = 1000
_test_losses_results = np.sort(np.random.normal(loc=1, scale=0.25, size=n))
_test_mean_predictions = np.sort(np.random.normal(loc=0, scale=0.2, size=n))
_uncalibrated_results_uncertainties = np.sort(np.random.normal(loc=0.3, scale=0.5, size=n))[::-1]
_calibrated_results_uncertainties = np.sort(np.random.normal(loc=0.2, scale=0.01, size=n))
quantiles = np.arange(0, 1, 1/10)

class TestCalibration:
    def test_naive_calibration(self):
        pred_mean = 0.
        # determine uncertainty quantiles
        quantiles = np.arange(0, 1.1, 0.1)
        reference_fractions_in_quantile = []
        for unc_q in quantiles:
            predictions_in_quantile = []
            for y_pred, unc in zip(_test_mean_predictions, _calibrated_results_uncertainties):
                # test if it falls into the Gaussian, zero-mean
                if (y_pred + unc) <= pred_mean+unc_q and (y_pred - unc) >= pred_mean-unc_q:
                    predictions_in_quantile.append(y_pred)
            frac = len(predictions_in_quantile)/len(_test_mean_predictions)
            reference_fractions_in_quantile.append(frac)
        # assert against modudle function
        _frac_quantiles, _unc_quantiles = confidence_based_calibration(_test_mean_predictions, _calibrated_results_uncertainties, y_ref_mean=pred_mean)
        np.testing.assert_equal(reference_fractions_in_quantile, _frac_quantiles)
        np.testing.assert_equal(quantiles, _unc_quantiles)

    def test_perfect_calibration(self):
        """Testcase: perfect prediction, 0,0 and 1,1 as diagonal"""
        diag = np.arange(0, 1.1, 1/10)
        predictions = np.arange(0, 1.1, 0.1)
        uncertainties = np.arange(0.01, 0.12, 0.01)
        frac_quantiles, _unc_quantiles = confidence_based_calibration(predictions, uncertainties)
        np.testing.assert_almost_equal(diag, frac_quantiles, 1)

    def test_random_calibration_flat(self):
        random_uncertainties = np.random.normal(loc=0.5, scale=0.2, size=n)
        frac_quantiles, unc_quantiles = confidence_based_calibration(_test_mean_predictions, random_uncertainties)
        np.testing.assert_array_less(frac_quantiles[:int(len(frac_quantiles)/2)], 0.25)

    def test_mce(self):
        frac_quantiles, unc_quantiles = confidence_based_calibration(_test_mean_predictions, _calibrated_results_uncertainties)
        _max_diff = np.max(np.abs(frac_quantiles - unc_quantiles))
        mce = max_calibration_error(frac_quantiles, unc_quantiles)
        assert mce == _max_diff

    def test_ece_calibrated_lower_random(self):
        calibrated_frac_quantiles, calibrated_unc_quantiles = confidence_based_calibration(_test_mean_predictions, _calibrated_results_uncertainties)
        calibrated_ece = expected_calibration_error(calibrated_frac_quantiles, calibrated_unc_quantiles)
        rand_frac_quantiles, rand_unc_quantiles = confidence_based_calibration(_test_mean_predictions, np.random.normal(0.3, 0.1, n))
        rand_ece = expected_calibration_error(rand_frac_quantiles, rand_unc_quantiles)
        assert calibrated_ece <= rand_ece

    def test_ece_uncalibrated_greater_random(self):
        uncal_frac_quantiles, uncal_unc_quantiles = confidence_based_calibration(_test_mean_predictions, _uncalibrated_results_uncertainties)
        uncal_ece = expected_calibration_error(uncal_frac_quantiles, uncal_unc_quantiles)
        rand_frac_quantiles, rand_unc_quantiles = confidence_based_calibration(_test_mean_predictions, np.random.normal(0.3, 0.1, n))
        rand_ece = expected_calibration_error(rand_frac_quantiles, rand_unc_quantiles)
        assert rand_ece <= uncal_ece

    def test_ence_calibrated_lesser_random(self):
        ence = expected_normalized_calibration_error(_test_losses_results, _calibrated_results_uncertainties)
        random_unc = np.random.normal(0.2, 0.5, n)
        random_ence = expected_normalized_calibration_error(_test_losses_results, random_unc)
        assert ence <= random_ence

    def test_ence_uncalibrated_great_random(self):
        ence = expected_normalized_calibration_error(_test_losses_results, _uncalibrated_results_uncertainties)
        random_unc = np.random.normal(0.2, 0.5, n)
        random_ence = expected_normalized_calibration_error(_test_losses_results, random_unc)
        assert ence >= random_ence

    def test_ence_random(self):
        # TODO verify is ENCE expected to be around 0.59 ??
        random_unc = np.random.normal(0.2, 0.1, n)
        ence = expected_normalized_calibration_error(_test_losses_results, random_unc)
        assert 0.49 <= ence <= 0.6


