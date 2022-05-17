import numpy as np
import pytest
from uncertainty_quantification.calibration import confidence_based_calibration, max_calibration_error
from uncertainty_quantification.calibration import error_based_calibration, expected_normalized_calibration_error
from uncertainty_quantification.calibration import expected_calibration_error
from uncertainty_quantification.confidence import ranking_confidence_curve
from uncertainty_quantification.confidence import area_confidence_oracle_error, decreasing_ratio
from make_uncertain_plots import prep_reliability_diagram, quantile_and_oracle_errors


np.random.seed(1234)

n = 1000
_test_losses_results = np.sort(np.random.normal(loc=1, scale=0.25, size=n))
_test_mean_predictions = np.sort(np.random.normal(loc=0, scale=0.2, size=n))
_uncalibrated_results_uncertainties = np.sort(np.random.normal(loc=0.2, scale=0.1, size=n))[::-1]
_calibrated_results_uncertainties = np.sort(np.random.normal(loc=0.2, scale=0.1, size=n))
quantiles = np.arange(0, 1.1, 1/10)

class TestConfidence:

    def test_mismatch_results_uncertainties(self):
        with pytest.raises(AssertionError):
            ranking_confidence_curve(_test_losses_results[:-2], uncertainties=_uncalibrated_results_uncertainties)

    def test_auco_naive(self):
        quantile = range(10)
        uncalibrated_unc, uncalibrated_oracle = ranking_confidence_curve(_test_losses_results, _uncalibrated_results_uncertainties)
        ref_auco = 0
        for j in quantile:
            ref_auco += uncalibrated_unc[j] - uncalibrated_oracle[j]
        auco = area_confidence_oracle_error(uncalibrated_unc, uncalibrated_oracle)
        np.testing.assert_almost_equal(ref_auco, auco)

    def test_auco_zero_if_perfect(self):
        unc, oracle = ranking_confidence_curve(_test_losses_results, _calibrated_results_uncertainties)
        perfect_auco = area_confidence_oracle_error(unc, oracle)
        assert perfect_auco == 0.

    def test_auco_smaller_if_calibrated(self):
        cal_unc, cal_oracle = ranking_confidence_curve(_test_losses_results, _calibrated_results_uncertainties)
        uncal_unc, uncal_oracle = ranking_confidence_curve(_test_losses_results, _uncalibrated_results_uncertainties)
        cal_auco = area_confidence_oracle_error(cal_unc, cal_oracle)
        uncal_auco = area_confidence_oracle_error(uncal_unc, uncal_oracle)
        assert cal_auco < uncal_auco

    def test_naive_confidence_curve(self):
        example_losses = np.random.normal(loc=1, scale=0.25, size=11)
        example_uncertainties = np.random.normal(loc=0.2, scale=0.1, size=11)
        # compute quantiles
        quantiles_uncertainties = []
        quantiles_losses = []
        for q in quantiles:
            quantiles_uncertainties.append(np.quantile(example_uncertainties, q))
            quantiles_losses.append(np.quantile(example_losses, q))
        # invert quantiles, starting with largest:
        quantiles_uncertainties = np.flip(quantiles_uncertainties)
        quantiles_losses = np.flip(quantiles_losses)
        # compute mean loss by quantiles
        ref_average_losses = []
        for q_u in quantiles_uncertainties:
            losses = []
            for l, unc in zip(example_losses, example_uncertainties):
                if unc <= q_u:
                    losses.append(l)
            ref_average_losses.append(np.sum(losses) / len(losses))
        # oracle is ordered by losses
        ref_average_oracle_loss = []
        for q_l in quantiles_losses:
            losses = []
            for l, unc in zip(example_losses, example_uncertainties):
                if l <= q_l:
                    losses.append(l)
            ref_average_oracle_loss.append(np.sum(losses) / len(losses))
        # compare with
        _losses_unc, _losses_oracle = ranking_confidence_curve(example_losses, example_uncertainties)
        np.testing.assert_almost_equal(ref_average_losses, _losses_unc)
        np.testing.assert_almost_equal(ref_average_oracle_loss, _losses_oracle)

    def test_uncertain_outlier(self):
        """
        test-case: if uncalibrated highest quantile 
        """
        loss_ranked_by_confidence, _ = ranking_confidence_curve(_test_losses_results, _uncalibrated_results_uncertainties)
        # last element larger than first element, bc lower uncertainty with higher loss
        assert loss_ranked_by_confidence[-1] > loss_ranked_by_confidence[0]

    def test_confident_outlier(self):
        loss_ranked_by_confidence, _ = ranking_confidence_curve(_test_losses_results, _calibrated_results_uncertainties)
        # last element smaller than first, bc lower uncertainty with lower loss
        assert loss_ranked_by_confidence[-1] < loss_ranked_by_confidence[1]


    def test_loss_is_oracle(self):
        """
        In our data generation case observations and associated uncertainties are perfectly aligned, hence ordering by uncertainty
        and ordering by loss gives the same curve
        """
        loss_ranked_by_confidence, loss_ranked_by_loss = ranking_confidence_curve(_test_losses_results, _calibrated_results_uncertainties)
        np.testing.assert_array_equal(loss_ranked_by_confidence, loss_ranked_by_loss)

    def test_imperfect_loss_higher_than_oracle(self):
        """
        """
        unordered_test_loss = np.random.normal(loc=1, scale=0.25, size=n)
        loss_by_conf, loss_oracle = ranking_confidence_curve(unordered_test_loss, _calibrated_results_uncertainties)
        # discard starting point because it is equal
        np.testing.assert_array_less(loss_oracle[1:], loss_by_conf[1:])

    def test_random_sampling_constant(self):
        """
        Test requirement: if uncertainties random we have a straight line,
        compare to 1
        ignore last point drop.
        """
        random_uncertainties = np.random.rand(n)
        ranked_confidence, _ = ranking_confidence_curve(_test_losses_results, random_uncertainties)
        np.testing.assert_allclose(ranked_confidence[:-1], np.ones(len(ranked_confidence)-1), rtol=0.1)

    def test_decreasing_ratio_calibrated(self):
        ranked_conf, _ = ranking_confidence_curve(_test_losses_results, _calibrated_results_uncertainties)
        ratio = decreasing_ratio(ranked_conf)
        np.testing.assert_approx_equal(ratio, 1)

    def test_decreasing_ratio_uncalibrated(self):
        ranked_conf, _ = ranking_confidence_curve(_test_losses_results, _uncalibrated_results_uncertainties)
        ratio = decreasing_ratio(ranked_conf)
        np.testing.assert_almost_equal(ratio, 0.)

    def test_decreasing_ratio_random(self):
        random_unc = np.random.rand(n)
        ranked_conf, _ = ranking_confidence_curve(_test_losses_results, random_unc)
        ratio = decreasing_ratio(ranked_conf)
        assert 0.4 <= ratio <= 0.6