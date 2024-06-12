from itertools import combinations
from typing import List, Tuple

import numpy as np

from util.mlflow.constants import MSE


def filter_functional_variant_data_less_than(
    results_dict: dict, functional_thresholds: list
) -> dict:
    """
    Routine to filter recorded observations by functional thresholds w.r.t. true observations.
    In experimental setup the observations are inverted, thus functional candidates are less equal to threshold
    """
    ref_func_obs = {}
    for protocol, protocol_val in results_dict.items():
        ref_func_obs[protocol] = {}
        for (prot, prot_val), f_threshold in zip(
            protocol_val.items(), functional_thresholds
        ):
            ref_func_obs[protocol][prot] = {}
            for method, method_val in prot_val.items():
                ref_func_obs[protocol][prot][method] = {}
                for rep, rep_val in method_val.items():
                    ref_func_obs[protocol][prot][method][rep] = {}
                    for cv_split, split_val in rep_val.get(None).items():
                        ref_func_obs[protocol][prot][method][rep][None] = {}
                        true_vals = []
                        pred_vals = []
                        for true_val, pred_val in zip(
                            split_val.get("trues"), split_val.get("pred")
                        ):
                            if true_val <= f_threshold:
                                true_vals.append(true_val)
                                pred_vals.append(pred_val)
                        train_vals = []
                        for train_val in split_val.get("train_trues"):
                            if train_val <= f_threshold:
                                train_vals.append(train_val)
                        filtered_obs_dict = {
                            "trues": true_vals,
                            "pred": pred_vals,
                            "train_trues": train_vals,
                        }
                        ref_func_obs[protocol][prot][method][rep][None][
                            cv_split
                        ] = filtered_obs_dict
    return ref_func_obs


def filter_functional_variant_data_greater_than(
    results_dict: dict, functional_thresholds: list
) -> dict:
    ref_func_obs = {}
    for protocol, protocol_val in results_dict.items():
        ref_func_obs[protocol] = {}
        for (prot, prot_val), f_threshold in zip(
            protocol_val.items(), functional_thresholds
        ):
            ref_func_obs[protocol][prot] = {}
            for method, method_val in prot_val.items():
                ref_func_obs[protocol][prot][method] = {}
                for rep, rep_val in method_val.items():
                    ref_func_obs[protocol][prot][method][rep] = {}
                    for cv_split, split_val in rep_val.get(None).items():
                        ref_func_obs[protocol][prot][method][rep][None] = {}
                        true_vals = []
                        pred_vals = []
                        for true_val, pred_val in zip(
                            split_val.get("trues"), split_val.get("pred")
                        ):
                            if true_val > f_threshold:
                                true_vals.append(true_val)
                                pred_vals.append(pred_val)
                        train_vals = []
                        for train_val in split_val.get("train_trues"):
                            if train_val > f_threshold:
                                train_vals.append(train_val)
                        filtered_obs_dict = {
                            "trues": true_vals,
                            "pred": pred_vals,
                            "train_trues": train_vals,
                        }
                        ref_func_obs[protocol][prot][method][rep][None][
                            cv_split
                        ] = filtered_obs_dict
    return ref_func_obs


def parse_baseline_mutation_observations(
    results_dict: dict, metric: callable = np.sum
) -> Tuple[np.ndarray, np.ndarray]:
    """
    REFERENCE ADDITIVE BASELINE.
    A multi-mutant is the combination of its constituents.
        Parse mutation (test) observations  by its constituents (training).
        Return test observations and associated added training observations.
        This is only dependent on training/test data (irrespective of method+representation).
    Input: dict: results
    Output: Tuple: list summed training observations per k-1 mutations, list test observations.
    """
    proteins = list(results_dict.keys())
    methods = list(results_dict.get(proteins[0]).keys())
    representations = list(results_dict.get(proteins[0]).get(methods[0]).keys())
    _dict = (
        results_dict.get(proteins[0]).get(methods[0]).get(representations[0]).get(None)
    )
    additive_training_values = []
    true_observations = []
    train_trues = []
    for cv_split in _dict.keys():
        train_mutations = _dict.get(cv_split).get("train_mutation")
        training_observations = np.array(_dict.get(cv_split).get("train_trues"))
        flat_train_mutations = [
            ("".join(_m), i) for i, _m in enumerate(train_mutations)
        ]
        assert len(train_mutations) == len(training_observations)
        assert len(_dict.get(cv_split).get("test_mutation")) == len(
            _dict.get(cv_split).get("trues")
        )
        split_observations = []
        split_additive_values = []
        for test_mutation, test_val in zip(
            _dict.get(cv_split).get("test_mutation"), _dict.get(cv_split).get("trues")
        ):
            idx = []
            # BASE CASE: go over all mutations components (true for single, double, ... in test)
            for mutation in test_mutation:
                for m, i in flat_train_mutations:
                    if mutation == m:
                        idx.append(i)
            split_observations.append(test_val)
            split_additive_values.append(metric(training_observations[idx]))
            if len(test_mutation) > 2:  # case of k variants look at combination
                for mutation in combinations(
                    test_mutation, len(test_mutation) - 1
                ):  # TODO extend to k-mutations by combinations int is range
                    remainder = np.setdiff1d(test_mutation, mutation)
                    source_mutations_idx = [
                        i for m, i in flat_train_mutations if m == "".join(mutation)
                    ]
                    remainder_idx = [
                        i for m, i in flat_train_mutations if m == remainder
                    ]
                    idx = (
                        source_mutations_idx + remainder_idx
                    )  # concatenate lists of indices
                    # for each combination one observation
                    split_observations.append(test_val)
                    split_additive_values.append(metric(training_observations[idx]))
        assert len(split_additive_values) == len(split_observations)
        true_observations.append(split_observations)
        additive_training_values.append(split_additive_values)
        train_trues.append(training_observations)
    return (
        np.array(additive_training_values),
        np.array(true_observations),
        np.array(train_trues),
    )


def compute_delta_between_results(comp_lst: List[dict]):
    diff = {}
    a, b = comp_lst
    assert a.keys() == b.keys()
    for splitter in a.keys():
        diff[splitter] = {}
        for data in a.get(splitter).keys():
            diff[splitter][data] = {}
            for method in a.get(splitter).get(data).keys():
                diff[splitter][data][method] = {}
                for rep in a.get(splitter).get(data).get(method).keys():
                    diff[splitter][data][method][rep] = {None: {}}
                    for metric in (
                        a.get(splitter).get(data).get(method).get(rep).get(None)
                    ):
                        if (
                            metric == MSE
                        ):  # NOTE: 1-NMSE - 1-NMSE = -NMSE + NMSE , diff in std.R2 scores, denote as such, otherwise downstream issues when computing the metric
                            diff[splitter][data][method][rep][None]["R2"] = (
                                1
                                - np.array(
                                    a.get(splitter)
                                    .get(data)
                                    .get(method)
                                    .get(rep)
                                    .get(None)
                                    .get(metric)
                                )
                            ) - (
                                1
                                - np.array(
                                    b.get(splitter)
                                    .get(data)
                                    .get(method)
                                    .get(rep)
                                    .get(None)
                                    .get(metric)
                                )
                            )
                        else:
                            diff[splitter][data][method][rep][None][metric] = np.array(
                                a.get(splitter)
                                .get(data)
                                .get(method)
                                .get(rep)
                                .get(None)
                                .get(metric)
                            ) - np.array(
                                b.get(splitter)
                                .get(data)
                                .get(method)
                                .get(rep)
                                .get(None)
                                .get(metric)
                            )
    return diff
