import numpy as np
from itertools import combinations
from typing import Tuple


def filter_functional_variant_data(results_dict: dict, functional_threshold) -> dict:
    functional_observations = {protocol: {
                                prot: {
                                    method: {
                                        rep: {
                                            cv_split: {
                                                'trues': [val for val in split_val.get('trues') if val > functional_threshold], 
                                                'mse': [val for true_val, val in zip(split_val.get('trues'), split_val.get('mse')) if true_val > functional_threshold],
                                    } for cv_split, split_val in rep_val.get(None).items()
                                } for rep, rep_val in method_val.items()
                        } for method, method_val in prot_val.items()
                    } for prot, prot_val in protocol_val.items()
                } for protocol, protocol_val in results_dict.items()}
    return functional_observations


def parse_baseline_mutation_observations(results_dict: dict, metric: callable=np.sum) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse mutation (test) observations  by its constituents (training).
    Return test observations and associated added training observations.
    This is only dependent on training/test data (irrespective of method+representation).
    Input: dict: results
    Output: Tuple: list summed training observations per k-1 mutations, list test observations.
    """
    proteins = list(results_dict.keys())
    methods = list(results_dict.get(proteins[0]).keys())
    representations = list(results_dict.get(proteins[0]).get(methods[0]).keys())
    _dict = results_dict.get(proteins[0]).get(methods[0]).get(representations[0]).get(None)
    additive_training_values = []
    true_observations = []
    for cv_split in _dict.keys():
        train_mutations = _dict.get(cv_split).get("train_mutation")
        training_observations = np.array(_dict.get(cv_split).get("train_trues"))
        flat_train_mutations = [("".join(_m), i) for i, _m in enumerate(train_mutations)]
        split_observations = []
        split_additive_values = []
        for test_mutation, test_val in zip(_dict.get(cv_split).get("test_mutation"), _dict.get(cv_split).get("trues")):
            idx = []
            # BASE CASE: go over all mutations components (true for single, double, ... in test)
            for mutation in test_mutation: 
                for m, i in flat_train_mutations:
                    if mutation==m:
                        idx.append(i)
            split_observations.append(test_val)
            split_additive_values.append(metric(training_observations[idx]))
            if len(test_mutation) > 2: # case of k variants look at combination 
                for mutation in combinations(test_mutation, len(test_mutation)-1): # TODO extend to k-mutations by combinations int is range
                    remainder = np.setdiff1d(test_mutation, mutation)
                    source_mutations_idx = [i for m, i in flat_train_mutations if m=="".join(mutation)]
                    remainder_idx = [i for m, i in flat_train_mutations if m==remainder]
                    idx = source_mutations_idx + remainder_idx # concatenate lists of indices
                    # for each combination one observation
                    split_observations.append(test_val)
                    split_additive_values.append(metric(training_observations[idx]))
        assert len(split_additive_values) == len(split_observations)
        true_observations.append(split_observations)
        additive_training_values.append(split_additive_values)       
    return np.array(additive_training_values), np.array(true_observations)