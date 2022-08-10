import pandas as pd
import numpy as np
from itertools import product
from data.train_test_split import AbstractTrainTestSplitter
from gpflow.kernels import SquaredExponential
from algorithms import GPonRealSpace, RandomForest, KNN
from data.train_test_split import RandomSplitter, BlockPostionSplitter
from util.mlflow.convenience_functions import get_mlflow_results
from util.mlflow.constants import ONE_HOT, TRANSFORMER, VAE, ESM, VAE_AUX, VAE_RAND
from util.mlflow.constants import GP_LEN, GP_L_VAR, GP_VAR, MLL, MSE, SPEARMAN_RHO


def extract_parameters(datasets: list, algos: list, reps: list, metrics: list, train_test_splitter: AbstractTrainTestSplitter, ):
    results_dict = get_mlflow_results(datasets, algos, reps, metrics, train_test_splitter)
    cv_name = train_test_splitter(datasets[0]).get_name()
    sub_metrics = ["mu", "std"]
    header_multi_index = pd.MultiIndex.from_tuples(((a, m, sm) for a in results_dict.get(datasets[0]).keys() 
                                                        for m in metrics for sm in sub_metrics), 
                                            names=["algo", "metric", "sub_m"])
    multi_index = pd.MultiIndex.from_tuples(((d,r, cv_name) for d in datasets for r in reps), 
                                            names=["data", "representation", "cv"])
    df = pd.DataFrame(columns=header_multi_index, index=multi_index)
    for alg in results_dict.get(datasets[0]).keys():
        for metric in metrics:
            for dat in datasets:
                for rep in reps:
                    entry = results_dict.get(dat).get(alg).get(rep).get(None).get(metric)
                    df.loc[(dat, rep, cv_name), (alg, metric)] = (np.round(np.mean(entry), 2), np.round(np.std(entry), 2))
    return df


if __name__ == "__main__":
    datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"]
    algos = [GPonRealSpace().get_name()] # GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()
    representations = [ONE_HOT, TRANSFORMER, ESM, VAE, VAE_AUX, VAE_RAND]
    metrics = [GP_LEN, GP_L_VAR, GP_VAR, MLL, MSE, SPEARMAN_RHO]
    train_test_splitter = RandomSplitter # BlockPostionSplitter # 
    parameter_df = extract_parameters(datasets=datasets, algos=algos, reps=representations, metrics=metrics, 
                                    train_test_splitter=train_test_splitter)
    # TODO do we visualize this parameter dataframe?
    print(parameter_df.to_latex(bold_rows=True, multicolumn_format="r"))