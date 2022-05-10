import pandas as pd
from data.train_test_split import AbstractTrainTestSplitter
from gpflow.kernels import SquaredExponential
from algorithms import GPonRealSpace, RandomForest, KNN
from data.train_test_split import RandomSplitter, BlockPostionSplitter
from util.mlflow.convenience_functions import get_mlflow_results
from util.mlflow.constants import ONE_HOT, TRANSFORMER, VAE
from util.mlflow.constants import GP_LEN, GP_L_VAR, GP_VAR, MLL, MSE, SPEARMAN_RHO


def extract_parameters(datasets: list, algos: list, reps: list, metrics: list, train_test_splitter: AbstractTrainTestSplitter, ):
    results_dict = get_mlflow_results(datasets, algos, reps, metrics, train_test_splitter)
    print(results_dict)
    df = pd.DataFrame(results_dict) # TODO create dataframe from dictionary
    return df


if __name__ == "__main__":
    datasets = ["MTH3"]
    algos = [GPonRealSpace().get_name()]
    representations = [ONE_HOT, TRANSFORMER, VAE]
    metrics = [GP_LEN, GP_L_VAR, GP_VAR, MLL, MSE, SPEARMAN_RHO]
    # datasets = ["MTH3", "TIMB", "CALM", "1FQG", "UBQT", "BRCA"]
    # algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()]
    # representations = [ONE_HOT, TRANSFORMER, VAE]
    # metrics = [GP_LEN, GP_L_VAR, GP_VAR, MLL, MSE, SPEARMAN_RHO]
    train_test_splitter = RandomSplitter # BlockPostionSplitter # 
    parameter_df = extract_parameters(datasets=datasets, algos=algos, reps=representations, metrics=metrics, 
                                    train_test_splitter=train_test_splitter)
    # TODO do we visualize this parameter dataframe?
    print(parameter_df.to_latex(index=False))