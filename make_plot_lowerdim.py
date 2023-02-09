from algorithms import GPonRealSpace
from gpflow.kernels import SquaredExponential
from algorithms import RandomForest
from algorithms import KNN
from data.train_test_split import BlockPostionSplitter, RandomSplitter, PositionSplitter
from util.mlflow.constants import MSE, ONE_HOT, VAE, TRANSFORMER, ESM, LINEAR, NON_LINEAR, SPEARMAN_RHO, MLL, EVE
from visualization.plot_lowerdim import plot_lower_dim_results


if __name__ == "__main__":
    datasets = ["1FQG", "UBQT", "CALM"] # "TIMB", "CALM", "1FQG", "UBQT", "BRCA"
    algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name()] # 
    metrics = [MSE, SPEARMAN_RHO]
    representations = [TRANSFORMER, ESM, ONE_HOT, EVE]
    dimensions = [2, 10, 100, 1000, None]
    dim_reduction = LINEAR # LINEAR, NON_LINEAR
    cv_types = [RandomSplitter(datasets[0]), PositionSplitter(datasets[0])]
    for dataset in datasets:
        plot_lower_dim_results(datasets=[dataset], algorithms=algos, representations=representations, cv_types=cv_types, 
                        dimensions=dimensions, metrics=metrics, dim_reduction=dim_reduction)