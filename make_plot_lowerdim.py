from algorithms import GPonRealSpace
from gpflow.kernels import SquaredExponential
from algorithms import RandomForest
from algorithms import KNN
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from util.mlflow.constants import MSE, ONE_HOT, VAE, TRANSFORMER, ESM, LINEAR, NON_LINEAR, SPEARMAN_RHO, MLL
from visualization.plot_lowerdim import plot_lower_dim_results


if __name__ == "__main__":
    datasets = ["1FQG"]
    algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), RandomForest().get_name()]
    metrics = [MSE, MLL]
    representations = [ONE_HOT, VAE, TRANSFORMER, ESM]
    dimensions = [2, 10, 100, 1000, None]
    dim_reduction = LINEAR # LINEAR, NON_LINEAR
    cv_types = [BlockPostionSplitter, RandomSplitter]
    plot_lower_dim_results(datasets=datasets, algorithms=algos, representations=representations, cv_types=cv_types, 
                        dimensions=dimensions, metrics=metrics, dim_reduction=dim_reduction)