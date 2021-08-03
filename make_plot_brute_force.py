# %%
import mlflow
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.linear_gp import GPOneHotSequenceSpace
from util.mlflow.constants import DATASET, METHOD, MSE
from util.mlflow.convenience_functions import find_experiments_by_tags
from visualization.plot_metric_for_dataset import plot_metric_for_dataset


def read_dict(results_dict, cv_type=None):
    # read jacobs dicts
    if cv_type == 'reg':
        mse = results_dict['mse_reg']
        std = results_dict['mse_reg_std']
    elif cv_type == 'bio':
        mse = results_dict['mse_bio']
        std = results_dict['mse_bio_std']
    return mse, std


# gathers all our results and saves them into a numpy array
dataset = "1FQG"
metric = MSE
results_list_reg = []
results_list_bio = []


# Jacob's code
rep_names = ['onehot', 'avgProtbert', 'avgProtbertUMAP', 'avgProtbertVAE', 'VAE1', 'VAE2', 'GPonehot']
names = ['blat', 'mth3', 'timb', 'calm', 'brca']
for name in names:
    mse_list_cvreg = []
    std_list_cvreg = []
    mse_list_cvbio = []
    std_list_cvbio = []

    # load dicts of dataset
    results_vae = pickle.load( open('results/jacob_results/'+name.lower()+'_VAE_results.pkl', "rb" ) )
    results_transavg = pickle.load( open('results/jacob_results/'+name.lower()+'_avgtransformer_results.pkl', "rb" ) )
    results_transavgumap = pickle.load( open('results/jacob_results/'+name.lower()+'_avgtransformerUMAP_results.pkl', "rb" ) )
    results_vae_phyla = pickle.load( open('results/jacob_results/'+name.lower()+'_VAEphyla_labelled_results.pkl', "rb" ) )
    results_onehot = pickle.load( open('results/jacob_results/'+name.lower()+'_onehot_results.pkl', "rb" ) )
    results_transavgvae = pickle.load(open('results/jacob_results/'+name.lower()+'_avgtransVAE_results.pkl', 'rb'))

    result_dicts = [results_onehot, results_transavg, results_transavgumap, results_transavgvae,
                    results_vae, results_vae_phyla]

    for r_dict in result_dicts:
        mse, std = read_dict(r_dict, cv_type='reg')
        mse_list_cvreg.append(mse)
        std_list_cvreg.append(std)

        mse, std = read_dict(r_dict, cv_type='bio')
        mse_list_cvbio.append(mse)
        std_list_cvbio.append(std)

    results_list_reg.append([mse_list_cvreg, std_list_cvreg])
    results_list_bio.append([mse_list_cvbio, std_list_cvbio])

# Simon's code
result_list = []
algorithm_name_list = []
simons_algos = [GPOneHotSequenceSpace(alphabet_size=0)]
for a in simons_algos:
    exps = find_experiments_by_tags({DATASET: dataset, METHOD: a.get_name()})
    assert(len(exps) == 1)
    algorithm_name_list.append(a.get_name())
    runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id])
    results = []
    for id in runs['run_id'].to_list():
        for metric in mlflow.tracking.MlflowClient().get_metric_history(id, metric):
            results.append(metric.value)
    print(results)
    if len(result_list) > 0:
        assert(len(results) == len(result_list[-1]))
    result_list.append(results)
gponehot_mse_bio = [np.mean(res) for res in result_list]
gponehot_std_bio = [np.std(res, ddof=1) for res in result_list]
gponehot_mse_bio, gponehot_std_bio

# Richard's code
# then calls #plot_metric_for_dataset
results = np.array(results_list_bio)
print(results.shape)

# %%
import mlflow
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.linear_gp import GPOneHotSequenceSpace
from util.mlflow.constants import DATASET, METHOD, MSE
from util.mlflow.convenience_functions import find_experiments_by_tags
from visualization.plot_metric_for_dataset import plot_metric_for_dataset

rep_names = ['onehot', 'avgProtbert', 'avgProtbertUMAP', 'avgProtbertVAE', 'VAE1', 'VAE2', 'GPonehot']
names = ['blat', 'mth3', 'timb', 'calm', 'brca']

results_list_reg = [[[1,2,1],[2,1,2],[3,2,3],[4,5,4],[5,4,5]], 
                    [[1,2,1],[2,1,2],[3,2,3],[4,5,4],[5,4,5]],
                    [[1,2,1],[2,1,2],[3,2,3],[4,5,4],[5,4,5]], 
                    [[1,2,1],[2,1,2],[3,2,3],[4,5,4],[5,4,5]], 
                    [[1,2,1],[2,1,2],[3,2,3],[4,5,4],[5,4,5]]]

plot_metric_for_dataset(datasets=names, metric_values=results_list_reg, reps=rep_names, cvtype='reg')

# %%
