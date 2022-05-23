import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from gpflow.kernels import Matern12, Matern52, SquaredExponential
from gpflow.kernels.linears import Linear
from algorithms import GPonRealSpace


def experiment(dimensions: np.array, models: list, random_seeds=[17, 42, 73]):
    results = {d: {m.get_name(): {} for m in models} for d in dimensions}
    for d in tqdm(dimensions):
        X, y = make_regression(n_samples=1000, n_features=d)
        for model in models:
            mse_results, mll_results, r_results, mus, uncertainties = [], [], [], [], []
            for rand in random_seeds:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand)
                y_train = y_train[:, np.newaxis]
                y_test = y_test[:, np.newaxis]
                #model.init_length = 1/d
                model.train(X_train, y_train)
                try:
                    _mu, unc = model.predict_f(X_test)
                except tf.errors.InvalidArgumentError as _:
                    print(f"Experiment: {model.get_name()} in d={d} not stable, prediction failed!")
                    _mu, unc = np.full(y_test.shape, np.nan), np.full(y_test.shape, np.nan)
                baseline = np.mean(np.square(y_test - np.repeat(np.mean(y_test), len(y_test)).reshape(-1,1)))
                err2 = np.square(y_test - _mu)
                mse = np.mean(err2)/baseline
                mll = np.mean(err2 / unc / 2 + np.log(2 * np.pi * unc) / 2)
                r = spearmanr(y_test, _mu)[0] 
                mse_results.append(mse)
                mll_results.append(mll)
                r_results.append(r)
                mus.append(_mu)
                uncertainties.append(np.concatenate(unc))
            results[d][model.get_name()]["mse"] = mse_results
            results[d][model.get_name()]["mll"] = mll_results
            results[d][model.get_name()]["r"] = r_results
            results[d][model.get_name()]["mu"] = mus
            results[d][model.get_name()]["unc"] = uncertainties
    return results


if __name__ == "__main__":
    #dimensions = np.arange(2, 1000, 200)
    dimensions=[2, 5, 10, 20, 100, 250]
    lin_gp = GPonRealSpace(kernel_factory= lambda: Linear(), optimize=True)
    mat52_gp = GPonRealSpace(kernel_factory= lambda: Matern52(), optimize=True)
    squareexp_gp = GPonRealSpace(kernel_factory= lambda: SquaredExponential(), optimize=True)
    method_set = [mat52_gp, squareexp_gp]
    results = experiment(dimensions=dimensions, models=method_set)
    # plot results
    df = pd.DataFrame(columns=["d", "method", "r", "mse", "mll", "unc"])
    for d, dim_entry in results.items():
        for m, entry in dim_entry.items():
            for _mse, _mll, _r, _unc in zip(entry['mse'], entry['mll'], entry['r'], entry['unc']):
                df = pd.concat([pd.DataFrame.from_dict({'d': [d], 'method': [m], 'mse': [_mse], 'mll': [_mll], 'r': [_r], 'unc': [_unc]}), df])
    fig, axs = plt.subplots(1, 4, figsize=(22, 7))
    p1 = sns.boxplot(data=df, x='d', y='r', hue='method', ax=axs[0])
    p2 = sns.boxplot(data=df, x='d', y='mse', hue='method', ax=axs[1])
    p3 = sns.boxplot(data=df, x='d', y='mll', hue='method', ax=axs[2])
    for d in dimensions:
        _df = df[df.d==d]
        for method in results[d].keys():
            __df = _df[_df.method==method]
            try:
                axs[3].hist(np.concatenate(__df.unc.values), 10, label=f'd={d};{method}', alpha=0.5)
                axs[3].set_xlim(0, 2)
            except ValueError:
                continue
    p1.legend_.remove()
    p2.legend_.remove()
    axs[0].set_title("Corr. per dimension")
    axs[1].set_title("MSE per dimension")
    axs[2].set_title("MLL per dimension")
    axs[3].set_title("Sigmas per dimension")
    plt.legend()
    plt.show()