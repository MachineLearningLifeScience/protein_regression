import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_1d(X_test, y_test, predictions, unc, method, rep, dim_reduction):
    plt.scatter(X_test, y_test, label="test data", alpha=0.4)
    plt.errorbar(np.array(X_test).ravel(), np.array(predictions).ravel(), yerr=np.array(unc).ravel(),label=f"predictions {method}", alpha=0.4, fmt="ro")
    plt.xlabel("dELBO")
    plt.ylabel("observations (Y)")
    plt.legend()
    plt.savefig(f"./results/figures/predictions_test_data_overview_{method.get_name()}_1d.png")
    plt.show()


def plot_prediction_2d(X_test, y_test, predictions, unc, method, rep, dim_reduction):
    plt.scatter(X_test[:, 0].ravel(), X_test[:, 1].ravel(), c=y_test, label="assay values", s=7)
    plt.errorbar(X_test[:, 0].ravel(), X_test[:, 1].ravel(), yerr=unc.ravel(), color="k", fmt="o", label="predictions", alpha=0.7, ms=5)
    plt.ylabel("d=2")
    plt.xlabel("d=1")
    plt.title(f"2D representation predictions {method.get_name()}, {rep}, {dim_reduction}")
    plt.legend()
    plt.savefig(f"./results/figures/predictions_test_data_overview_{method.get_name()}_2d_flat.png")
    plt.show()


def plot_prediction_CV(X_test, y_test, predictions, unc, method, rep, dim_reduction):
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    predictions = np.array(predictions)
    unc = np.array(unc)
    if X_test.shape[1] == 1:
        plot_prediction_1d(X_test, y_test, predictions, unc, method, rep, dim_reduction)
    elif X_test.shape[1] == 2:
        plot_prediction_2d(X_test, y_test, predictions, unc, method, rep, dim_reduction)
    else:
        raise ValueError(f"Dimensionality Error! \n Shape of data: {X_test.shape[1]} != [1;2]")