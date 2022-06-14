import numpy as np
import matplotlib.pyplot as plt

def plot_mid_training(X_test, y_test, predictions, unc, method):
    plt.scatter(X_test, y_test, label="test data", alpha=0.4)
    plt.errorbar(np.array(X_test).ravel(), np.array(predictions).ravel(), yerr=np.array(unc).ravel(),label=f"predictions {method}", alpha=0.4, fmt="ro")
    plt.xlabel("dELBO")
    plt.ylabel("observations (Y)")
    plt.legend()
    plt.savefig(f"./results/figures/test_data_overview_{method.get_name()}.png")
    plt.show()