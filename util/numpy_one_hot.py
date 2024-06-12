import numpy as np


def numpy_one_hot(X: np.ndarray, max: int) -> np.ndarray:
    Z = np.zeros([X.shape[0], X.shape[1], max], dtype=float)
    ids = np.arange(X.shape[1])
    for j in range(0, X.shape[0]):
        Z[j, ids, X[j, :]] = 1.0
    return Z


def numpy_one_hot_2dmat(X: np.ndarray, max: int) -> np.ndarray:
    return numpy_one_hot(X, max).reshape([X.shape[0], X.shape[1] * max])


def numpy_one_hot_or_zero(X: np.ndarray, max: int, special_char: int) -> np.ndarray:
    """
    One-hot encoding where special_character is encoded as all-zero.
    :param X:
    :param max:
    :param special_char:
    :return:
    """
    Z = np.zeros([X.shape[0], X.shape[1], max], dtype=float)
    ids_ = np.arange(X.shape[1])
    for j in range(0, X.shape[0]):
        ids = np.setdiff1d(ids_, np.where(X[j, :] == special_char))
        Z[j, ids, X[j, :]] = 1.0
    return Z
