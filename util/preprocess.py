import numpy as np

def scale_observations(y):
    mean_y = np.mean(y)
    y -= mean_y
    std_y = np.std(y)
    y = np.nan_to_num(y / std_y) # low-sequence setting: std of one-element array is NaN -> zero
    return mean_y.astype(np.float64), std_y.astype(np.float64), y.astype(np.float64)