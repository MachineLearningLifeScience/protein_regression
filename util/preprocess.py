import numpy as np

def scale_observations(y):
    mean_y = np.mean(y)
    y -= mean_y
    std_y = np.std(y)
    y /= std_y
    return mean_y, std_y, y