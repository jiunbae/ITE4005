import numpy as np

def RMSE(predictions, values):
    return np.sqrt(np.mean((predictions - values)**2))
