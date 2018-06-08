import numpy as np

def RMSE(predictions: np.ndarray, values: np.ndarray) -> float:
    return np.sqrt(np.mean((predictions - values)**2))
