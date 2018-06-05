from lib.estimator import Estimator

class SVD(Estimator):
    def __init__(self, factors=100, epochs=20, mean=0,
                 lr=.005, reg=.02, random_state=None):
        self.state = random_state
        self.factors = factors
        self.epochs = epochs
        self.mean = mean
        self.lr = lr
        self.reg = reg

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        state = random_state

        pass

    def predict(self, X: np.ndarray) -> Any:
        pass
