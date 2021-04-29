import sklearn
import joblib
import numpy as np
from multiprocessing import Pool

class SklearnModel():
    """
    docstring
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, x):
        try:
            y = self.model.predict_proba(x)
            y = y[:, 1]
        except:
            y = self.model.predict(x)
        return y
    
    def predict_generator(self, generator, steps, workers=2, use_multiprocessing=True, verbose=0):
        """
        docstring
        """
        if use_multiprocessing:
            pool = Pool(workers)
            y_all = np.array([])
            for xs, _ in generator:
                y = pool.map(self.predict, np.expand_dims(xs, 1))
                y = np.array(y)
                if len(y) > 0:
                    y_all = np.concatenate((y_all.reshape(-1, y.shape[-1]), y))
            y = y_all
        else:
            for xs, _ in generator:
                y = self.predict(xs)
        return y

    def predict_with_multiproc(self, x, pool_size=4):
        pool = Pool(pool_size)
        y = pool.map(self.predict, np.expand_dims(x, 1))
        return y