import sklearn
import joblib
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
    
    # def predict_with_multiproc(self, x, pool_size=4):
    #     pool = Pool(pool_size)
    #     y = pool.map(self.predict, x)
    #     return y