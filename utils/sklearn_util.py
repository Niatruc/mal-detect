import sklearn
import joblib
import numpy as np
from multiprocessing import Pool
import joblib

class SklearnModel():
    """
    docstring
    """
    def __init__(self, model_path, max_len=2**20, input_shape=None, scaler_path=None):
        self.model = joblib.load(model_path)
        self.max_len = max_len
        self.input_shape = input_shape
        if input_shape is None:
            self.input_shape = (max_len, )

        self.scaler = None
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)

    def predict(self, x, batch_size=None):
        x = x[:, :self.max_len]
        x = x.reshape(len(x), *self.input_shape)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        try:
            y = self.model.predict_proba(x)
            y = y[:, 1].reshape(y.shape[0], 1)  # 上面方法输出的是各个标签对应的置信度组成的数组, 取第二个, 即标签1对应的置信度
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