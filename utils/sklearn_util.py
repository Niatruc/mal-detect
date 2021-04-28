import sklearn
import joblib

class SklearnModel():
    """
    docstring
    """
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, x):
        """
        docstring
        """
        y = self.model.predict