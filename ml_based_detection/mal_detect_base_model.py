from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation
from utils import file_util

class KerasModel():
    def __init__(self, input_shape=None, path=None, max_len=2**20):
        self.max_len = max_len
        if path is not None:
            self.model = load_model(path)
        
        self.input_shape = input_shape
        if input_shape is None:
            self.input_shape = (max_len, )
    
    def preprocess_input(self, input_data_batch):
        # seqs = file_util.preprocess(fn_list, self.max_len)
        input_data_batch = input_data_batch[:, :self.max_len].reshape(len(input_data_batch), *self.input_shape)
        return input_data_batch
    
    def predict(self, input_data, batch_size=1):
        input_data = self.preprocess_input(input_data)
        res = self.model.predict(input_data, batch_size=batch_size)
        return res