import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from mal_detect.file_util import preprocess

def limit_gpu_memory(per):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per
    set_session(tf.Session(config=config))

    
def train_test_split(data, label, val_size=0.1):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    split = int(len(data)*val_size)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    return x_train, x_test, y_train, y_test


class Logger():
    def __init__(self):
        self.fn = []
        self.len = []
        self.modified_bytes_cnt = []
        self.pred = []
        self.org = []

    def print(self, fn, org_score, file_len, modified_bytes_cnt, pred):
        self.fn.append(fn.split('/')[-1])
        self.org.append(org_score)
        self.len.append(file_len)
        self.modified_bytes_cnt.append(modified_bytes_cnt)
        self.pred.append(pred)
        
        print('\nFILE:', fn)
        if modified_bytes_cnt > 0:
            print('\tfile length:', file_len)
            print('\tpad length:', modified_bytes_cnt)
            print('\tscore:', pred)
        else:
            print('\tfile length:', file_len, ', Exceed max length ! Ignored !')
        print('\toriginal score:', org_score)
        
    def save(self, path):
        d = {'filename':self.fn, 
             'original score':self.org, 
             'file length':self.len,
             'modified bytes':self.modified_bytes_cnt,
             'predict score':self.pred}
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False, columns=['filename', 'original score', 
                                              'file length', 'pad length', 
                                              'predict score'])
        print('\nLog saved to "%s"\n' % path)
