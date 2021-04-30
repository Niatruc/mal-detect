import numpy as np
import math
import pandas as pd
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session


def limit_gpu_memory(per):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if per > 0:
        config.gpu_options.per_process_gpu_memory_fraction = per
    set_session(tf.Session(config=config))


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


class ExeContentSequence(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(idx)
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)