from keras import load_model
from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation
import mal_detect_base_model

class KerasModel(mal_detect_base_model.KerasModel):
    def __init__(self, path=None, max_len=2**20, win_size=500, vocab_size=256):
        super().__init__()
        
        if not hasattr(self, 'model'):
            model = Sequential([
                Convolution2D(
                    filters=50, kernel_size=5, border_mode='valid', activation='relu',
                    batch_input_shape= (batch_size, *input_shape),
                    data_format='channels_last'
                ), # channel_last表示通道数放在input_shape的最后; border_mode其实可不用填,默认是valid, 也就是输出的长宽会小一点
                MaxPooling2D(pool_size=(2, 2), strides=1, border_mode='valid'),
            #     BatchNormalization(),
            # # ])
            # # ([  
                Convolution2D(filters=70, kernel_size=3, activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=1),
                BatchNormalization(),
                
                Convolution2D(filters=70, kernel_size=3, activation='relu'),
                MaxPooling2D(pool_size=(2, 2), strides=1),
                BatchNormalization(),
                
                Flatten(),
                Dense(units=256, activation='relu'),
                Dense(units=1, activation='sigmoid'),
            ])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        