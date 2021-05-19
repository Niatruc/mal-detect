from keras import load_model
from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation
import mal_detect_base_model

class KerasModel(mal_detect_base_model.KerasModel):
    def __init__(self, path=None, max_len=2**20, win_size=500, vocab_size=256):
        super().__init__()
        
        if not hasattr(self, 'model'):
            inp = Input((max_len,))
            emb = Embedding(vocab_size, 8)(inp)

            conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
            conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
            a = Activation('sigmoid', name='sigmoid')(conv2)
            
            mul = multiply([conv1, a])
            a = Activation('relu', name='relu')(mul)
            p = GlobalMaxPool1D()(a)
            d = Dense(64)(p)
            out = Dense(1, activation='sigmoid')(d)

            self.model = Model(inp, out)
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        