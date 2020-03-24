import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalDect, self).__init__()

        # self.embed = nn.Embedding(257, 8, padding_idx=0)
        self.f2v = nn.Linear(1, 2)

        self.conv_1 = nn.Conv1d(1, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(1, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # x = self.embed(x)
        x = torch.unsqueeze(x, 2)
        x = self.f2v(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 1))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 1, 1)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x

#########################################################################################################
from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation

def KerasModel(max_len=200000, win_size=500, vocab_size=256):    
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

    model = Malconv(max_len, win_size)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model