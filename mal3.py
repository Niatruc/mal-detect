import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np

class MalDect(nn.Module):
    def __init__(self, input_length, batch_size):
        super(MalDect, self).__init__()

        self.dense = torch.nn.Sequential(
            torch.nn.BatchNorm2d(
                1,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            torch.nn.Conv2d(1, 8, 5, 1, 2),
            torch.nn.Dropout(0.5),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(
                8,
                eps=1e-05,
                momentum=0.1,
                affine=False,
                track_running_stats=True,
            ),
            torch.nn.Conv2d(8, 1, 5, 1, 2),
            torch.nn.Dropout(0.2),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.linear = torch.nn.Linear(150, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        cur_batch_size = x.size()[0]
        x= F.pad(x, (0,19,0,0)).reshape((cur_batch_size, 1, 60, 40))

        x = self.dense(x)
        # x = x.flatten()
        x = x.reshape((cur_batch_size, 150))
        # for _, i in enumerate(x):
        #     print(i)

        x = self.linear(x)
        x = self.sigmoid(x)

        return x