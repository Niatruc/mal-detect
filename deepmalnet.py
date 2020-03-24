import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np

class MalDect(nn.Module):
    def __init__(self):
        super(MalDect, self).__init__()

        self.model = torch.nn.Sequential(

        )

        layer_neurons_cnt = [2381, 4608, 4096, 3584, 3072, 2560, 2048, 1536, 1024, 512, 128]

        self.model.add_module(
            "pre_BN",
            torch.nn.BatchNorm1d(
                2381,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        )

        for i, c in enumerate(layer_neurons_cnt):
            if i < len(layer_neurons_cnt) - 1:
                self.add_layer(
                    "layer#%i" % i,
                    c,
                    layer_neurons_cnt[i + 1],
                )

        self.model.add_module(
            "last_layer",
            torch.nn.Sequential(
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            )
        )

    def add_layer(self, layer_name, input_cnt, output_cnt):
        self.model.add_module(
            layer_name,
            torch.nn.Sequential(
                torch.nn.Linear(input_cnt, output_cnt),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(
                    output_cnt,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
                torch.nn.Dropout(0.2),
            )
        )

    def forward(self, x):
        x = self.model(x)
        return x