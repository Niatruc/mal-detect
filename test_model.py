import deepmalnet
import torch
import numpy as np

print(deepmalnet)
mal = deepmalnet.MalDect()
print(mal)

a = [i for i in range(2381)]
a.extend(a)
a = torch.Tensor(a).view(2, 2381)
# a = a.view(1, 2381)
o = mal(a)
print(a, o)