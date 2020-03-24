import sys
base_dir = "/home/bohan/res/ml_models/zbh/"
from mal_detect import mal1, mal2, deepmalnet
# mal_detect.mal1

from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable
import numpy as np
import time

# 参数配置
READ_LEN = 10
batch_size = 16
max_step = 1000
test_step = 20
input_length = 2381
window_size = 24
learning_rate = 0.0001
use_gpu = True
display_step = 10
chkpt_acc_path = base_dir + "mal_dect"

step_msg = 'step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}'
valid_msg = 'step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}'
log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
history = {}
history['tr_loss'] = []
history['tr_acc'] = []

valid_best_acc = 0.0


# 选择模型、损失函数、优化器等
# malconv = mal1.MalDect(input_length = input_length,window_size = window_size)
malconv = deepmalnet.MalDect()

# malconv = mal2.Mal2()

bce_loss = nn.BCEWithLogitsLoss()
adam_optim = optim.Adam([{'params':malconv.parameters()}],lr=learning_rate)
sigmoid = nn.Sigmoid()

if use_gpu:
    malconv = malconv.cuda()
    bce_loss = bce_loss.cuda()
    sigmoid = sigmoid.cuda()

# 载入ember数据集
import ember
ember_data_dir = "/home/bohan/res/ml_dataset/ember2018/"
X_train, y_train, X_test, y_test = ember.read_vectorized_features(ember_data_dir)
metadata_dataframe = ember.read_metadata(ember_data_dir)

# 仅选出已经有标签的数据
train_rows = (y_train != -1)
test_rows = (y_test != -1)
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test[0:batch_size], y_test[0:batch_size])
train_ds = TensorDataset(X_train[train_rows], y_train[train_rows])
test_ds = TensorDataset(X_test[test_rows], y_test[test_rows])

train_dataloader = DataLoader(
    dataset = train_ds,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
    pin_memory=True,
)
test_dataloader = DataLoader(
    dataset = test_ds,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
    pin_memory=True,
)

# 使用ember数据集训练基于pytorch的malconv模型
total_step = 0
while total_step < max_step:

    # Training
    for step, batch_data in enumerate(train_dataloader):
        start = time.time()

        adam_optim.zero_grad()

        cur_batch_size = batch_data[0].size(0)

        exe_input = batch_data[0].cuda() if use_gpu else batch_data[0] # => (batch_size, 2381)
        # exe_input = Variable(exe_input, requires_grad=False)

        label = batch_data[1].cuda() if use_gpu else batch_data[1]
        # label = Variable(label.float(), requires_grad=False)
        label = torch.unsqueeze(label, 1) # => (batch_size, 1)

        pred = malconv(exe_input)
        loss = bce_loss(pred, label)
        print("step: ", step, "loss: ", float(loss), "used time: ", time.time() - start)
        loss.backward()
        adam_optim.step()

        # print('gpu memory usage: ', torch.cuda.memory_allocated(0)/1024**3, 'GB', torch.cuda.memory_cached(0)/1024**3, 'GB')

        history['tr_loss'].append(loss.cpu().data.numpy() + 0)
        history['tr_acc'].extend(
            list(label.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))

        step_cost_time = time.time() - start

        if (step) % display_step == 0:
            print(step_msg.format(total_step, np.mean(history['tr_loss']),
                                  np.mean(history['tr_acc']), step_cost_time))
        total_step += 1

        # Interupt for validation
        if total_step % test_step == 0:
            break
    #
    # # Testing
    # history['val_loss'] = []
    # history['val_acc'] = []
    # history['val_pred'] = []
    #
    # for _, val_batch_data in enumerate(test_dataloader):
    #     cur_batch_size = val_batch_data[0].size(0)
    #
    #     exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
    #     exe_input = Variable(exe_input, requires_grad=False)
    #
    #     label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
    #     label = Variable(label.float(), requires_grad=False)
    #     label = torch.unsqueeze(label, 1)
    #
    #     pred = malconv(exe_input)
    #     loss = bce_loss(pred, label)
    #
    #     history['val_loss'].append(loss.cpu().data.numpy() + 0)
    #     history['val_acc'].extend(
    #         list(label.cpu().data.numpy().astype(int) == (sigmoid(pred).cpu().data.numpy() + 0.5).astype(int)))
    #     history['val_pred'].append(list(sigmoid(pred).cpu().data.numpy()))
    #
    # # print(log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
    # #                      np.mean(history['val_loss']), np.mean(history['val_acc']), step_cost_time),
    # #       file=log, flush=True)
    #
    # print(valid_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
    #                        np.mean(history['val_loss']), np.mean(history['val_acc'])))
    # if valid_best_acc < np.mean(history['val_acc']):
    #     valid_best_acc = np.mean(history['val_acc'])
    #     torch.save(malconv, chkpt_acc_path)
    #     print('Checkpoint saved at', chkpt_acc_path)
    #
    # history['tr_loss'] = []
    # history['tr_acc'] = []