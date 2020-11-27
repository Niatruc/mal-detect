import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K, losses
from sklearn.neighbors import NearestNeighbors
import utils, de, fgsm, evade_at_test_time, exe_util
from file_util import preprocess
import functools


# 实验发现,pad_len为32时没有效果,到64时则可以
def gen_adv_samples(
        model, fn_list,
        strategy=0, changed_bytes_cnt=16, thres=0.5, batch_size=10,
        *, step_size=0.1, max_iter=1000, individual_cnt=10,
        change_range=0b1111, use_kick_mutation=True, check_convergence_per_iter=100
):
    max_len = int(model.input.shape[1])  # 模型接受的输入数据的长度

    if strategy == 0 or strategy == 1:
        inp2emb = K.function([model.input]+ [K.learning_phase()], [model.layers[1].output]) # 嵌入层函数
        embs = [inp2emb([i])[0] for i in range(0,256)] # 求0~255各数字对应的嵌入向量

    # log = utils.Logger()
    adv_samples = []
    test_info = {}

    predict_func = functools.partial(model.predict, batch_size=batch_size)

    for e, fn in enumerate(fn_list):
        inp, len_list = preprocess([fn], max_len)
        pad_idx = len_list[0]   # 以文件的长度作为填充字节的起始下标
        org_score = model.predict(inp)[0][0]    # 模型对未添加噪声的文件的预测概率(1表示恶意)
        # loss, pred = float('nan'), float('nan')

        if strategy == 0 or strategy == 1:
            pad_len = max(min(changed_bytes_cnt, max_len - pad_idx), 0)
            if pad_len > 0:
                # 填充字节
                noise = np.zeros(pad_len)
                noise = np.random.randint(0, 255, pad_len)
                inp[0][pad_idx: pad_idx + pad_len] = noise
                inp_emb = np.squeeze(np.array(inp2emb([inp, False])), 0)

                if thres < org_score:
                    if strategy == 0:
                        adv, gradient, loss = fgsm(model, inp, inp_emb, pad_idx, pad_len, e, step_size)
                    elif strategy == 1:
                        adv, gradient, loss = evade_at_test_time(model, inp, inp_emb, pad_idx, pad_len, embs, step_size, rounds = 100)
                final_adv = adv[0][:pad_idx + pad_len]
            else:  # 使用原始文件
                final_adv = inp[0][:pad_idx]
        elif strategy == 2:
            # de_attack(model, inp, DOS_HEADER_MODIFY_RANGE[0], change_byte_cnt=4)
            # de_algo = de.DE(inp, model.predict, dim_cnt=2, change_byte_cnt=32, individual_cnt=32 * 2, bounds=[[(pad_idx, pad_idx + 32)], [(0, 255)]])
            modifiable_range_list = exe_util.find_pe_modifiable_range(fn, use_range=change_range)
            de_algo = de.DE(inp, predict_func, dim_cnt=2, changed_bytes_cnt=changed_bytes_cnt, individual_cnt=individual_cnt, bounds=[
                    modifiable_range_list,
                    [(0, 255)]
                ], F=0.2, kick_units_rate=1.,
                check_convergence_per_iter=check_convergence_per_iter,
            )
            adv, iter_sum = de_algo.update(iter_cnt=max_iter, use_kick_mutation=use_kick_mutation)
            final_adv = adv[0]
            test_info['iter_sum'] = iter_sum

        pred = model.predict(adv)[0][0]
        test_info['final_score'] = pred
        # log.write(fn, org_score, pad_idx, pad_len, loss, pred)

        # 整数数组转字节序列
        bin_adv = bytes(list(final_adv))
        adv_samples.append(bin_adv)

    return adv_samples, test_info

