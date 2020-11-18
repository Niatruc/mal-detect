import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K, losses
from sklearn.neighbors import NearestNeighbors
from mal_detect import utils, de
from mal_detect.differential_evolution import differential_evolution
# from mal_detect import file_util
from mal_detect.file_util import preprocess
import functools

DOS_HEADER_MODIFY_RANGE = [(2, 0x40 - 4 - 1)]

def evade_at_test_time(model, inp, inp_emb, pad_idx, pad_len, embs, step_size=0.1, stop_threshold=0.5, rounds=10):
    adv = inp.copy()
    adv_emb = inp_emb.copy()

    loss = losses.binary_crossentropy(model.output[:, 0], 1)
    grads = K.gradients(loss, model.layers[1].output)[0]
    neg_grads_sign = K.sign(grads)     # 梯度上升方向的向量(各个方向长为1)

    a = [1., 1., 1., 1., 1., 1., 1., 1., ]
    a = a / np.sqrt(np.sum(np.square(a)))

    # dire_vec = neg_grads_sign / (K.sqrt(K.sum(K.square(neg_grads_sign))) + 1e-8) # 梯度上升方向的单位向量
    dire_vec = neg_grads_sign * a

    iterate = K.function([model.layers[1].output], [model.output, loss, grads, neg_grads_sign, dire_vec])

    for r in range(rounds):
        model_output, loss_val, grads_val, neg_grads_sign_val, dire_vec_val = iterate(adv_emb)

        for j in range(pad_len):    # 对每个填充字节执行操作
            emb_j = adv_emb[0][pad_idx + j] # 第j个填充字节的嵌入形式
            d = float('inf')
            for i in range(len(embs)):  # 遍历0~255所有256个字节
                emb_i = embs[i]

                dire_vec_val_j = dire_vec_val[0][pad_idx + j]  # 得到第j个填充字节的嵌入向量
                if np.all(dire_vec_val_j == 0):
                    dire_vec_val_j = a

                # 求将emb_j移动到<emb_i在梯度向量的投影点>需要的距离
                s_i = (emb_i - emb_j).dot(dire_vec_val_j)

                # (emb_j + s_i * dire_vec_val_j)得到<emb_i在梯度向量的投影点>对应的向量
                d_i = np.linalg.norm(emb_i - (emb_j + np.dot(s_i, dire_vec_val_j))) # np.linalg.norm这个函数用来求范数(默认是二范数)

                if s_i > 0 and d_i <= d:    # s_i大于0是为了确保干扰点顺着梯度上升方向移动
                    adv[0][pad_idx + j] = i
                    adv_emb[0][pad_idx + j] = emb_i
                    d = d_i
            # print("predict score after change %dth byte to value %d: %f " % (pad_idx + j, i, model.predict(adv)))

        adv_score = model.predict(adv)
        print("predict score after %dth round: %f " % (r, adv_score))
        if adv_score <= stop_threshold:
            break

    return adv, grads_val, loss_val


def fgsm(model, inp,  inp_emb, pad_idx, pad_len, e, step_size=0.1):
    adv_emb = inp_emb.copy()
    # loss = K.mean(model.output[:, 0])
    loss = losses.binary_crossentropy(model.output[:, 0], 1)
    grads = K.gradients(loss, model.layers[1].output)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-8)

    mask = np.zeros(model.layers[1].output.shape[1:]) # embedding layer output shape
    mask[pad_idx:pad_idx+pad_len] = 1
    origin_grads = grads
    grads *= K.constant(mask) # 只保留末尾附加字节对应的梯度

    iterate = K.function([model.layers[1].output], [model.output, loss, grads, origin_grads])
    g = 0.
    step = int(1/step_size)*10
    for _ in range(step):
        model_output, loss_val, grads_val, origin_grads_val = iterate([adv_emb])
        grads_val *= step_size
        if np.all(grads_val==0): # 实验中发现存在梯度全0的情况，遂加上这句判断
            grads_val = np.array([step_size*mask])
        g += grads_val
        adv_emb += grads_val
        # adv += K.sign(grads_value) * 0.1
        # adv += 0.1*mask
        #print (e, loss_val, end='\r')
        # if loss_val >= 0.9:
        if model_output <= 0.5:
            break

    # 用K近邻来寻找嵌入向量对应的字节
    emb_layer = model.layers[1]  # 嵌入层
    emb_weight = emb_layer.get_weights()[0];  # shape: (257, 8)
    neigh = NearestNeighbors(1)
    neigh.fit(emb_weight)
    out = emb_search(inp, adv_emb[0], pad_idx, pad_len, neigh)

    return out, g, loss_val

# 用K近邻来寻找嵌入向量对应的字节
def emb_search(org, adv, pad_idx, pad_len, neigh):
    out = org.copy()
    for idx in range(pad_idx, pad_idx+pad_len):
        target = adv[idx].reshape(1, -1)
        best_idx = neigh.kneighbors(target, 1, False)[0][0]
        out[0][idx] = best_idx
    return out

def de_attack(model, inp, modify_range, change_byte_cnt = 1):
    bounds = [modify_range, (0, 256)] * change_byte_cnt
    best_fitness_val = float('inf')
    def predict_fn(diff_vectors):
        nonlocal best_fitness_val
        advs = []
        for diff_vector in diff_vectors:
            adv = inp.copy()[0]
            i = 0
            while i < len(diff_vector):
                pos = int(diff_vector[i])
                val = int(diff_vector[i + 1])
                adv[pos] = val
                i += 2
            advs.append(adv)
        scores = model.predict(np.array(advs), batch_size=10)
        min_score = np.min(scores)
        if min_score < best_fitness_val:
            best_fitness_val = min_score
        return scores

    def callback_fn(x, convergence):  # x是所有扰动值的信息向量合并所得向量
        print(best_fitness_val)
        return best_fitness_val < 0.5

    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=1000, popsize=5,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    return attack_result.x

# 实验发现,pad_len为32时没有效果,到64时则可以
def gen_adv_samples(model, fn_list, pad_len=128, pad_percent=0.1, step_size=0.1, thres=0.5):

    max_len = int(model.input.shape[1])  # 模型接受的输入数据的长度

    inp2emb = K.function([model.input]+ [K.learning_phase()], [model.layers[1].output]) # [function] Map sequence to embedding
    embs = [inp2emb(i)[0] for i in range(0,256)]

    log = utils.logger()
    adv_samples = []

    for e, fn in enumerate(fn_list):
        inp, len_list = preprocess([fn], max_len)

        pad_idx = len_list[0]   # 以文件的长度作为填充字节的起始下标

        # 填充字节
        noise = np.zeros(pad_len)
        noise = np.random.randint(0, 255, pad_len)
        inp[0][pad_idx: pad_idx + pad_len] = noise

        inp_emb = np.squeeze(np.array(inp2emb([inp, False])), 0)

        # pad_len = max(min(int(len_list[0]*pad_percent), max_len-pad_idx), 0)
        pad_len = max(min(pad_len, max_len-pad_idx), 0)
        org_score = model.predict(inp)[0][0]    # 模型对未添加噪声的文件的预测概率(1表示恶意)
        loss, pred = float('nan'), float('nan')
        predict_func = functools.partial(model.predict, batch_size=10)

        if pad_len > 0:

            if thres < org_score:
                # adv, gradient, loss = fgsm(model, inp, inp_emb, pad_idx, pad_len, e, step_size)
                adv, gradient, loss = fgsm(model, inp, inp_emb, 2, 4, e, step_size)

                # pad_idx = 0
                # adv, gradient, loss = evade_at_test_time(model, inp, inp_emb, pad_idx, pad_len, embs, step_size, rounds = 100)
                # de_algo = de.DE(inp, model.predict, dim_cnt=2, change_byte_cnt=32, individual_cnt=32 * 2, bounds=[[(pad_idx, pad_idx + 32)], [(0, 255)]])
                # de_algo = de.DE(inp, predict_func, dim_cnt=2, change_byte_cnt=32, individual_cnt=57, bounds=[DOS_HEADER_MODIFY_RANGE, [(0, 255)]], F=0.2)
                # adv = de_algo.update()

                de_attack(model, inp, DOS_HEADER_MODIFY_RANGE[0], change_byte_cnt=4)

                pred = model.predict(adv)[0][0]
                final_adv = adv[0][:pad_idx+pad_len]

            else: # use origin file
                final_adv = inp[0][:pad_idx]


        # log.write(fn, org_score, pad_idx, pad_len, loss, pred)

        # 整数数组转字节序列
        bin_adv = bytes(list(final_adv))
        adv_samples.append(bin_adv)

    return adv_samples, log

