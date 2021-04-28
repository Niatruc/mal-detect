from keras import backend as K, losses
import numpy as np
from sklearn.neighbors import NearestNeighbors


def fgsm(model, inp, embs, modifiable_range_list, step_size=0.1, step=100, thres=0.5):
    inp2emb = K.function([model.input] + [K.learning_phase()], [model.layers[1].output])  # 嵌入层函数

    def init_adv_emb(): # 用随机字节设置inp中的噪声字节, 并返回嵌入张量
        for bound in modifiable_range_list:
            bound_len = bound[1] - bound[0]
            noise = np.random.randint(0, 256, bound_len)
            inp[0][bound[0]: bound[1]] = noise
        inp_emb = inp2emb([inp, False])[0]
        return inp_emb

    out = inp.copy()
    last_score = 1.0

    loss = losses.binary_crossentropy(model.output[:, 0], 0)
    grads = K.gradients(loss, model.layers[1].output)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-8)

    mask_e = np.zeros(model.layers[1].output.shape[1:]) # embedding layer output shape
    for bound in modifiable_range_list:
        mask_e[bound[0] : bound[1]] = 1
    origin_grads = grads
    grads *= K.constant(mask_e) # 只保留末尾附加字节对应的梯度
    iterate = K.function([model.layers[1].output], [model.output, loss, grads, origin_grads])

    # 用K近邻来寻找嵌入向量对应的字节
    neigh = NearestNeighbors(1)
    neigh.fit(embs)

    adv_emb = init_adv_emb()
    for i in range(step):
        model_output, loss_val, grads_val, origin_grads_val = iterate([adv_emb])
        model_output = model_output[0][0]
        print(i, model_output)
        if model_output < thres:
            adv = emb_search(inp, adv_emb[0], modifiable_range_list, neigh)
            score = model.predict(adv)[0][0]
            print("实际分数: ", score)
            if score < last_score:
                out = adv
                last_score = score
                if last_score < thres:
                    break

        if np.all(grads_val==0): # 实验中发现存在梯度全0的情况，遂加上这句判断
            print("修改部分梯度全零")
            adv_emb = init_adv_emb() # 重新初始化噪声字节
        else:
            grads_val *= step_size
            adv_emb -= grads_val

    return out, i


# 用K近邻来寻找嵌入向量对应的字节
def emb_search(org, adv, modifiable_range_list, neigh):
    out = org.copy()
    for bound in modifiable_range_list:
        for idx in range(*bound):
            target = adv[idx].reshape(1, -1)
            best_idx = neigh.kneighbors(target, 1, False)[0][0]
            out[0][idx] = best_idx
    return out
