from keras import backend as K, losses
import numpy as np
from sklearn.neighbors import NearestNeighbors


def fgsm(model, inp,  inp_emb, modifiable_range_list, step_size=0.1, step=100, thres=0.5):
    adv_emb = inp_emb.copy()
    # loss = K.mean(model.output[:, 0])
    loss = losses.binary_crossentropy(model.output[:, 0], 1)
    grads = K.gradients(loss, model.layers[1].output)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-8)

    mask = np.zeros(model.layers[1].output.shape[1:]) # embedding layer output shape
    for bound in modifiable_range_list:
        mask[bound[0] : bound[1]] = 1
        # mask[pad_idx:pad_idx+pad_len] = 1
    origin_grads = grads
    grads *= K.constant(mask) # 只保留末尾附加字节对应的梯度

    iterate = K.function([model.layers[1].output], [model.output, loss, grads, origin_grads])
    g = 0.
    # step = int(1/step_size)*10
    for i in range(step):
        model_output, loss_val, grads_val, origin_grads_val = iterate([adv_emb])
        grads_val *= step_size
        grads_info = ""
        if np.all(grads_val==0): # 实验中发现存在梯度全0的情况，遂加上这句判断
            grads_info = " 修改部分梯度全零"
            grads_val = np.array([step_size*mask])
        g += grads_val
        adv_emb += grads_val
        # adv += K.sign(grads_value) * 0.1
        # adv += 0.1*mask
        print (i, model_output, grads_info)
        # if loss_val >= 0.9:
        if model_output <= thres:
            break

    # 用K近邻来寻找嵌入向量对应的字节
    emb_layer = model.layers[1]  # 嵌入层
    emb_weight = emb_layer.get_weights()[0];  # shape: (257, 8)
    neigh = NearestNeighbors(1)
    neigh.fit(emb_weight)
    out = emb_search(inp, adv_emb[0], modifiable_range_list, neigh)

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
