from keras import backend as K, losses
import numpy as np

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