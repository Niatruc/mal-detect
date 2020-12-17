from keras import backend as K, losses
import numpy as np

def evade_at_test_time(model, inp, inp_emb, modifiable_range_list, embs, step_size=0.1, stop_threshold=0.5, rounds=10):
    adv = inp.copy()
    adv_emb = inp_emb.copy()

    loss = losses.binary_crossentropy(model.output[:, 0], 1)
    grads = K.gradients(loss, model.layers[1].output)[0]
    grads_sign = K.sign(grads)     # 梯度上升方向的向量(各个方向长为1)

    a = [1., 1., 1., 1., 1., 1., 1., 1., ]
    a = a / np.sqrt(np.sum(np.square(a)))  # 单位向量

    mask = np.zeros(model.layers[1].output.shape[1])
    for bound in modifiable_range_list:
        mask[range(*bound)] = 1

    dire_vec = grads_sign / (K.sqrt(K.sum(K.square(grads_sign))) + 1e-8) # 梯度上升方向的单位向量
    # dire_vec = grads_sign * a

    iterate = K.function([model.layers[1].output], [model.output, loss, grads, grads_sign, dire_vec])

    has_succeed = False
    for r in range(rounds):
        if has_succeed:
            break

        model_output, loss_val, grads_val, grads_sign_val, dire_vec_val = iterate(adv_emb)
        used_dire_vec_val = mask.reshape(mask.shape[0], 1) * dire_vec_val[0]
        if np.all(used_dire_vec_val == 0):
            print("出现全零梯度")
            break

        for bound in modifiable_range_list:
            for j in range(*bound):    # 对每个修改字节执行操作
                emb_j = adv_emb[0][j] # 第j个修改字节的嵌入形式
                dire_vec_val_j = dire_vec_val[0][j]  # 得到第j个修改字节的嵌入向量
                # if np.all(dire_vec_val_j == 0):
                #     dire_vec_val_j = a
                # d = float('inf')
                # for i in range(len(embs)):  # 遍历0~255所有256个字节
                #     emb_i = embs[i]
                #
                #     # 求将emb_j移动到<emb_i在梯度向量的投影点>需要的距离
                #     s_i = (emb_i - emb_j).dot(dire_vec_val_j)
                #
                #     # (emb_j + s_i * dire_vec_val_j)得到<emb_i在梯度向量的投影点>对应的向量
                #     d_i = np.linalg.norm(emb_i - (emb_j + np.dot(s_i, dire_vec_val_j))) # np.linalg.norm这个函数用来求范数(默认是二范数)
                #
                #     if s_i > 0 and d_i <= d:    # s_i大于0是为了确保干扰点顺着梯度上升方向移动
                #         adv[0][j] = i
                #         adv_emb[0][j] = emb_i
                #         d = d_i

                s = (embs - emb_j).dot(dire_vec_val_j)
                s = s.clip(0, 1)
                s[s > 0] = 1

                d = np.linalg.norm(embs - (emb_j + s.reshape(256, 1) * dire_vec_val_j), axis=1)

                s_d = s * d
                try:
                    min_d_idx = np.where(s_d > 0)[0][s_d[s_d > 0].argmin()]
                    adv[0][j] = min_d_idx
                    adv_emb[0][j] = embs[min_d_idx]
                except Exception as e:
                    pass

        adv_score = model.predict(adv)[0][0]
        print("%d %f" % (r, adv_score))
        if adv_score <= stop_threshold:
            has_succeed = True
            break

    return adv, r