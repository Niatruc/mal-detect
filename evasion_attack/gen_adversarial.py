import os
import numpy as np
from keras import backend as K
# import utils
from mal_detect.utils import exe_util
from . import de, fgsm2, evade_at_test_time, gwo
from mal_detect.utils.file_util import preprocess
import functools
import struct

# 实验发现,pad_len为32时没有效果,到64时则可以
def gen_adv_samples(
        model, fn_list, predict_func=None, max_len=2**20,
        strategy=0, changed_bytes_cnt=16, thres=0.5, batch_size=10, workers=1,
        *, step_size=0.1, max_iter=1000,
        de_F=0.2, individual_cnt=10,
        change_range=0b1111, use_kick_mutation=True, kick_units_rate=1.,
        check_convergence_per_iter=100, sub_strategy=0, check_dim_convergence_tolerate_cnt = 3,
        save_units=False,save_units_path="units", save_as_init_unit_when_below_thres=False, save_units_with_lower_itersum=100,
        init_units=None, init_units_upper_amount=15, used_init_units_cnt=5,use_increasing_units=False,
        pre_modify_file_func=None,
):
    # max_len = int(model.input.shape[1])  # 模型接受的输入数据的长度

    if strategy == 0 or strategy == 1:
        inp2emb = K.function([model.input]+ [K.learning_phase()], [model.layers[1].output]) # 嵌入层函数
        embs = [inp2emb([i])[0] for i in range(0,256)] # 求0~255各数字对应的嵌入向量
        # embs = model.layers[1].get_weights()[0][0:256];  # shape: (257, 8)

    # log = utils.Logger()
    adv_samples = []
    test_info = {}

    if predict_func is None:
        predict_func = functools.partial(model.predict, batch_size=batch_size)

    units = np.array([]) # 这个数组存储新产生的个体
    new_unit_itersum_pairs = [] # 把每个成功样本的unit和对应的迭代数存起来. 优先存迭代数多的样本对应的unit
    if save_units_path and os.path.exists(save_units_path + '.npy'):
        units = np.load(save_units_path + '.npy')
        if os.path.exists(save_units_path + '_withIterSum.npy'):
            new_unit_itersum_pairs = np.load(save_units_path + '_withIterSum.npy')
            print("已保存%d个新的优良个体." % len(new_unit_itersum_pairs))
            if len(new_unit_itersum_pairs) > 0:
                print("它们对应的迭代数: ", new_unit_itersum_pairs[0:20, -1])
            new_unit_itersum_pairs = list(new_unit_itersum_pairs)

    if init_units is None:
        init_units = np.array([])

    # new_unit_upper_amount = init_units_upper_amount - len(init_units)
    # new_unit_upper_amount = new_unit_upper_amount if new_unit_upper_amount > 0 else 0
    # print("额外导入的新优良个体有%d个." % new_unit_upper_amount)

    init_units_1 = init_units
    if use_increasing_units: # and init_units.shape[1] == units.shape[1]:
        try:
            if len(init_units) <= 0 and len(units) > 0: # 如果init_units是空的而units非空
                init_units_1 = units
            elif len(init_units) > 0 and len(units) > 0: # 如果二者皆非空
                init_units_1 = np.concatenate((units, init_units))[0:init_units_upper_amount]

        except Exception as e:
            print(e)

    for e, fn in enumerate(fn_list):
        print("文件: " + fn)
        inp, len_list = preprocess([fn], max_len)
        pad_idx = len_list[0]   # 以文件的长度作为填充字节的起始下标
        org_score = predict_func(inp)[0]    # 模型对未添加噪声的文件的预测概率(1表示恶意)
        test_info['org_score'] = org_score
        print("原始预测分数: ", org_score)
        if org_score < thres:
            test_info['final_score'] = org_score[0]
            test_info['iter_sum'] = 0
            print("最终置信度: ", org_score)
            continue

        modifiable_range_list = []
        pad_len = max(min(changed_bytes_cnt, max_len - pad_idx), 0)
        if pre_modify_file_func is not None:
            modifiable_range_list, inp = pre_modify_file_func(fn)
            inp = np.array(inp, dtype=np.int32)
        elif change_range == 0 and pad_len > 0:
            modifiable_range_list = [(pad_idx, pad_idx + pad_len)]
        else: # 从可改的第一个字节开始到第changed_bytes_cnt个字节结束
            modifiable_range_list = exe_util.get_modifiable_range_list(fn, change_range, changed_bytes_cnt)

        modifiable_bytes_pos_list = [] # 存储所有可改字节的位置
        for bound in modifiable_range_list:
            bound_len = bound[1] - bound[0]
            # noise = np.zeros(bound_len)
            noise = np.random.randint(0, 256, bound_len)
            inp[0][bound[0]: bound[1]] = noise # struct.unpack("B"*7, b"abcdefg")

            for pos in range(*bound):
                modifiable_bytes_pos_list.append(pos)
        modifiable_bytes_pos_ary = np.array(modifiable_bytes_pos_list)
        changed_bytes_cnt = len(modifiable_bytes_pos_ary) # 可改的长度可能不够,所以需要调整
        print("可修改的字节数为: %d" % changed_bytes_cnt)

        if strategy == 0 or strategy == 1: # 采用基于梯度的白盒攻击方法
            if len(modifiable_range_list) > 0:
                inp_emb = np.squeeze(np.array(inp2emb([inp, False])), 0)

                if strategy == 0:
                    adv, iter_sum = fgsm2.fgsm(model, inp, embs, modifiable_range_list, step_size, max_iter, thres)
                elif strategy == 1:
                    adv, iter_sum = evade_at_test_time.evade_at_test_time(model, inp, inp_emb, modifiable_range_list, embs, step_size, thres, max_iter)
                final_adv = adv[0] # [:pad_idx + pad_len]
                test_info['iter_sum'] = iter_sum
            else:  # 使用原始文件
                final_adv = inp[0][:pad_idx]
        elif strategy >= 2:
            # diff_adv函数用于计算修改后的adv
            if sub_strategy == 0:
                individual_dim_bounds = [[(0, 256)]] * changed_bytes_cnt
                individual_dim_cnt = changed_bytes_cnt

                def diff_adv(adv, diff_vector):
                    adv1 = adv.copy()[0]
                    for i in range(changed_bytes_cnt):
                        val = int(diff_vector[i])
                        pos = modifiable_bytes_pos_ary[i]
                        adv1[pos] = val
                    return adv1
            elif sub_strategy == 1:
                def diff_adv(adv, diff_vector):
                    adv1 = adv.copy()[0]

                    i = 0
                    while i < len(diff_vector):
                        pos = int(diff_vector[i])
                        val = int(diff_vector[i + 1])
                        adv1[pos] = val
                        i += 2
                    return adv1

            if strategy == 2:
                de_algo = de.DE(
                    inp, predict_func, individual_dim_cnt=individual_dim_cnt, individual_cnt=individual_cnt,
                    bounds=individual_dim_bounds, F=de_F, kick_units_rate=kick_units_rate,
                    check_convergence_per_iter=check_convergence_per_iter,
                    check_dim_convergence_tolerate_cnt=check_dim_convergence_tolerate_cnt,
                    apply_individual_to_adv_func=diff_adv,
                    init_units=init_units_1, used_init_units_cnt=used_init_units_cnt,
                )
                adv, iter_sum, unit = de_algo.update(iter_cnt=max_iter, fitness_value_threshold=thres, use_kick_mutation=use_kick_mutation)
                if len(units) <= 0 or (len(units) > 0 and unit.shape == units[0].shape):
                    units = units.tolist()
                    units.append(unit)
                    units = np.array(units)

                    if use_increasing_units:
                        if not save_as_init_unit_when_below_thres or (save_as_init_unit_when_below_thres and unit[-1] <= thres):
                            if iter_sum >= save_units_with_lower_itersum: # 主要是为了防止那些迭代一次就成功的样本对应的unit也混进来
                                new_unit_itersum_pairs.append(np.append(unit, iter_sum))
                                new_unit_itersum_pairs.sort(key=lambda x: x[-1], reverse=True)
                                # new_unit_itersum_pairs = new_unit_itersum_pairs[0: new_unit_upper_amount] # 优先存迭代次数较大的样本对应的unit
                                # init_units = np.concatenate((init_units, np.array([unit])))
                                new_unit = np.array(new_unit_itersum_pairs)[:, 0:-1]
                                init_units_1 = np.append(init_units, new_unit).reshape(-1, new_unit.shape[1])
                        # 是否需要在适应值低于阈值时才保存unit ( unit[-1] <= thres 确保最终分数低于阈值)
                test_info['iter_sum'] = iter_sum
            elif strategy == 3:
                gwo_algo = gwo.GWO(
                    inp, predict_func, diff_adv, dim_cnt=individual_dim_cnt,
                    bounds=individual_dim_bounds, pack_size=individual_cnt,
                )
                adv, alpha = gwo_algo.optimize(iterations=max_iter)

            final_adv = adv[0]

        # pred = model.predict(adv)[0][0]
        pred = predict_func(adv).reshape(1,)[0]
        test_info['final_score'] = pred
        print("最终置信度: ", pred)
        # log.write(fn, org_score, pad_idx, pad_len, loss, pred)

        # 整数数组转字节序列
        bin_adv = bytes(list(final_adv))
        adv_samples.append(bin_adv)

        if save_units:
            np.save(save_units_path, units)
            np.save(save_units_path + '_withIterSum', new_unit_itersum_pairs)

    return adv_samples, test_info
