import numpy as np
import math
import random
import os
from functools import reduce
import functools


class GWO:
    def __init__(self, inp, target_func, apply_individual_to_adv_func, dim_cnt, bounds, pack_size=15, init_wolves=None):
        self.adv = inp.copy()
        self.target_func = target_func
        self.dim_cnt = dim_cnt
        self.bounds = bounds
        self.diff_adv = apply_individual_to_adv_func
        self.pack_size = pack_size

        # 计算每一维的取值区间总长度
        self.each_dim_bounds_total_len_list = []
        for dim_bounds in bounds:  # 注意dim_bounds是多个代表取值区间的元组构成的列表, 形如[(1, 2), (5, 7), (9, 12)]
            dim_bounds = [(0, 0)] + dim_bounds
            each_dim_bound_total_len = reduce(lambda x, y: (0, (x[1] - x[0]) + (y[1] - y[0])), dim_bounds)[1]
            self.each_dim_bounds_total_len_list.append(each_dim_bound_total_len)
        self.each_dim_bounds_total_len_ary = np.array(self.each_dim_bounds_total_len_list)

        self.alpha = self.init_wolves_position(1)[0]
        self.beta = self.init_wolves_position(1)[0]
        self.delta = self.init_wolves_position(1)[0]
        self.omigas = self.init_wolves_position(self.pack_size)  # 每个个体向量的最后一个元素保存适应值

        if init_wolves is not None:
            try:
                self.alpha = init_wolves[0]
                self.beta = init_wolves[1]
                self.delta = init_wolves[2]

                min_size = np.minimum(len(init_wolves) - 3, self.pack_size)
                self.omigas = init_wolves[min_size]
            except Exception as e:
                pass

    def init_wolves_position(self, wolves_cnt):
        wolf_position = np.zeros((wolves_cnt, self.dim_cnt + 1))
        for i in range(0, wolves_cnt):
            for j in range(self.dim_cnt):
                bound_len = self.each_dim_bounds_total_len_ary[j]
                wolf_position[i, j] = random.randint(0, bound_len)
        advs = np.array([self.diff_adv(self.adv, wolf_position[0: -1]) for wolf_position in wolf_position])
        fitness_values = self.target_func(advs)
        wolf_position[:, -1] = fitness_values.transpose() # 每个个体向量的最后一个元素保存适应值
        return wolf_position

    # 与其余狼对比,并更新头三头狼
    def update_pack(self):
        updated_omigas_first_three = self.omigas[self.omigas[:, -1].argsort()]
        for omiga in updated_omigas_first_three:
            if (omiga[-1] < self.alpha[-1]):
                self.alpha = np.copy(omiga)
            if (omiga[-1] > self.alpha[-1] and omiga[-1] < self.beta[-1]):
                self.beta = np.copy(omiga)
            if (omiga[-1] > self.alpha[-1] and omiga[-1] > self.beta[-1] and omiga[-1] < self.delta[-1]):
                self.delta = np.copy(omiga)

    # 求wolf_pos狼向target_wolf_pos移近之后的位置向量
    def calc_move_component_vector(self, wolf_pos, target_wolf_pos, a_linear_component):
        # r1 = np.random.rand()    # n(self.dim_cnt)
        # r2 = np.random.rand()    # n(self.dim_cnt)
        # r1 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        # r2 = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        r1 = np.array([int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1) for _ in range(self.dim_cnt)])
        r2 = np.array([int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1) for _ in range(self.dim_cnt)])
        A = 2 * a_linear_component * r1 - a_linear_component
        C = 2 * r2
        D = C * target_wolf_pos[0 : -1] - wolf_pos[0 : -1] # omiga狼向目标狼移动的方向向量
        X = target_wolf_pos[0 : -1] - A * D
        return X

    # 更新底层omiga狼的位置
    def update_position(self,a_linear_component=2):
        for omiga in self.omigas:  # 对每一头底层狼进行操作
            X1 = self.calc_move_component_vector(omiga, self.alpha, a_linear_component)
            X2 = self.calc_move_component_vector(omiga, self.beta, a_linear_component)
            X3 = self.calc_move_component_vector(omiga, self.delta, a_linear_component)

            # omiga[0 : -1] = ((X1 + X2 + X3) / 3) % self.each_dim_bounds_total_len_ary
            omiga[0 : -1] = ((X1 + X2 + X3) / 3).clip(np.zeros(self.dim_cnt), self.each_dim_bounds_total_len_ary)
            adv = self.diff_adv(self.adv, omiga[0 : -1])
            fitness_val = self.target_func(np.array([adv]))[0][0]
            omiga[-1] = fitness_val
        print(self.omigas[:, -1])

            # GWO Function
    def optimize(self, iterations=50):
        iter_cnt = 0
        while (iter_cnt <= iterations):
            print("Iteration = ", iter_cnt, " f(x) = ", self.alpha[-1])
            a_linear_component = 2 - iter_cnt * (2 / iterations)
            self.update_pack()
            self.update_position(a_linear_component=a_linear_component)
            iter_cnt = iter_cnt + 1
        adv = self.diff_adv(self.adv, self.alpha[0: -1])
        return np.array([adv]), self.alpha


# ===============================================================================应用测试

# # Function to be Minimized (Six Hump Camel Back). Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values=[0, 0]):
    func_value = 4 * variables_values[0] ** 2 - 2.1 * variables_values[0] ** 4 + (1 / 3) * variables_values[0] ** 6 + \
                 variables_values[0] * variables_values[1] - 4 * variables_values[1] ** 2 + 4 * variables_values[1] ** 4
    return func_value

def x_2(v):
    return np.square(v).sum()

def diff_adv(adv, diff_vec):
    return diff_vec

def target_func(variables_values=[[0, 0]]):
    res = []
    for v in variables_values:
        fv = x_2(v - 5)
        res.append([fv])

    return np.array(res)

# dim_cnt = 256
# gwo_algo = GWO(
#     np.array([]), target_func, diff_adv, dim_cnt=dim_cnt,
#     bounds=[[(-5, 5)]] * dim_cnt, pack_size=15,
# )
#
# adv, alpha = gwo_algo.optimize(iterations=1000)
# print(alpha - np.array([5, 5, 0]))
