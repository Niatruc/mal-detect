import numpy as np
import random
import sys
from functools import reduce


class DE:

    def __init__(
            self, inp, target_func, individual_dim_cnt, individual_cnt, bounds, apply_individual_to_adv_func,
            F = 0.5, CR = 0.8, check_convergence_per_iter=100, check_dim_convergence_tolerate_cnt=-1, kick_units_rate=0.5,
            init_units=[], used_init_units_cnt=0,
    ):
        '''
        :param inp: 输入样本
        :param target_func: 目标优化参数
        :param individual_dim_cnt: 个体的维度数
        :param individual_cnt: DE算法中的个体数量(每个个体unit由所有要修改的字节的信息组成)
        :param bounds: 每一个要改的字节包含两个信息:字节位置和更改后的值. bounds包含的两个数组分别是这两个值对应的取值范围,如[[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)]]
        :param F: 变异参数,越大则越能跳出局部最优点,但收敛速度会降低
        :param CR: 交叉概率
        '''
        self.adv = inp.copy()

        self.target_func = target_func
        self.dim_cnt = individual_dim_cnt  # 维度
        self.individual_cnt = individual_cnt  # 个体个数
        self.bounds = bounds # [[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)], ...]
        self.diff_adv = apply_individual_to_adv_func

        self.each_dim_bounds_total_len_list = [] # 计算每一维的取值区间总长度. 有一些维可能有多个不连续的取值区间, 需要将各区间长度相加.
        for dim_bounds in bounds:  # 注意dim_bounds是多个代表取值区间的元组构成的列表, 形如[(1, 2), (5, 7), (9, 12)]
            dim_bounds = [(0, 0)] + dim_bounds
            each_dim_bound_total_len = reduce(lambda x, y: (0, (x[1] - x[0]) + (y[1] - y[0])), dim_bounds)[1]
            self.each_dim_bounds_total_len_list.append(each_dim_bound_total_len)

        self.best_fitness_value = float('Inf')
        self.F = F
        self.CR = CR
        self.best_unit = None  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.check_convergence_per_iter = check_convergence_per_iter
        self.kick_units_rate = kick_units_rate
        self.check_dim_convergence_tolerant_cnt = check_dim_convergence_tolerate_cnt # 如果连续检查的收敛的维度数相同的次数达到此次数,则对不收敛的维度
        self.convergent_dim_cnt_ary = np.zeros((check_dim_convergence_tolerate_cnt, )) - 1
        self.init_units = init_units
        self.used_init_units_cnt = used_init_units_cnt # 每次用init_units里的多少个个体来初始化unit_list
        if used_init_units_cnt > len(self.init_units):
            self.used_init_units_cnt = len(self.init_units)

        self.init_unit_list(0, self.individual_cnt)

        self.calc_best_fitness()
        self.fitness_val_list.append(self.best_fitness_value)

    def init_unit_list(self, from_num, to_num):
        # 对种群进行初始化(每个代表个体的向量的最后一位存其当前适应值)
        if hasattr(self, 'unit_list'):
            self.unit_list[from_num: to_num] = (
               np.random.rand(self.individual_cnt, self.dim_cnt + 1) * [
                   np.array(self.each_dim_bounds_total_len_list + [1]) - 1 for _ in range(self.individual_cnt)
               ]
           )[from_num: to_num]
        else:
            self.unit_list = np.zeros((self.individual_cnt, self.dim_cnt + 1))

        ids = np.arange(len(self.init_units))
        # np.random.shuffle(ids)
        init_units = self.init_units[ids[0: self.used_init_units_cnt]]
        for i, unit in enumerate(init_units):
            unit = self.init_units[i]
            if self.unit_list.shape[1] == unit.shape[0]: # 要确保维度相同才能赋值
                self.unit_list[from_num + i] = unit
        self.mutate_unit_list = self.unit_list[:, 0:-1].copy()
        self.crossover_unit_list = self.unit_list[:, 0:-1].copy()

    # 求种群的初始最优值
    def calc_best_fitness(self):
        all_unit_advs = []
        for unit in self.unit_list:
            adv = self.diff_adv(self.adv, unit[0 : -1])
            all_unit_advs.append(adv)
        fitness_values = self.target_func(np.array(all_unit_advs))

        self.unit_list[:, -1] = fitness_values.reshape(1, -1)

        self.best_fitness_value = np.min(fitness_values)
        self.best_unit = self.unit_list[np.argmin(fitness_values)]

    # 变异
    def mutation_func(self):
        for i in range(self.individual_cnt): # 对每个个体执行操作
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.individual_cnt - 1)  # 随机数范围为[0, individual_cnt-1]的整数
                r2 = random.randint(0, self.individual_cnt - 1)
                r3 = random.randint(0, self.individual_cnt - 1)

            # 计算变异向量
            mutation = self.unit_list[r1][0 : -1] + self.F * (self.unit_list[r2][0 : -1] - self.unit_list[r3][0 : -1])
            # pseudo_mutation = pseudo_mutation.clip(np.zeros(self.dim_cnt), np.array(self.each_dim_bounds_total_len_list) - 1)
            mutation = mutation % np.array(self.each_dim_bounds_total_len_list)
            self.mutate_unit_list[i] = mutation

    # 交叉
    def crossover_func(self):
        rand_j = np.array([[random.randint(0, self.dim_cnt - 1)] for _ in range(self.individual_cnt)])
        ordered_j = np.array([range(self.dim_cnt) for _ in range(self.individual_cnt)])
        rand_float = np.random.rand(self.individual_cnt, self.dim_cnt)
        judges = (rand_float <= self.CR) | (rand_j == ordered_j)
        selected_mutations = (judges * -1) & self.mutate_unit_list.astype(int)
        selected_org_units = (~judges * -1) & self.unit_list[:, 0 : -1].astype(int)
        self.crossover_unit_list = selected_mutations + selected_org_units

    # 选择
    def selection_func(self):
        new_fitness_values = self.calc_all_after_crossover()
        judges = (new_fitness_values.reshape(1, -1) < self.unit_list[:, -1]).reshape(-1)
        self.unit_list = np.array([
            np.hstack((self.crossover_unit_list[i], new_fitness_values[i])) if j else self.unit_list[i]
            for i, j in enumerate(judges)
        ])
        self.best_fitness_value = np.min(self.unit_list[:, -1])
        self.best_unit = self.unit_list[np.argmin(self.unit_list[:, -1])]

    # 计算交叉后的新个体的适应值. 个体数很多的时候会算得很慢, 甚至卡在这里
    def calc_all_after_crossover(self):
        advs = []
        for crossover_unit in self.crossover_unit_list:
            adv = self.diff_adv(self.adv, crossover_unit)
            advs.append(adv)
        scores = self.target_func(np.array(advs))
        return scores

    def print_all_units_vector(self):
        for unit in self.unit_list:
            print(unit.vector)

    # 判断种群中所有个体的某一维是否收敛了
    def check_dim_convergence(self):
        unit_vectors = self.unit_list[:, 0 : -1]
        unit_vector_dim_values = unit_vectors.transpose()
        return [True if 1 == len(np.unique(unit_vector_dim_value)) else False for unit_vector_dim_value in unit_vector_dim_values]

    def update(self, fitness_value_threshold=0.5, iter_cnt=-1, use_kick_mutation=True):
        cnt = 0
        iter_sum = 0
        has_check_convergence_cnt = -1
        while True:
            # 每隔一定代数, 检查是否有某些维已经收敛. 对这些随机取一些个体, 对它们在这些维上重新取随机值(刺激性的突变), 并算一次适应值
            if iter_sum % self.check_convergence_per_iter == 0:
                dim_convergence = self.check_dim_convergence()
                if True in dim_convergence:
                    convergent_dims = []
                    for i, is_convergent in enumerate(dim_convergence):
                        if is_convergent:
                            convergent_dims.append(i)
                    print("有收敛的维度为: ", convergent_dims)
                    print("总收敛的维度数量为: ", len(convergent_dims))

                    check_dim_convergence_beyond_tolerance = False
                    self.convergent_dim_cnt_ary[has_check_convergence_cnt % self.check_dim_convergence_tolerant_cnt] = len(convergent_dims)
                    if len(np.unique(self.convergent_dim_cnt_ary)) <= 1:
                        print("连续%d次无新的维度收敛" % self.check_dim_convergence_tolerant_cnt)
                        check_dim_convergence_beyond_tolerance = True

                    population_convergent = False
                    if len(convergent_dims) >= len(self.bounds): # and not use_kick_mutation:
                        population_convergent = True
                        print("所有维度收敛(种群收敛)")
                        # break

                    if use_kick_mutation and (population_convergent or check_dim_convergence_beyond_tolerance):
                        print("进行刺激性的突变")
                        kick_units_cnt = np.maximum(int(self.individual_cnt * self.kick_units_rate), 1)
                        # has_preserve_one = False
                        self.init_unit_list(1, kick_units_cnt)
                self.calc_best_fitness()
                self.fitness_val_list.append(self.best_fitness_value)
            has_check_convergence_cnt += 1

            self.mutation_func()
            self.crossover_func()
            self.selection_func()
            self.fitness_val_list.append(self.best_fitness_value)
            iter_sum += 1
            print(iter_sum, ' ', self.best_fitness_value)
            sys.stdout.flush()

            if iter_cnt > -1:
                cnt += 1
                if cnt >= iter_cnt:
                    break
            if self.best_fitness_value < fitness_value_threshold:
                break

        final_adv = self.diff_adv(self.adv, self.best_unit)
        return np.array([final_adv]), iter_sum, self.best_unit

