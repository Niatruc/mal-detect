import numpy as np
import random
import sys
from functools import reduce

class DE:
    class Unit:
        def __init__(self, bounds, dim_cnt, each_dim_bounds_total_len_list, each_dim_bounds_acc_len_lists):
            '''
            :param bounds: # 个体每一维的取值区间, 形如:[[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)], ...]
            :param dim_cnt: 个体有多少维
            '''
            self.bounds = bounds
            self.dim_cnt = dim_cnt
            self.each_dim_bounds_total_len_list = each_dim_bounds_total_len_list
            self.each_dim_bounds_acc_len_lists = each_dim_bounds_acc_len_lists
            self.vector = np.empty(shape=(dim_cnt,), dtype=int)
            self.fitness_value = float('inf')
            for i in range(dim_cnt): # 为每一维取随机数
                self.vector[i] = self.get_rand_value_for_ith_dim(i)

            self.vector_after_mutation = self.vector.copy()
            self.vector_after_crossover = self.vector.copy()

        # 为第i维取随机值
        def get_rand_value_for_ith_dim(self, i):
            bound = self.bounds[i]
            in_which_interval = random.randint(0, len(bound) - 1) # 先随机取一个取值区间, 然后再在这个取值区间内随机取值
            b_min, b_max = bound[in_which_interval]
            return random.randint(b_min, b_max - 1) # 右边界开放. 比如范围是(1, 3)时, 则取值只能是1或2

        # 将vector根据各维度的取值区间转为pseudo_vector
        def get_pseudo_vector(self):
            self.pseudo_vector = self.vector.copy()
            for i, v in enumerate(self.vector):     # 对向量的每一维执行操作
                if len(self.bounds[i]) <= 1: # 对于取值空间为连续单一空间的维度, 维持原值不变
                    continue
                else: # 有多个取值区间时要搜索所在区间, 并转换值
                    for j, bound in enumerate(self.bounds[i]):
                        if v in range(*bound):
                            break
                    v_offset_in_bound = v - bound[0] # v减去bound的左边界可得其在所处取值区间的位置
                    pseudo_v = ([0] + self.each_dim_bounds_acc_len_lists[i])[j] + v_offset_in_bound
                    self.pseudo_vector[i] = pseudo_v

            return self.pseudo_vector

        # 由pseudo_vector得到可用的vector
        def __pseudo_vector_to_vector(self, pseudo_vector):
            vector = pseudo_vector.copy()
            for i, v in enumerate(pseudo_vector):     # 对向量的每一维执行操作
                if len(self.bounds[i]) <= 1:
                    continue
                else:
                    v = int(v)
                    acc_len_list = self.each_dim_bounds_acc_len_lists[i]
                    j, k = 0, len(acc_len_list) - 1
                    while j != k - 1:
                        m = (j + k) // 2
                        if v < acc_len_list[m]:
                            k = m
                        else:
                            j = m

                    begin_val = acc_len_list[j]
                    if v < acc_len_list[0]: # 边界情形, 当v值处于第一个区间
                        k = 0
                        begin_val = 0
                    v = self.bounds[i][k][0] + (v - begin_val)
                    vector[i] = v
            return vector

        # 检查value是否在第i维数据要求的范围内
        def check_ith_dim_in_bound(self, i, value):
            for bound in self.bounds[i]:
                b_min, b_max = bound
                if b_min <= value < b_max: # 右边界开放. 比如范围是(1, 3)时, 则取值只能是1或2
                    return True
            return False

        def set_mutation(self, mutation_vector):
            for i, dim_value in enumerate(mutation_vector):
                value = int(dim_value) # 先转成整数再做边界检查
                if self.check_ith_dim_in_bound(i, value): # 边界检查
                    self.vector_after_mutation[i] = value
                else:
                    self.vector_after_mutation[i] = self.get_rand_value_for_ith_dim(i)

        # 使用伪向量设置mutation向量
        def set_mutation_by_pseudo_mutation(self, pseudo_mutation):
            for i, dim_value in enumerate(pseudo_mutation):
                value = int(dim_value) # 先转成整数再做边界检查
                if len(self.bounds[i]) <= 1:
                    if self.check_ith_dim_in_bound(i, value):  # 边界检查
                        pseudo_mutation[i] = value
                    else:
                        pseudo_mutation[i] = self.get_rand_value_for_ith_dim(i)
                    continue

                if value in range(0, self.each_dim_bounds_total_len_list[i]): # 边界检查
                    pseudo_mutation[i] = value
                else:
                    pseudo_mutation[i] = random.randint(0, self.each_dim_bounds_total_len_list[i] - 1)
            self.vector_after_mutation = self.__pseudo_vector_to_vector(pseudo_mutation)

    def __init__(self, inp, target_func, dim_cnt, changed_bytes_cnt, individual_cnt, bounds, F = 0.5, CR = 0.8, check_convergence_per_iter=100, kick_units_rate=0.5):
        '''
        :param inp: 输入样本
        :param target_func: 目标优化参数
        :param dim_cnt: 值为2,表示每一个要改的字节包含两个信息:字节位置和更改后的值
        :param changed_bytes_cnt: 要修改的字节的数量
        :param individual_cnt: DE算法中的个体数量(每个个体unit由所有要修改的字节的信息组成)
        :param bounds: 每一个要改的字节包含两个信息:字节位置和更改后的值. bounds包含的两个数组分别是这两个值对应的取值范围,如[[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)]]
        :param F: 变异参数,越大则越能跳出局部最优点,但收敛速度会降低
        :param CR: 交叉概率
        '''
        self.adv = inp.copy()

        self.target_func = target_func
        self.dim_cnt = dim_cnt  # 维度
        self.individual_cnt = individual_cnt  # 个体个数
        self.bounds = bounds * changed_bytes_cnt # [[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)], ...]

        # 计算每一维的取值区间总长度. 有一些维可能有多个不连续的取值区间, 需要将各区间长度相加.
        self.each_dim_bounds_total_len_list = []
        self.each_dim_bounds_acc_len_lists = []  # 对于每个取值区间,保存其左边所有取值区间的长度的累加值(也加上自身的区间长度)
        for dim_bounds in bounds:  # 注意dim_bounds是多个代表取值区间的元组构成的列表, 形如[(1, 2), (5, 7), (9, 12)]
            if len(dim_bounds) > 1:
                dim_bounds = [(0, 0)] + dim_bounds
                each_dim_bound_total_len = reduce(lambda x, y: (0, (x[1] - x[0]) + (y[1] - y[0])), dim_bounds)[1]
                dim_bounds_acc_len = [reduce(lambda x, y: (0, (x[1] - x[0]) + (y[1] - y[0])), dim_bounds[0:i + 1])[1]
                                      for i in range(len(dim_bounds))]
                dim_bounds_acc_len.pop(0)

            else:
                each_dim_bound_total_len = dim_bounds[0][1] - dim_bounds[0][0]  # 若当前维只有一个取值区间,则直接大减小
                dim_bounds_acc_len = [dim_bounds[0][1] - dim_bounds[0][0]]
            self.each_dim_bounds_total_len_list.append(each_dim_bound_total_len)
            self.each_dim_bounds_acc_len_lists.append(dim_bounds_acc_len)
        self.each_dim_bounds_total_len_list *= changed_bytes_cnt
        self.each_dim_bounds_acc_len_lists *= changed_bytes_cnt

        self.best_fitness_value = float('Inf')
        self.F = F
        self.CR = CR
        self.best_unit = None  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.check_convergence_per_iter = check_convergence_per_iter
        self.kick_units_rate = kick_units_rate

        # 对种群进行初始化
        self.unit_list = [self.Unit(
            self.bounds, dim_cnt * changed_bytes_cnt, self.each_dim_bounds_total_len_list, self.each_dim_bounds_acc_len_lists
        ) for i in range(self.individual_cnt)]

        self.calc_best_fitness()
        self.fitness_val_list.append(self.best_fitness_value)

    # 求种群的初始最优值
    def calc_best_fitness(self):
        all_unit_advs = []
        for unit in self.unit_list:
            adv = self.diff_adv(unit.vector)
            all_unit_advs.append(adv)
        fitness_values = self.target_func(np.array(all_unit_advs))

        for i, fitness_value in enumerate(fitness_values):
            self.unit_list[i].fitness_value = fitness_value[0]

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
            # mutation = self.unit_list[r1].vector + self.F * (self.unit_list[r2].vector - self.unit_list[r3].vector)
            # self.unit_list[i].set_mutation(mutation)

            pseudo_mutation = self.unit_list[r1].get_pseudo_vector() + self.F * (self.unit_list[r2].get_pseudo_vector() - self.unit_list[r3].get_pseudo_vector())
            self.unit_list[i].set_mutation_by_pseudo_mutation(pseudo_mutation)

    # 交叉
    def crossover_func(self):
        for unit in self.unit_list:
            for j in range(unit.dim_cnt): # 对个体的每一维进行操作
                rand_j = random.randint(0, self.dim_cnt - 1)
                rand_float = random.random()
                if rand_float <= self.CR or rand_j == j:
                    unit.vector_after_crossover[j] = unit.vector_after_mutation[j]
                else:
                    unit.vector_after_crossover[j] = unit.vector[j]

    # 选择
    def selection_func(self):
        new_fitness_values = self.calc_all_after_crossover()
        for i, unit in enumerate(self.unit_list): # 对每一个个体, 对比各自的交叉后值, 作优胜劣汰
            new_fitness_value = new_fitness_values[i][0]
            if new_fitness_value < unit.fitness_value: # 若新的个体的适应值更小,则更新个体
                unit.fitness_value = new_fitness_value
                unit.vector = unit.vector_after_crossover.copy()
            if new_fitness_value < self.best_fitness_value: # 顺道更新全局最优个体
                self.best_fitness_value = new_fitness_value
                self.best_unit = unit

    # 计算修改后的adv
    def diff_adv(self, diff_vector):
        adv = self.adv.copy()[0]

        i = 0
        while i < len(diff_vector):
            pos = int(diff_vector[i])
            val = int(diff_vector[i + 1])
            adv[pos] = val
            i += 2
        return adv

    # 计算交叉后的新个体的适应值. 个体数很多的时候会算得很慢, 甚至卡在这里
    def calc_all_after_crossover(self):
        advs = []
        for unit in self.unit_list:
            diff_vector = unit.vector_after_crossover
            adv = self.diff_adv(diff_vector)
            advs.append(adv)
        scores = self.target_func(np.array(advs))
        return scores

    def print_all_units_vector(self):
        for unit in self.unit_list:
            print(unit.vector)

    # 判断种群中所有个体的某一维是否收敛了
    def check_dim_convergence(self):
        unit_vectors = np.array([unit.vector for unit in self.unit_list])
        unit_vector_dim_values = unit_vectors.transpose()
        return [True if 1 == len(np.unique(unit_vector_dim_value)) else False for unit_vector_dim_value in unit_vector_dim_values]

    def update(self, fitness_value_threshold=0.5, iter_cnt=-1, use_kick_mutation=True):
        cnt = 0
        iter_sum = 0
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

                    if len(convergent_dims) == len(self.bounds) and not use_kick_mutation:
                        print("所有维度收敛")
                        break

                    if use_kick_mutation:
                        print("进行刺激性的突变")
                        kick_units_cnt = int(self.individual_cnt * self.kick_units_rate)
                        unit_nums = list(range(self.individual_cnt))
                        random.shuffle(unit_nums)
                        for unit_num in unit_nums[0:kick_units_cnt]:
                            unit = self.unit_list[unit_num]
                            if unit != self.best_unit:  # 原有的最佳个体不要改
                                for dim in convergent_dims:
                                    unit.vector[dim] = unit.get_rand_value_for_ith_dim(dim)
                self.calc_best_fitness()
                self.fitness_val_list.append(self.best_fitness_value)

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

        final_adv = self.diff_adv(self.best_unit.vector)
        return np.array([final_adv]), iter_sum

