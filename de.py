import numpy as np
import random

class DE:
    class Unit:
        def __init__(self, bounds, dim_cnt):
            '''
            :param bounds: # 个体每一维的取值区间, 形如:[[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)], ...]
            :param dim_cnt: 个体有多少维
            '''
            self.bounds = bounds
            self.dim_cnt = dim_cnt
            self.vector = np.empty(shape=(dim_cnt,), dtype=int)
            for i in range(dim_cnt): # 为每一维取随机数
                self.vector[i] = self.get_rand_value_for_ith_dim(i)

            self.vector_after_mutation = self.vector.copy()
            self.vector_after_crossover = self.vector.copy()

        def get_rand_value_for_ith_dim(self, i):
            bound = self.bounds[i]
            in_which_interval = random.randint(0, len(bound) - 1)
            b_min, b_max = bound[in_which_interval]
            return random.randint(b_min, b_max)

        # 检查value是否在第i维数据要求的范围内
        def check_ith_dim_in_bound(self, i, value):
            for bound in self.bounds[i]:
                b_min, b_max = bound
                if b_min <= value <= b_max:
                    return True
            return False

        def set_mutation(self, mutation_vector):
            for i, dim_value in enumerate(mutation_vector):
                value = int(dim_value)
                if self.check_ith_dim_in_bound(i, value): # 边界检查
                    self.vector_after_mutation[i] = value
                else:
                    self.vector_after_mutation[i] = self.get_rand_value_for_ith_dim(i)

    def __init__(self, inp, target_func, dim_cnt, change_byte_cnt, individual_cnt, bounds, F = 0.5, CR = 0.8):
        '''
        :param inp: 输入样本
        :param target_func: 目标优化参数
        :param dim_cnt: 值为2,表示每一个要改的字节包含两个信息:字节位置和更改后的值
        :param change_byte_cnt: 要修改的字节的数量
        :param individual_cnt: DE算法中的个体数量(每个个体unit由所有要修改的字节的信息组成)
        :param bounds: 每一个要改的字节包含两个信息:字节位置和更改后的值. bounds包含的两个数组分别是这两个值对应的取值范围,如[[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)]]
        :param F: 变异参数,越大则越能跳出局部最优点,但收敛速度会降低
        :param CR: 交叉概率
        '''
        self.adv = inp.copy()

        self.target_func = target_func
        self.dim_cnt = dim_cnt  # 维度
        self.individual_cnt = individual_cnt  # 个体个数
        self.bounds = bounds * change_byte_cnt # [[(i1_a1, i1_b1), (i1_a2, i1_b2)], [(i2_a1, i2_b1)], ...]
        self.best_fitness_value = float('Inf')
        self.F = F
        self.CR = CR
        self.best_unit = None  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.unit_list = [self.Unit(bounds * change_byte_cnt, dim_cnt * change_byte_cnt) for i in range(self.individual_cnt)]

        # 求种群的初始最优值
        for unit in self.unit_list:
            fitness_value = self.de_target_func(unit.vector)
            unit.fitness_value = fitness_value
            if self.best_fitness_value > fitness_value:
                self.best_fitness_value = fitness_value
                self.best_unit = unit

        self.fitness_val_list.append(self.best_fitness_value)

    # 变异
    def mutation_func(self):
        for i in range(self.individual_cnt): # 对每个个体执行操作
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.individual_cnt - 1)  # 随机数范围为[0, individual_cnt-1]的整数
                r2 = random.randint(0, self.individual_cnt - 1)
                r3 = random.randint(0, self.individual_cnt - 1)

            # 计算变异向量
            mutation = self.unit_list[r1].vector + self.F * (self.unit_list[r2].vector - self.unit_list[r3].vector)
            self.unit_list[i].set_mutation(mutation)

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

    def calc_all_after_crossover(self):
        advs = []
        for unit in self.unit_list:
            adv = self.adv.copy()[0]
            diff_vector = unit.vector_after_crossover
            i = 0
            while i < len(diff_vector):
                pos = int(diff_vector[i])
                val = int(diff_vector[i + 1])
                adv[pos] = val
                i += 2
            advs.append(adv)
        scores = self.target_func(np.array(advs))
        return scores

    # 目标优化函数
    def de_target_func(self, diff_vector):
        i = 0
        adv = self.adv.copy()
        while i < len(diff_vector):
            pos = diff_vector[i]
            val = diff_vector[i + 1]
            adv[0][pos] = val
            i += 2
        return self.target_func(adv)[0][0]

    def print_all_units_vector(self):
        for unit in self.unit_list:
            print(unit.vector)
    def update(self, fitness_value_threshold=0.5, iter_cnt=-1):
        cnt = 0
        iter_sum = 0
        while True:
            self.mutation_func()
            self.crossover_func()
            self.selection_func()
            self.fitness_val_list.append(self.best_fitness_value)
            iter_sum += 1
            print(iter_sum, ' ', self.best_fitness_value)
            # print(self.unit_list[0].vector_after_crossover)

            if iter_cnt > -1:
                cnt += 1
                if cnt >= iter_cnt:
                    break
            if self.best_fitness_value < fitness_value_threshold:
                break
        return self.adv

