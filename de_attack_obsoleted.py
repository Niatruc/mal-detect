from mal_detect.differential_evolution import differential_evolution
import numpy as np

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