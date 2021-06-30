# formatter = get_ipython().display_formatter.formatters['text/plain']
# formatter.for_type(int, lambda n, p, cycle: p.text("0x%x" % n))

import os, pandas as pd
from keras.models import load_model
import numpy as np
import argparse

import utils
from utils import util, sklearn_util, exe_util
from ml_based_detection import mal_detect_base_model
from evasion_attack import gen_adversarial
import lightgbm as lgb
import ember
from multiprocessing import Pool

extractor = ember.PEFeatureExtractor()

# 这个函数定义必须放在模块的顶层,不能在函数内定义
def calc_feature(adv):
    features = np.array(extractor.feature_vector(adv[1].astype(np.byte).tostring()), dtype=np.float32)
    return adv[0], features


pool = Pool(10) # 这个初始化必须放calc_feature后面,不然会报找不到某attribute的错


TEST_LIGHTGBM = True
TEST_LIGHTGBM = False

TEST_DEEPMALNET = True
TEST_DEEPMALNET = False

TEST_KERAS_MODEL = True
TEST_KERAS_MODEL = False

TEST_SKLEARN_MODEL = True
TEST_SKLEARN_MODEL = False

max_len = 2**10
# max_len = 2**20
input_shape = (max_len, ) # (32, 32, 1)
input_shape = (32, 32, 1)

models_path = "/home/bohan/res/ml_models/zbh/model_files/"
# model_file_name = 'mlw_classification_cnn_img_1024.18-0.93.hdf5'
# model_file_name = 'mlw_classification_one_bilstm_1024.07-0.97.hdf5'
# model_file_name = 'DT2021052201_acc_0_8892.model'
# model_file_name = 'GBDT2021042701_acc_0_9133.model'
model_file_name = 'SVC2021052101_acc_0_9508.model'

scaler_path = models_path + "standard_scaler_4000_software.model"
# scaler_path = None

# './de_attack_result_256_bytes_mlw_cnn_img_2021051901.csv'
# attack_result_path = './de_attack_result_256_bytes_DT2021052201.csv'
# attack_result_path = './de_attack_result_1024_bytes_ember_lightgbm.csv'
attack_result_path = './de_attack_result_ember_lightgbm_stubborn_3.csv'

# save_units_path = "file_units_DT2021052201"
# save_units_path = "file_units_ember_lightgbm_20210607"
save_units_path = "file_units_ember_lightgbm_20210615"

utils.util.limit_gpu_memory(0)
# records = pd.read_csv('/home/bohan/res/ml_models/zbh/test_result/model_test/virusshare_1000.csv', index_col=False)
records = pd.read_csv('/home/bohan/res/ml_models/zbh/test_result/evasion_attack_test/de_attack_result_ember_lightgbm_stubborn.csv', index_col=False)

predict_func = None
pre_modify_file_func = None

if TEST_LIGHTGBM or TEST_DEEPMALNET:
    if TEST_LIGHTGBM:
        model = lgb.Booster(model_file="/home/bohan/res/ml_dataset/ember2018/ember_model_2018.txt")
        # records = pd.read_csv('../model_test_result/virusshare_1000_lightgbm.csv', index_col=False)
    if TEST_DEEPMALNET:
        model = load_model("/home/bohan/res/ml_models/zbh/deepmalnet.h5")
        records = pd.read_csv('../model_test_result/virusshare_1000_deepmalnet.csv', index_col=False)

    def predict_func(adv_ary):
        adv_feature_ary = []
        adv_ary_ = [(i, adv) for i, adv in enumerate(adv_ary)]

        # adv_feature_ary.append(features)
        # I.append(i)
        adv_feature_ary = pool.map(calc_feature, adv_ary_)
        # a = [pool.apply_async(calc_feature, args=(adv,)) for adv in adv_ary]
        # adv_feature_ary = [p.get() for p in a]
        # print(np.array(adv_feature_ary)[:, 0]) # 这句是我为了证明pool生成数据是按顺序来的
        adv_feature_ary = np.array([adv_[1] for adv_ in adv_feature_ary])
        res = model.predict(adv_feature_ary)
        res = np.expand_dims(res, 1)
        return res

    # 先向文件添加冗余节,导出函数等
    def pre_modify_file_func(fn):
        mrl, bytez = exe_util.find_pe_modifiable_range_ember_preproc(fn, export_func_cnt=10, inserted_sec_cnt=0x8)
        return mrl, bytez

elif TEST_SKLEARN_MODEL:
    # model = load_model(r + "/model_files/GBDT2021042701_acc_0_9133.model")
    model = sklearn_util.SklearnModel(models_path + model_file_name, max_len=max_len, scaler_path=scaler_path)
    records = pd.read_csv('/home/bohan/res/ml_models/zbh/test_result/model_test/virusshare_1000.csv', index_col=False)
elif TEST_KERAS_MODEL:
    model = mal_detect_base_model.KerasModel(path=models_path + model_file_name, max_len=max_len, input_shape=input_shape)
else:
    model = load_model("../../ember/malconv/malconv.h5")
    # records = pd.read_csv('../model_test_result/de_attack_result_256_bytes_from_first_stubborn.csv', index_col=0)
    # records = pd.read_csv('../model_test_result/virusshare_1000.csv', index_col=0)

init_units3 = np.load("/home/bohan/res/ml_models/zbh/mal_detect/tmp/stubborn_units_more_powerful.npy")

try:
    attack_result = pd.read_csv(attack_result_path, index_col=0)
except Exception:
    attack_result = pd.DataFrame(columns=('file_name', 'org_score', 'iter_sum', 'final_score'))

virusshare_dir = "/home/bohan/res/ml_dataset/virusshare/"
file_names = []
# org_scores = []
for index, row in records.iterrows():
    file_names.append(row.file_name)
    # org_scores.append(row.predict_score)
    # org_scores.append(row.org_score)

# file_names = ['VirusShare_3c8c59d25ecb9bd91e7b933113578e40', 'VirusShare_46bef7b95fb19e0ce5542332d9ebfe48',]
for i, file_name in enumerate(file_names[200:]):
    # print("原始预测分数: ", org_scores[i])
    adv_samples, test_info = gen_adversarial.gen_adv_samples(
        model, [virusshare_dir + file_name], predict_func,
        strategy=2,
        sub_strategy=0,
        workers=1,
        changed_bytes_cnt=256,
        max_iter=5000,
        thres=0.5,

        de_F=1.,
        individual_cnt=10,
        batch_size=32,
        change_range=0b1111,
        use_kick_mutation=True,
        kick_units_rate=1.,
        check_convergence_per_iter=100,
        check_dim_convergence_tolerate_cnt=3,

        save_units=True,
        save_units_path=save_units_path,
        save_as_init_unit_when_below_thres=True,
        save_units_with_lower_itersum=10, # 保存的unit对应的迭代数至少要多少(保存到**_withIterSum.npy文件中)
        init_units=None, # init_units3,
        init_units_upper_amount=15,
        used_init_units_cnt=4,
        use_increasing_units=True, # 是否把对每个样本产生作用的unit都加到初始units中供下一个样本使用

        pre_modify_file_func=pre_modify_file_func,
    )

    attack_result = attack_result.append({
        'file_name': file_name,
        'org_score': test_info['org_score'][0],
        'iter_sum': test_info['iter_sum'],
        'final_score': test_info['final_score']
    }, ignore_index=True)

    attack_result.to_csv(attack_result_path)
