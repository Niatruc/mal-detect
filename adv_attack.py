import os, pandas as pd
from keras.models import load_model
import numpy as np
import argparse

import utils
from utils import util, sklearn_util
from ml_based_detection import mal_detect_base_model
from evasion_attack import gen_adversarial
import lightgbm as lgb
import ember
from multiprocessing import Pool

# os.sys.path.append("/home/bohan/res/ml_models/zbh/")
parser = argparse.ArgumentParser(description='Malconv-keras adversarial attack')
parser.add_argument('--limit', type=float, default=0., help="limit gpu memory percentage")
parser.add_argument('--strategy', type=int, default=2, help="Strategy (0/1: fgsm; 2: de")
parser.add_argument('--gpu_num', type=str, default='0', help="Choose a gpu")
parser.add_argument('--workers_cnt', type=int, default=0, help="Threads count")
parser.add_argument('--model_path', type=str, default='../../ember/malconv/malconv.h5',    help="MalConv's path")
parser.add_argument('--save_path', type=str, default='../model_test_result/attack_result.csv',    help="Path for saving attack result")
parser.add_argument('--malware_test_res_csv_path', type=str, default='../model_test_result/virusshare_1000.csv',    help="malware_test_res_csv_path")
parser.add_argument('--from_row', type=int, default=0,    help="From which row in <malware_test_res_csv_path>.csv")
parser.add_argument('--to_row', type=int, default=1000,    help="To which row in <malware_test_res_csv_path>.csv")
parser.add_argument('--virusshare_dir', type=str, default='../../../ml_dataset/virusshare/',    help="virusshare_dir")
parser.add_argument('--changed_bytes_cnt', type=int, default=8,    help="changed_bytes_cnt")
parser.add_argument('--max_iter', type=int, default=1,    help="max_iter")
parser.add_argument('--batch_size', type=int, default=10,    help="batch_size")
parser.add_argument('--de_individual_cnt', type=int, default=32,    help="de_individual_cnt")
parser.add_argument('--de_change_range', type=int, default=0b1111,    help="de_change_range")
parser.add_argument('--use_kick_mutation', type=bool, default=True,    help="use_kick_mutation")
parser.add_argument('--search_exact_len', type=bool, default=True,    help="search_exact_len")

TEST = False
TEST = True

if __name__ == '__main__' and not TEST:
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    utils.util.limit_gpu_memory(args.limit)
    model = load_model(args.model_path)

    malware_test_res = pd.read_csv(args.malware_test_res_csv_path)
    try:
        attack_result = pd.read_csv(args.save_path, index_col=0)
    except Exception:
        attack_result = pd.DataFrame(columns=('file_name', 'org_score', 'iter_sum', 'final_score'))

    malware_test_res = malware_test_res[args.from_row : args.to_row]
    for index, row in malware_test_res.iterrows():
        virus_path = args.virusshare_dir + row.file_name
        print("开始操作: %d: %s" % (index, virus_path))

        _, test_info = gen_adversarial.gen_adv_samples(
            model, [virus_path],
            workers=args.workers_cnt,
            strategy=args.strategy,
            changed_bytes_cnt=args.changed_bytes_cnt,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            individual_cnt=args.de_individual_cnt,
            change_range=args.de_change_range,
            use_kick_mutation=args.use_kick_mutation,
            exact_len=args.search_exact_len,
        )
        attack_result = attack_result.append({
            'file_name': row.file_name,
            'org_score': row.predict_score,
            'iter_sum': test_info['iter_sum'],
            'final_score': test_info['final_score']
        }, ignore_index=True)

        attack_result.to_csv(args.save_path)

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

TEST_SKLEARN_MODEL = False
TEST_SKLEARN_MODEL = True

max_len = 2**10
input_shape = (max_len, ) # (32, 32, 1)

models_path = "/home/bohan/res/ml_models/zbh/model_files/"
# 'mlw_classification_cnn_img_1024.18-0.93.hdf5'
# 'GBDT2021042701_acc_0_9133.model'
model_file_name = 'DT2021052201_acc_0_8892.model'
# model_file_name = 'mlw_classification_one_bilstm.07-0.97.hdf5'

# './de_attack_result_256_bytes_mlw_cnn_img_2021051901.csv'
stubborn_attack_result_path = './de_attack_result_256_bytes_DT2021052201.csv'
# stubborn_attack_result_path = './de_attack_result_256_bytes_mlw_classification_one_bilstm.csv'

# save_units_path = "file_units_mlw_classification_one_bilstm"
save_units_path = "file_units_DT2021052201"

if TEST:
    utils.util.limit_gpu_memory(0)
    records = pd.read_csv('/home/bohan/res/ml_models/zbh/test_result/model_test/virusshare_1000.csv', index_col=False)

    predict_func = None
    if TEST_LIGHTGBM or TEST_DEEPMALNET:
        if TEST_LIGHTGBM:
            model = lgb.Booster(model_file="/home/bohan/res/ml_dataset/ember2018/ember_model_2018.txt")
            records = pd.read_csv('../model_test_result/virusshare_1000_lightgbm.csv', index_col=False)
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
            print(np.array(adv_feature_ary)[:, 0]) # 这句是我为了证明pool生成数据是按顺序来的
            adv_feature_ary = np.array([adv_[1] for adv_ in adv_feature_ary])
            res = model.predict(adv_feature_ary)
            return res

    elif TEST_SKLEARN_MODEL:
        # model = load_model(r + "/model_files/GBDT2021042701_acc_0_9133.model")
        model = sklearn_util.SklearnModel(models_path + model_file_name, max_len=max_len, scaler_path=models_path + "standard_scaler_4000_software.model")
        records = pd.read_csv('/home/bohan/res/ml_models/zbh/test_result/model_test/virusshare_1000.csv', index_col=False)
    elif TEST_KERAS_MODEL:
        model = mal_detect_base_model.KerasModel(path=models_path + model_file_name, max_len=max_len, input_shape=input_shape)
    else:
        model = load_model("../../ember/malconv/malconv.h5")
        # records = pd.read_csv('../model_test_result/de_attack_result_256_bytes_from_first_stubborn.csv', index_col=0)
        # records = pd.read_csv('../model_test_result/virusshare_1000.csv', index_col=0)

    init_units3 = np.load("/home/bohan/res/ml_models/zbh/mal_detect/tmp/stubborn_units_more_powerful.npy")

    try:
        stubborn_attack_result = pd.read_csv(stubborn_attack_result_path, index_col=0)
    except Exception:
        stubborn_attack_result = pd.DataFrame(columns=('file_name', 'org_score', 'iter_sum', 'final_score'))

    virusshare_dir = "/home/bohan/res/ml_dataset/virusshare/"
    file_names = []
    org_scores = []
    for index, row in records.iterrows():
        file_names.append(row.file_name)
        org_scores.append(row.predict_score)

    # file_names = ['VirusShare_3c8c59d25ecb9bd91e7b933113578e40', 'VirusShare_46bef7b95fb19e0ce5542332d9ebfe48',]
    for i, file_name in enumerate(file_names[242:]):
        # print("原始预测分数: ", org_scores[i])
        adv_samples, test_info = gen_adversarial.gen_adv_samples(
            model, [virusshare_dir + file_name], predict_func,
            strategy=2,
            sub_strategy=0,
            workers=1,
            changed_bytes_cnt=256,
            max_iter=2000,
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
            init_units=None,
            init_units_upper_amount=15,
            used_init_units_cnt=4,
            use_increasing_units=True, # 是否把对每个样本产生作用的unit都加到初始units中供下一个样本使用
        )

        stubborn_attack_result = stubborn_attack_result.append({
            'file_name': file_name,
            'org_score': org_scores[i],
            'iter_sum': test_info['iter_sum'],
            'final_score': test_info['final_score']
        }, ignore_index=True)

        stubborn_attack_result.to_csv(stubborn_attack_result_path)

# python adv_attack.py
# --from_row 70
# --to_row 71
# --use_kick_mutation False
# --changed_bytes_cnt 128
# --de_individual_cnt 32
# --de_change_range 7
# --gpu_num 1
# --model_path /dataset/2/zhangbohan/malconv.h5
# --max_iter 3000
# --save_path /dataset/1/zhangbohan/virusshare_1000_attack_result_64_bytes.csv
# --virusshare_dir /dataset/1/virusshare/

# python adv_attack.py
# --from_row 0
# --to_row 1000
# --use_kick_mutation False
# --changed_bytes_cnt 128
# --de_individual_cnt 32
# --de_change_range 7
# --gpu_num 1
# --model_path /dataset/2/zhangbohan/malconv.h5
# --max_iter 3000
# --save_path /dataset/1/zhangbohan/virusshare_1000_attack_result_64_bytes.csv
# --virusshare_dir /dataset/1/virusshare/
