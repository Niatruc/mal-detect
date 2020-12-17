import os, sys, imp, keras, pandas as pd
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse

import file_util, gen_adversarial, utils

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

    utils.limit_gpu_memory(args.limit)
    malconv = load_model(args.model_path)

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
            malconv, [virus_path],
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

if TEST:
    utils.limit_gpu_memory(0)
    malconv = load_model("../../ember/malconv/malconv.h5")

    file_paths = [
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_24580df24fb34966023b5dd6b37b1a3c',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_3c8c59d25ecb9bd91e7b933113578e40',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_5a76aa2603e49a7dd6bac0ce743d25c0',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_1e4997bc0fced91b25632c3151f91710',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_3a4fac1796f0816d7567abb9bf0a9440',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_01cd58ba6e5f9d1e1f718dfba7478d30',
        # '/home/bohan/res/ml_dataset/Malware_Detection_PE-Based_Analysis_Using_Deep_Learning_Algorithm_Dataset_old/Dataset/Virus/Virus train/Locker/VirusShare_13c63e0329202076f45796dba3ed6b8f.exe'
    ]
    successful_file_paths = [
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_3c8c59d25ecb9bd91e7b933113578e40',
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_3a4fac1796f0816d7567abb9bf0a9440',
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_01cd58ba6e5f9d1e1f718dfba7478d30',
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_40fd3647c44239df91fc5d7765dd0d9f',
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_22fd8d088ef3ccadc6baa44dc8cb7490',
    ]

    stubborn_file_paths = [
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_1e4997bc0fced91b25632c3151f91710',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_01dd838da5efd739579f412e4f56b180',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_21d3b6c1cd1873add493e0675fbd8220',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_46bef7b95fb19e0ce5542332d9ebfe48',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_13351c7d2aa385a6b0e2b08f676f8250',
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_327ab01f70084d5fc63bc5669e235740',
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_06f1c1bc8ad03a43633807618a8e3158',

    ]
    # 4401, 4818, 46036, 4387, 50000(x), 4047
    init_units1 = np.load("stubborn_file_units.npy")
    init_units2 = np.load("units_more_powerful.npy")
    init_units3 = np.load("stubborn_units_more_powerful.npy")
    init_units = np.concatenate((init_units1, init_units2))

    stubborn_records = pd.read_csv('../model_test_result/de_attack_result_256_bytes_from_first_stubborn.csv', index_col=0)
    try:
        stubborn_attack_result = pd.read_csv('./fgsm_attack_result_256_bytes_from_first_stubborn.csv', index_col=0)
    except Exception:
        stubborn_attack_result = pd.DataFrame(columns=('file_name', 'org_score', 'iter_sum', 'final_score'))

    virusshare_dir = "/home/bohan/res/ml_dataset/virusshare/"
    file_names = []
    for index, row in stubborn_records.iterrows():
        file_names.append(row.file_name)

    file_names = ['VirusShare_3c8c59d25ecb9bd91e7b933113578e40', 'VirusShare_46bef7b95fb19e0ce5542332d9ebfe48',]
    for file_name in file_names:
        adv_samples, test_info = gen_adversarial.gen_adv_samples(
            malconv, [virusshare_dir + file_name],
            strategy=1,
            sub_strategy=0,
            workers=1,
            changed_bytes_cnt=256,
            max_iter=50000,
            thres=0.5,

            de_F=1.,
            individual_cnt=16,
            batch_size=32,
            change_range=0b0111,
            use_kick_mutation=True,
            kick_units_rate=1.,
            check_convergence_per_iter=100,

            save_units=False,
            save_units_path="stubborn_file_units_4",
            save_when_below_thres=True,
            init_units=init_units3,
            used_init_units_cnt=7,
            use_increasing_units=True,
        )

        # stubborn_attack_result = stubborn_attack_result.append({
        #     'file_name': file_name,
        #     'org_score': 1.0,
        #     'iter_sum': test_info['iter_sum'],
        #     'final_score': test_info['final_score']
        # }, ignore_index=True)
        #
        # stubborn_attack_result.to_csv("de_attack_result_256_bytes_from_first_stubborn_2.csv")

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