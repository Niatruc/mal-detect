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
