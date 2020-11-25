import os, sys, imp, keras, pandas as pd
from keras.models import load_model
import tensorflow as tf
import argparse

import file_util, gen_adversarial, utils

# os.sys.path.append("/home/bohan/res/ml_models/zbh/")
parser = argparse.ArgumentParser(description='Malconv-keras adversarial attack')
parser.add_argument('--limit', type=float, default=0., help="limit gpu memory percentage")
parser.add_argument('--strategy', type=int, default=2, help="Strategy (0/1: fgsm; 2: de")
parser.add_argument('--gpu_num', type=int, default=0, help="Choose a gpu")
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

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    utils.limit_gpu_memory(args.limit)
    malconv = load_model(args.model_path)

    malware_test_res = pd.read_csv(args.malware_test_res_csv_path)
    try:
        attack_result = pd.read_csv(args.save_path, index_col=0)
    except Exception:
        attack_result = pd.DataFrame(columns=('file_name', 'org_score', 'iter_sum', 'final_score'))

    malware_test_res = malware_test_res[args.from_row : args.to_row + 1]
    for index, row in malware_test_res.iterrows():
        virus_path = args.virusshare_dir + row.file_name
        print("开始操作: %s" % virus_path)

        _, test_info = gen_adversarial.gen_adv_samples(malconv, [virus_path],
            strategy=args.strategy,
            changed_bytes_cnt=args.changed_bytes_cnt,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            individual_cnt=args.de_individual_cnt,
            change_range=args.de_change_range,
            use_kick_mutation=args.use_kick_mutation
        )
        attack_result = attack_result.append({
            'file_name': row.file_name,
            'org_score': row.predict_score,
            'iter_sum': test_info['iter_sum'],
            'final_score': test_info['final_score']
        }, ignore_index=True)

        attack_result.to_csv(args.save_path)


# adv_samples, log = gen_adversarial.gen_adv_samples(malconv, ['/home/bohan/res/ml_dataset/Malware_Detection_PE-Based_Analysis_Using_Deep_Learning_Algorithm_Dataset_old/Dataset/Virus/Virus train/Locker/VirusShare_13c63e0329202076f45796dba3ed6b8f.exe'])
# adv_samples, log = gen_adversarial.gen_adv_samples(
#     malconv, [
#         # '/home/bohan/res/ml_dataset/virusshare/VirusShare_24580df24fb34966023b5dd6b37b1a3c',
#         '/home/bohan/res/ml_dataset/virusshare/VirusShare_3c8c59d25ecb9bd91e7b933113578e40',
#         # '/home/bohan/res/ml_dataset/Malware_Detection_PE-Based_Analysis_Using_Deep_Learning_Algorithm_Dataset_old/Dataset/Virus/Virus train/Locker/VirusShare_13c63e0329202076f45796dba3ed6b8f.exe'
#     ], strategy=2, changed_bytes_cnt=8, max_iter=50, individual_cnt=32, change_range=0b1111, use_kick_mutation=True)
# adv_samples, log = gen_adversarial.gen_adv_samples(malconv, ['/home/bohan/res/ml_dataset/virusshare/VirusShare_24580df24fb34966023b5dd6b37b1a3c'], 90)

