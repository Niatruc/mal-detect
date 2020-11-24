import os, imp, keras
from keras.models import load_model
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# os.sys.path.append("/home/bohan/res/ml_models/zbh/")

import mal_detect
from mal_detect import file_util, gen_adversarial

malconv = load_model("/home/bohan/res/ember/malconv/malconv.h5")

# adv_samples, log = gen_adversarial.gen_adv_samples(malconv, ['/home/bohan/res/ml_dataset/Malware_Detection_PE-Based_Analysis_Using_Deep_Learning_Algorithm_Dataset_old/Dataset/Virus/Virus train/Locker/VirusShare_13c63e0329202076f45796dba3ed6b8f.exe'])
adv_samples, log = gen_adversarial.gen_adv_samples(
    malconv, [
        # '/home/bohan/res/ml_dataset/virusshare/VirusShare_24580df24fb34966023b5dd6b37b1a3c',
        '/home/bohan/res/ml_dataset/virusshare/VirusShare_3c8c59d25ecb9bd91e7b933113578e40',
        # '/home/bohan/res/ml_dataset/Malware_Detection_PE-Based_Analysis_Using_Deep_Learning_Algorithm_Dataset_old/Dataset/Virus/Virus train/Locker/VirusShare_13c63e0329202076f45796dba3ed6b8f.exe'
    ], strategy=2, changed_bytes_cnt=8, max_iter=50, individual_cnt=20, change_range=0b1111, use_kick_mutation=True)
# adv_samples, log = gen_adversarial.gen_adv_samples(malconv, ['/home/bohan/res/ml_dataset/virusshare/VirusShare_24580df24fb34966023b5dd6b37b1a3c'], 90)

b = open('/home/bohan/res/ml_dataset/virusshare/VirusShare_24580df24fb34966023b5dd6b37b1a3c', 'rb').read()

