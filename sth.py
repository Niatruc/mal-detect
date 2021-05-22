import ember
import lightgbm as lgb

model = lgb.Booster(model_file="/home/bohan/res/ml_dataset/ember2018/ember_model_2018.txt")
vd = "/home/bohan/res/ml_dataset/virusshare/"
d = open(vd + "VirusShare_06f1c1bc8ad03a43633807618a8e3158", "rb").read()
pred = ember.predict_sample(model, d)

successful_file_paths = [
    'VirusShare_3c8c59d25ecb9bd91e7b933113578e40',
    'VirusShare_3a4fac1796f0816d7567abb9bf0a9440',
    'VirusShare_01cd58ba6e5f9d1e1f718dfba7478d30',
    'VirusShare_40fd3647c44239df91fc5d7765dd0d9f',
    'VirusShare_22fd8d088ef3ccadc6baa44dc8cb7490',
]

stubborn_files = [
    'VirusShare_1e4997bc0fced91b25632c3151f91710',
    'VirusShare_01dd838da5efd739579f412e4f56b180',
    'VirusShare_21d3b6c1cd1873add493e0675fbd8220',
    'VirusShare_13351c7d2aa385a6b0e2b08f676f8250',
    'VirusShare_46bef7b95fb19e0ce5542332d9ebfe48',
    'VirusShare_327ab01f70084d5fc63bc5669e235740',
    'VirusShare_06f1c1bc8ad03a43633807618a8e3158',
]
# 4401, 4818, 46036, 4387, 50000(x), 4047