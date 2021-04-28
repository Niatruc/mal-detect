import ember
import lightgbm as lgb

model = lgb.Booster(model_file="/home/bohan/res/ml_dataset/ember2018/ember_model_2018.txt")
vd = "/home/bohan/res/ml_dataset/virusshare/"
d = open(vd + "VirusShare_06f1c1bc8ad03a43633807618a8e3158", "rb").read()
pred = ember.predict_sample(model, d)