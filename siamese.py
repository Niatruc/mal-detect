'''
改自https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

def create_base_network_keras(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
	input = Input(shape=input_shape)
	x = Flatten()(input)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	return Model(input, x)

def contrastive_loss(y_true, y_pred, margin=1):
	'''Contrastive loss from Hadsell-et-al.'06
	对比损失函数
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

'''
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)
'''
def euclidean_distance(x, y):
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
	'''输入的shape对输出的shape的映射函数
	'''
	shape1, shape2 = shapes
	return (shape1[0], 1)

def accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

'''
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
'''

def pair_generator(file_name_label_arr, batch_size=64, max_len=2**20, shuffle=True):
	file_name_label_arr = np.array(file_name_label_arr)

	idx = np.arange(len(file_name_label_arr))
	if shuffle:
		np.random.shuffle(idx)

	batches = [
		file_name_label_arr[idx[range(batch_size*i, min(len(file_name_label_arr), batch_size*(i+1)))]]
		for i in range(len(file_name_label_arr) // batch_size + 1)
	]

	while True:
		for batch in batches:
			fn_list = [fn for fn, _ in batch]
			labels = [label for _, label in batch]
			seqs = preprocess(fn_list, max_len)[0]
			yield seqs, np.array(labels)

def train_model(model, epochs, train_file_name_label_arr, test_file_name_label_arr, save_path, batch_size=4, max_len=2**20, train_test_ratio=(7, 3), limit_cnt=(-1, -1), balanced=True, save_best=True):
	# mal_file_name_label_arr, benign_file_name_label_arr = collect_exe_file_name_label(mal_exe_samples_path_arr, benign_exe_samples_path_arr, limit_cnt, balanced)

	# train_file_name_label_arr, test_file_name_label_arr = split_train_test_samples(mal_file_name_label_arr, benign_file_name_label_arr, train_test_ratio)

	# 回调函数
	ear = EarlyStopping(monitor='val_acc', patience=5) # patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
	mcp = ModelCheckpoint(
		save_path,
		monitor="val_acc",
		save_best_only=save_best,
		save_weights_only=False
	)

	history = model.fit_generator(
		data_generator_3(train_file_name_label_arr, batch_size, max_len),
		steps_per_epoch=len(train_file_name_label_arr)//batch_size + 1,
		epochs=epochs, 
		verbose=1, 
		callbacks=[ear, mcp],
		validation_data=data_generator_3(test_file_name_label_arr, batch_size, max_len),
		validation_steps=len(test_file_name_label_arr)//batch_size + 1
	)
	return history