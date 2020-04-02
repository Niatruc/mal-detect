import os, math
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 解决问题: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def preprocess(fn_list, max_len):
	corpus = []
	for fn in fn_list:
		if not os.path.isfile(fn):
			print(fn, 'not exist')
		else:
			with open(fn, 'rb') as f:
				corpus.append(f.read())
    
	corpus = [[byte for byte in doc] for doc in corpus]
	len_list = [len(doc) for doc in corpus]
	seq = pad_sequences(corpus, maxlen=max_len, padding='post', truncating='post')
	return seq, len_list

def read_file_to_bytes_arr(file_path, max_len):
	file_bytes = open(file_path, "rb").read()
	corpus = [byte for byte in file_bytes]
	seq = pad_sequences([corpus], maxlen=max_len, padding='post', truncating='post')[0] # pad_sequences这个函数需要一个两层嵌套的列表
	return seq

def read_files_in_dir(file_dir, max_len=2**20):
	for root, dirs, files in os.walk(file_dir):
		for f in files:
			file_path = os.path.join(root, f)
			file_bytes = open(file_path, "rb").read()
			corpus = [byte for byte in file_bytes]
			seq = pad_sequences([corpus], maxlen=max_len, padding='post', truncating='post')

			yield file_path, seq

def read_files_batch_in_dir(file_dir, batch_size=32, max_len=2**20):
	batch = []
	file_paths = []
	bs = batch_size

	for root, dirs, files in os.walk(file_dir):
		files_cnt = len(files)

		for f in files:
			file_path = os.path.join(root, f)
			file_paths.append(file_path)

			seq = read_file_to_bytes_arr(file_path, max_len)
			batch.append(seq)

			bs -= 1
			files_cnt -= 1

			if bs == 0 or files_cnt == 0:
				yield file_paths, np.array(batch)
				file_paths = []
				batch = []
				bs = batch_size

def data_generator(exe_samples_path, ground_truth, batch_size=64, max_len=2**20, shuffle=True):
	for file_paths, file_bytes_batch in read_files_batch_in_dir(exe_samples_path, batch_size, max_len):
		# print(file_paths)
		yield (file_bytes_batch, ground_truth)

def split_train_test_samples(mal_file_name_label_arr, benign_file_name_label_arr, train_test_ratio=(7, 3)):
	train_rate = train_test_ratio[0] / sum(train_test_ratio)
	train_mal_file_cnt = math.ceil(len(mal_file_name_label_arr) * train_rate)
	train_benign_file_cnt = math.ceil(len(benign_file_name_label_arr) * train_rate)

	train_file_name_label_arr = mal_file_name_label_arr[0:train_mal_file_cnt] + benign_file_name_label_arr[0:train_benign_file_cnt]
	test_file_name_label_arr = mal_file_name_label_arr[train_mal_file_cnt:] + benign_file_name_label_arr[train_benign_file_cnt:]

	return train_file_name_label_arr, test_file_name_label_arr

def collect_exe_file_name_label(mal_exe_samples_path_arr, benign_exe_samples_path_arr, limit_cnt=(-1, -1), balanced=True):
	def get_file_name_label_arr(path_arr, label, limit_cnt):
		file_name_arr = []
		for path in path_arr:
			for root, dirs, files in os.walk(path):
				for f in files:
					file_name = os.path.join(root, f)
					file_name_arr.append(file_name)

		if limit_cnt > 0:
			file_name_arr = file_name_arr[0:limit_cnt]

		return [(file_name ,label) for file_name in file_name_arr]

	mal_file_name_label_arr = get_file_name_label_arr(mal_exe_samples_path_arr, 1, limit_cnt[0])
	benign_file_name_label_arr = get_file_name_label_arr(benign_exe_samples_path_arr, 0, limit_cnt[1])

	# 需要两个类的样本数量一致
	if balanced:
		cnt = min(len(mal_file_name_label_arr), len(benign_file_name_label_arr))
		mal_file_name_label_arr = mal_file_name_label_arr[0:cnt]
		benign_file_name_label_arr = benign_file_name_label_arr[0:cnt]

	return mal_file_name_label_arr, benign_file_name_label_arr

def data_generator_2(mal_file_name_label_arr, benign_file_name_label_arr, batch_size=64, max_len=2**20, shuffle=True, balanced=False):
	# idx = np.arange(len(data))
	# if shuffle:
	# 	np.random.shuffle(idx)

	# 需要两个类的样本数量一致
	if balanced:
		cnt = min(len(mal_file_name_label_arr), len(benign_file_name_label_arr))
		mal_file_name_label_arr = mal_file_name_label_arr[0:cnt]
		benign_file_name_label_arr = benign_file_name_label_arr[0:cnt]

	file_name_label_arr = mal_file_name_label_arr + benign_file_name_label_arr

	data_generator_3(file_name_label_arr, batch_size, max_len)

def data_generator_3(file_name_label_arr, batch_size=64, max_len=2**20, shuffle=True):
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

	# for file_name, label in file_name_label_arr:
	# 	seq = read_file_to_bytes_arr(file_name, max_len)
	# 	yield(seq, label)

# 一次只predict一个
def predict_test(model, exe_samples_path, max_len=2**20, ground_truth=-1, result_path="test_result.csv", print_test=False):
	test_result = pd.DataFrame(columns=('file_name', 'ground_truth', 'predict_score'))
	for file_path, file_bytes in read_files_in_dir(exe_samples_path):
		p = model.predict(file_bytes)[0][0]
		if print_test == True:
			print(p)
		test_result = test_result.append(pd.DataFrame({'file_name': [file_path], 'ground_truth': [ground_truth], 'predict_score': [p]}), ignore_index=True)

	test_result.to_csv(result_path, encoding="utf_8_sig")
	return test_result

# 一次predict多个
def predict_test_2(model, exe_samples_path, batch_size=32, max_len=2**20, ground_truth=-1, result_path="test_result.csv", print_test=False):
	test_result = pd.DataFrame(columns=('file_name', 'ground_truth', 'predict_score'))
	for file_paths, file_bytes_batch in read_files_batch_in_dir(exe_samples_path, batch_size, max_len):
		p = model.predict(file_bytes_batch, batch_size=batch_size)
		if print_test == True:
			print(p)
		new_df = pd.DataFrame([[file_paths[i], ground_truth, p[i][0]] for i in range(len(file_paths))], columns=('file_name', 'ground_truth', 'predict_score'))
		test_result = test_result.append(new_df, ignore_index=True)

	test_result.to_csv(result_path, encoding="utf_8_sig")
	return test_result

# 使用predict_generator方法，可多线程
def predict_test_3(model, exe_samples_path, batch_size=64, max_len=2**20, ground_truth=-1, result_path="test_result.csv", workers=1, use_multiprocessing=False):
	for root, _, files0 in os.walk(exe_samples_path):
		files = files0

	p = model.predict_generator(
		generator = data_generator(exe_samples_path, ground_truth, batch_size, max_len, shuffle=False),
		steps = len(files) // batch_size + 1,
		verbose = 1,
		workers = workers,
		use_multiprocessing = use_multiprocessing,
	)

	test_result = pd.DataFrame([[os.path.join(root, files[i]), ground_truth, p[i][0]] for i in range(len(files))], columns=('file_name', 'ground_truth', 'predict_score'))
	test_result.to_csv(result_path, encoding="utf_8_sig")
	return test_result

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

def read_predict_test_result(result_path="test_result.csv"):
	df = pd.read_csv(result_path, usecols = ('file_name', 'ground_truth', 'predict_score'), encoding="utf_8_sig")
	return df

def draw_roc(test_result_csv_arr, threshold_step=0.1):
	FPRs = []
	TPRs = []
	test_result = pd.DataFrame(columns=('file_name', 'ground_truth', 'predict_score'))

	for csv in test_result_csv_arr:
		df = pd.read_csv(csv, header=0, index_col=0, encoding="utf_8_sig")
		test_result = test_result.append(df, ignore_index=True)

	P = test_result[test_result['ground_truth'] == 1]
	N = test_result[test_result['ground_truth'] == 0]
	P_cnt = P.file_name.count()
	N_cnt = N.file_name.count()

	for threshold in np.append(np.arange(0, 1, threshold_step), 1):
		TP_cnt = P[P['predict_score'] >= threshold].file_name.count()
		FP_cnt = N[N['predict_score'] >= threshold].file_name.count()
		TPRs.append(TP_cnt / P_cnt)
		FPRs.append(FP_cnt / N_cnt)
	
	auc = metrics.auc(FPRs, TPRs)
	plt.plot(FPRs, TPRs, label='AUC: ' % auc)
	plt.title('ROC')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.show()
	
	return auc, TPRs, FPRs