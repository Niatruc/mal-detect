import os, math
from multiprocessing import Pool
import numpy as np
from sklearn import metrics
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences

# 解决问题: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

MAX_LEN = 2 ** 20

# 从Malconv的相关代码拷贝来的.
def preprocess(fn_list, max_len):
	'''
	:param fn_list: 完整文件路径列表
	:param max_len:	将fn_list中的二进制可执行文件按字节读进数组(按max_len补足长度)
	'''
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
	'''
	将一个二进制可执行文件按字节读进数组(按max_len补足长度)
	'''
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

# 只产生某类样本的批
def data_generator(exe_samples_path, ground_truth, batch_size=64, max_len=2**20, shuffle=True):
	for file_paths, file_bytes_batch in read_files_batch_in_dir(exe_samples_path, batch_size, max_len):
		# print(file_paths)
		yield (file_bytes_batch, ground_truth)

# 把(文件路径,标签)组成的列表分成训练集和测试集
def split_train_test_samples(mal_file_name_label_arr, benign_file_name_label_arr, train_test_ratio=(7, 3)):
	train_rate = train_test_ratio[0] / sum(train_test_ratio)
	train_mal_file_cnt = math.ceil(len(mal_file_name_label_arr) * train_rate)
	train_benign_file_cnt = math.ceil(len(benign_file_name_label_arr) * train_rate)

	train_file_name_label_arr = mal_file_name_label_arr[0:train_mal_file_cnt] + benign_file_name_label_arr[0:train_benign_file_cnt]
	test_file_name_label_arr = mal_file_name_label_arr[train_mal_file_cnt:] + benign_file_name_label_arr[train_benign_file_cnt:]

	return train_file_name_label_arr, test_file_name_label_arr

# 分别传入善意和恶意软件的存放路径,生成(文件路径,标签)组成的列表,善恶各一个
# limit_cnt大于0则表示样使用的本数量
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


def collect_exe_file_name_label_to_csv(mal_exe_samples_path_arr, benign_exe_samples_path_arr, limit_cnt=(-1, -1)):
	mal_file_name_label_arr, benign_file_name_label_arr = collect_exe_file_name_label(mal_exe_samples_path_arr, benign_exe_samples_path_arr, balanced=False)

# 使用包含(文件路径,标签)元组的列表(善恶分开的)来生成数据. 其调用了data_generator_3
# 使用前可先调用collect_exe_file_name_label生成元组
def data_generator_2(mal_file_name_label_arr, benign_file_name_label_arr, batch_size=64, max_len=2**20, shuffle=True, balanced=False):
	# idx = np.arange(len(data))
	# if shuffle:
	# 	np.random.shuffle(idx)

	# 是否需要两个类的样本数量一致
	if balanced:
		cnt = min(len(mal_file_name_label_arr), len(benign_file_name_label_arr))
		mal_file_name_label_arr = mal_file_name_label_arr[0:cnt]
		benign_file_name_label_arr = benign_file_name_label_arr[0:cnt]

	file_name_label_arr = mal_file_name_label_arr + benign_file_name_label_arr

	data_generator_3(file_name_label_arr, batch_size, max_len, shuffle)

# 使用包含(文件路径,标签)元组的列表来生成数据
def data_generator_3(file_name_label_arr, input_shape, data_type=float, batch_size=64, max_len=2**20, shuffle=True):
	file_name_label_arr = np.array(file_name_label_arr)

	idx = np.arange(len(file_name_label_arr))
	if shuffle:
		np.random.shuffle(idx)

	batches = [
		file_name_label_arr[idx[range(batch_size*i, min(len(file_name_label_arr), batch_size*(i+1)))]]
		for i in range(len(file_name_label_arr) // batch_size + 1)
	]

	# while True:
	for batch in batches:
		fn_list = [fn for fn, _ in batch]
		labels = [int(label) for _, label in batch]
		seqs = preprocess(fn_list, max_len)[0]
		seqs = seqs.reshape((len(seqs), *input_shape)).astype(data_type)
		yield seqs, np.array(labels)

# 给下面的get_all_data用
def preprocess2(file_name):
	return preprocess([file_name], MAX_LEN)[0]

# 一次性生成所有样本数据(可多进程)
# 注意在ipython或notebook中,第二次使用pool.map可能会卡住,会需要重启kernel. 
def get_all_data(file_name_label_arr, max_len=2**20, pool_size=4):
	# file_name_label_arr = np.array(mal_file_name_label_arr + benign_file_name_label_arr)
	file_name_label_arr = np.array(file_name_label_arr)
	file_name_arr = file_name_label_arr[:, 0]
	file_label_arr = file_name_label_arr[:, 1]
	print(file_name_arr)
	pool = Pool(pool_size)
	MAX_LEN = max_len
	seqs = pool.map(preprocess2, file_name_arr)
	pool.close()
	# pool.terminate()
	seqs = np.squeeze(np.array(seqs), 1) # squeeze把没用的维度去掉
	return seqs, file_label_arr.astype(int)

# 如果没有使用多进程的权限而不能用get_all_data, 又想一次性获取所有数据, 则用下列方法
def get_X_Y(file_name_label_arr, batch_size=128):
    X = np.array([])
    X = X.reshape(-1,2**20)
    Y = np.array([])
    for seqs, labels in file_util.data_generator_3(file_name_label_arr, batch_size=batch_size):
        X = np.concatenate((X, seqs))
        Y = np.concatenate((Y, labels))
        print(len(Y))
#         break
    return X, Y

