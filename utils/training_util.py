import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import file_util
from sklearn import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping

def predict_benign_mal(model, file_path, max_len=2 ** 20):
    seq = file_util.read_file_to_bytes_arr(file_path, max_len)
    score = model.predict(np.array([seq]))[0]
    return score


# 一次只predict一个(即没有用batch)
def predict_test(model, exe_samples_path, max_len=2 ** 20, ground_truth=-1, result_path="test_result.csv",
                 print_test=False):
    test_result = pd.DataFrame(columns=('file_name', 'ground_truth', 'predict_score'))
    for file_path, file_bytes in file_util.read_files_in_dir(exe_samples_path):
        p = model.predict(file_bytes)[0][0]
        if print_test == True:
            print(p)
        test_result = test_result.append(
            pd.DataFrame({'file_name': [file_path], 'ground_truth': [ground_truth], 'predict_score': [p]}),
            ignore_index=True)

    test_result.to_csv(result_path, encoding="utf_8_sig")
    return test_result


# 一次predict多个
def predict_test_2(model, exe_samples_path, batch_size=32, max_len=2 ** 20, ground_truth=-1,
                   result_path="test_result.csv", print_test=False):
    test_result = pd.DataFrame(columns=('file_name', 'ground_truth', 'predict_score'))
    for file_paths, file_bytes_batch in file_util.read_files_batch_in_dir(exe_samples_path, batch_size, max_len):
        p = model.predict(file_bytes_batch, batch_size=batch_size)
        if print_test == True:
            print(p)
        new_df = pd.DataFrame([[file_paths[i], ground_truth, p[i][0]] for i in range(len(file_paths))],
                              columns=('file_name', 'ground_truth', 'predict_score'))
        test_result = test_result.append(new_df, ignore_index=True)

    test_result.to_csv(result_path, encoding="utf_8_sig")
    return test_result


# 使用predict_generator方法，可多线程
def predict_test_3(model, exe_samples_path, batch_size=64, max_len=2 ** 20, ground_truth=-1,
                   result_path="test_result.csv", workers=1, use_multiprocessing=False):
    for root, _, files0 in os.walk(exe_samples_path):
        files = files0

    p = model.predict_generator(
        generator=file_util.data_generator(exe_samples_path, ground_truth, batch_size, max_len, shuffle=False),
        steps=len(files) // batch_size + 1,
        verbose=1,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
    )

    test_result = pd.DataFrame([[os.path.join(root, files[i]), ground_truth, p[i][0]] for i in range(len(files))],
                               columns=('file_name', 'ground_truth', 'predict_score'))
    test_result.to_csv(result_path, encoding="utf_8_sig")
    return test_result


def predict_test_4(model, test_file_name_label_arr, batch_size=64, max_len=2 ** 20, result_path="test_result.csv",
                   workers=1, use_multiprocessing=False):
    '''同上，为多线程，不过这里用的是样本集的 (文件路径，标签) 对，所以可同时测试正负两类样本
    '''
    p = model.predict_generator(
        generator=file_util.data_generator_3(test_file_name_label_arr, batch_size, max_len, shuffle=False),
        steps=len(test_file_name_label_arr) // batch_size + 1,
        verbose=1,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
    )

    test_result = pd.DataFrame([[test_file_name_label_arr[i][0], test_file_name_label_arr[i][1], p[i][0]] for i in
                                range(len(test_file_name_label_arr))],
                               columns=('file_name', 'ground_truth', 'predict_score'))
    test_result.to_csv(result_path, encoding="utf_8_sig")
    return test_result


def train_model(model, epochs, train_file_name_label_arr, test_file_name_label_arr, save_path, batch_size=4,
                max_len=2 ** 20, train_test_ratio=(7, 3), limit_cnt=(-1, -1), balanced=True, save_best=True):
    # mal_file_name_label_arr, benign_file_name_label_arr = collect_exe_file_name_label(mal_exe_samples_path_arr, benign_exe_samples_path_arr, limit_cnt, balanced)

    # train_file_name_label_arr, test_file_name_label_arr = split_train_test_samples(mal_file_name_label_arr, benign_file_name_label_arr, train_test_ratio)

    # 回调函数
    ear = EarlyStopping(monitor='val_acc',
                        patience=5)  # patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
    mcp = ModelCheckpoint(
        save_path,
        monitor="val_acc",
        save_best_only=save_best,
        save_weights_only=False
    )

    history = model.fit_generator(
        file_util.data_generator_3(train_file_name_label_arr, batch_size, max_len),
        steps_per_epoch=len(train_file_name_label_arr) // batch_size + 1,
        epochs=epochs,
        verbose=1,
        callbacks=[ear, mcp],
        validation_data=file_util.data_generator_3(test_file_name_label_arr, batch_size, max_len),
        validation_steps=len(test_file_name_label_arr) // batch_size + 1
    )
    return history


def read_predict_test_result(result_path="test_result.csv"):
    df = pd.read_csv(result_path, usecols=('file_name', 'ground_truth', 'predict_score'), encoding="utf_8_sig")
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