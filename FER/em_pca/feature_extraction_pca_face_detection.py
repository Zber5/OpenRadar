import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from time import time
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from FER.utils import get_label


def cos_sim(a, b):
    sim = dot(a, b) / (norm(a) * norm(b))
    return sim


def window(seq_len, window_size=8, step_size=2):
    start = 0
    while (start + window_size) <= seq_len:
        end = start + window_size
        yield start, end
        start += step_size


def cor(x, y):
    r = np.corrcoef(x, y)
    return r[0, 1]


def numpy_to_panda(array):
    n_samples, h, w = array.shape
    num_rows = n_samples * w
    num_colums = h + 2
    data = np.zeros((num_rows, num_colums))

    id = 0
    for xi in array:
        d = np.zeros((w, num_colums))
        d[:, 0] = id
        d[:, 1] = np.arange(0, w).astype(int)
        d[:, 2:] = xi.T
        data[id * w:(id + 1) * w] = d
        id += 1

    df = pd.DataFrame(data, columns=['id', 'time'] + ['angle_{}'.format(i) for i in range(h)])

    df = df.astype({'id': int, 'time': int})

    return df


def get_fc_feature():
    fc_parameters = {
        "abs_energy": None,
        "absolute_sum_of_changes": None,
        "mean": None,
        "mean_abs_change": None,
        "mean_second_derivative_central": None,
        "number_peaks": [{'n': 10}],
        'approximate_entropy': [{'m': 2, 'r': 0.3}, {'m': 2, 'r': 0.5}, {'m': 2, 'r': 0.7}],
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "median": None,
        "ratio_beyond_r_sigma": [{'r': 1}],
        "sample_entropy": None,
        "standard_deviation": None,
        "skewness": None,
    }
    return fc_parameters


def max_likelihood_similarity(X_train, X_test):
    n_samples, _ = X_train.shape
    X_test = np.expand_dims(X_test, axis=0)
    X = np.vstack((X_train, X_test))
    cov_x = np.cov(X)
    return np.var(cov_x[n_samples, :n_samples]) #* np.max(cov_x[n_samples, :n_samples])


if __name__ == "__main__":
    # get the data
    root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"

    face_angle = 45
    face_range_start, face_range_end = 42 - 4, 42 + 4
    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    start_index = 0
    end_index = 9
    npy_format = "{}_{}.npy"

    train_data = []
    train_label = []
    # x_train
    a_index = 0
    for e in emotion_list:
        for i in range(start_index, end_index):
            bin_path = os.path.join(root_dir, npy_format.format(e, i))
            if not os.path.exists(bin_path):
                continue
            npy_data = np.load(bin_path)
            npy_label = get_label(e)
            train_data.append(npy_data[face_range_start:face_range_end])
            train_label.append(npy_label)
            a_index += 1
    train_data = np.stack(train_data, axis=0)
    Y_train = np.asarray(train_label)

    # x_test
    # emotion_list = ['Joy']
    # start_index = 0
    # end_index = 9
    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_stand"
    test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"
    test_data_path = os.path.join(root_dir, npy_format.format('Surprise', 9))
    # test_data_path = os.path.join(test_dir, "Stand_Joy_0.npy")
    test_data = np.load(test_data_path)
    seq_len, w = test_data.shape
    ws = 8
    overlap = 1

    all_test_data = np.zeros(((seq_len - 8) // 1 + 1, 8, 300))

    t_index = 0
    for s, e in window(seq_len, 8, 1):
        all_test_data[t_index] = test_data[s:e]
        t_index += 1

    # normalization
    scaler = StandardScaler()
    n_samples, h, w = train_data.shape
    test_n_samples, test_h, test_w = all_test_data.shape
    all_data = np.reshape(train_data, (n_samples, -1))
    all_test_data = np.reshape(all_test_data, (all_test_data.shape[0], -1))
    X_train = scaler.fit_transform(all_data)
    X_test = scaler.fit_transform(all_test_data)

    X_train = np.reshape(X_train, (n_samples, h, w))
    X_test = np.reshape(X_test, (test_n_samples, test_h, test_w))

    # X_train = np.mean(X_train, 1, keepdims=True)
    # X_test = np.mean(X_test, 1, keepdims=True)

    # principle component analysis

    X_train = numpy_to_panda(X_train)
    X_test = numpy_to_panda(X_test)

    X_train = extract_features(X_train, column_id="id", column_sort="time", impute_function=impute,
                               default_fc_parameters=get_fc_feature())

    X_test = extract_features(X_test, column_id="id", column_sort="time", impute_function=impute,
                              default_fc_parameters=get_fc_feature())

    # t0 = time()
    # n_components = 10
    # pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
    # print("done in %0.3fs" % (time() - t0))
    # eigenfaces = pca.components_.reshape((n_components, h, w))
    #
    #
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)
    #
    #
    # sim_data = np.zeros((X_test_pca.shape[0]))
    #
    # for i_test, test in enumerate(X_test_pca):
    #     test_sim = np.zeros((X_train_pca.shape[0]))
    #     for i_train, train in enumerate(X_train_pca):
    #         test_sim[i_train] = cos_sim(test[1:], train[1:])
    #         # test_sim[i_train] = cos_sim(test[0:5], train[0:5])
    #     sim_data[i_test] = np.mean(test_sim)
    #
    # plt.plot(sim_data)
    # plt.show()

    # X_train = np.sum(X_train, -1)
    # X_test = np.sum(X_test, -1)

    t0 = time()
    n_components = 5
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    # eigenfaces = pca.components_.reshape((n_components, h, w))

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    scaler_pca = StandardScaler()
    X_train_pca = scaler_pca.fit_transform(X_train_pca)
    X_test_pca = scaler_pca.fit_transform(X_test_pca)

    sim_data = np.zeros((X_test_pca.shape[0]))
    var = np.zeros((X_test_pca.shape[0]))

    for i_test, test in enumerate(X_test_pca):
        test_sim = np.zeros((X_train_pca.shape[0]))
        for i_train, train in enumerate(X_train_pca):
            # test_sim[i_train] = cos_sim(test[:], train[:])
            test_sim[i_train] = cor(test[0], train[0])
            # test_sim[i_train] = cos_sim(test[0:5], train[0:5])
        sim_data[i_test] = np.var(test_sim) * np.mean(test_sim) # test ok
        # sim_data[i_test] = np.max(test_sim) # test ok
        # sim_data[i_test] = np.mean(test_sim) + np.max(test_sim)
        # var[i_test] = np.var(test_sim)

    # for i_test, test in enumerate(X_test_pca):
    #     sim_data[i_test] = max_likelihood_similarity(X_train_pca[:, :], test[:])

    plt.scatter(X_train_pca[:, 1], X_train_pca[:, 2], c=train_label)
    for i, txt in enumerate(range(len(X_train_pca))):
        plt.annotate(train_label[i], (X_train_pca[i, 1], X_train_pca[i, 2]))
    # plt.show()

    # plt.show()

    plt.scatter(X_test_pca[20:50, 1], X_test_pca[20:50, 2])
    for txt in range(20, 50):
        plt.annotate(txt, (X_test_pca[txt, 1], X_test_pca[txt, 2]))

    plt.plot(sim_data)
    plt.axvline(42)
    # plt.plot(var)
    plt.show()

    print("")
