import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from time import time
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
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


def norm(x):
    mean = np.mean(x)
    std = np.std(x)
    norm_x = (x - mean) / std
    return norm_x, mean, std


def stage1(x):
    y = []
    for td in x:
        m, s = np.mean(td), np.std(td)
        td[np.logical_and(td > (m - s), td < (m + s))] = 0
        y.append(np.true_divide(td.sum(1), (td != 0).sum(1)))
    y = np.stack(y, axis=0)
    y[np.isnan(y)] = 0
    return y


def stage1_v1(x):
    y = []
    for td in x:
        c = np.zeros((td.shape[0]))
        for ai, a in enumerate(td):
            m, s = np.mean(a), np.std(a)
            a[np.logical_and(a > (m - s), a < (m + s))] = 0
            c[ai] = np.true_divide(a.sum(0), (a != 0).sum(0))
        y.append(c)
    y = np.stack(y, axis=0)
    y[np.isnan(y)] = 0
    return y


if __name__ == "__main__":
    face_angle = 42
    face_range_start, face_range_end = 42 - 4, 42 + 4

    # get the data
    root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_v1"

    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    start_index = 0
    end_index = 9
    npy_format = "{}_{}.npy"

    # Train data
    train_data = []
    train_label = []

    for e in emotion_list:
        for i in range(start_index, end_index):
            bin_path = os.path.join(root_dir, npy_format.format(e, i))
            if not os.path.exists(bin_path):
                continue
            npy_data = np.load(bin_path)
            npy_label = get_label(e)
            train_data.append(npy_data[face_range_start:face_range_end])
            train_label.append(npy_label)
    train_data = np.stack(train_data, axis=0)
    Y_train = np.asarray(train_label)

    # Test data
    test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_stand"
    # test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_v1"
    test_data_path = os.path.join(test_dir, npy_format.format('Joy', 2))
    test_npy = np.load(test_data_path)
    seq_len, w = test_npy.shape
    ws = 8
    overlap = 1

    test_data = np.zeros(((seq_len - ws) // overlap + 1, ws, 300))

    t_index = 0
    for s, e in window(seq_len, ws, overlap):
        test_data[t_index] = test_npy[s:e]
        t_index += 1

    # all_data[a_index] = npy_data[face_range_start:face_range_end]

    # x_test
    # emotion_list = ['Joy']
    # start_index = 0
    # end_index = 9
    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"
    # test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"
    # test_data_path = os.path.join(test_dir, npy_format.format('Joy', 9))
    # test_data = np.load(test_data_path)
    # seq_len, w = test_data.shape
    # ws = 8
    # overlap = 1
    #
    # all_test_data = np.zeros(((seq_len - ws) // overlap + 1, ws, 300))
    #
    # t_index = 0
    # for s, e in window(seq_len, ws, overlap):
    #     all_test_data[t_index] = test_data[s:e]
    #     t_index += 1

    # principle component analysis

    # 1. normalization
    train_data, mean, std = norm(train_data)
    test_data = (test_data - mean) / std

    # preprocessing
    X_train = stage1_v1(train_data)
    X_test = stage1_v1(test_data)

    # scaler = StandardScaler()
    # n_samples, h, w = all_data.shape
    # test_n_samples, test_h, test_w = all_test_data.shape
    # all_data = np.reshape(all_data, (n_samples, -1))
    # all_test_data = np.reshape(all_test_data, (all_test_data.shape[0], -1))
    # X_train = scaler.fit_transform(all_data)
    # X_test = scaler.fit_transform(all_test_data)
    #
    # # reshape the data
    #
    # X_train = np.reshape(X_train, (n_samples, h, w))
    # X_test = np.reshape(X_test, (test_n_samples, test_h, test_w))

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

    sim_data = np.zeros((X_test_pca.shape[0]))

    for i_test, test in enumerate(X_test_pca):
        test_sim = np.zeros((X_train_pca.shape[0]))
        for i_train, train in enumerate(X_train_pca):
            # test_sim[i_train] = cos_sim(test[:], train[:])
            # test_sim[i_train] = cor(test[:], train[:])
            test_sim[i_train] = np.linalg.norm(test[1:]-train[1:])
            # test_sim[i_train] = cos_sim(test[0:5], train[0:5])
        # sim_data[i_test] = 1/np.max(test_sim)
        # sim_data[i_test] = np.var(test_sim) * np.mean(test_sim)
        # sim_data[i_test] = np.mean(test_sim)
        sim_data[i_test] = 1/ np.min(test_sim)

    # plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_label)
    # function(x, mu, sigma, rate, p) p*dnorm(x, mu, sigma) + (1-p)*dpois(x,rate) two distribution fit

        plt.scatter(X_train_pca[:, 1], X_train_pca[:, 2], c=train_label)
        for i, txt in enumerate(range(len(X_train_pca))):
            plt.annotate(train_label[i], (X_train_pca[i, 1], X_train_pca[i, 2]))
        # plt.show()

        # plt.show()

        plt.scatter(X_test_pca[20:35, 1], X_test_pca[20:35, 2])
        for txt in range(20, 35):
            plt.annotate(txt, (X_test_pca[txt, 1], X_test_pca[txt, 2]))


    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_label)
    for i, txt in enumerate(range(len(X_train_pca))):
        plt.annotate(train_label[i], (X_train_pca[i, 0], X_train_pca[i, 1]))
    # plt.show()

    # plt.show()

    plt.scatter(X_test_pca[0:50, 0], X_test_pca[0:50, 1])
    for txt in range(0, 50):
        plt.annotate(txt, (X_test_pca[txt, 0], X_test_pca[txt, 1]))



    for i, txt in enumerate(range(len(X_test_pca))):
        plt.annotate(txt, (X_test_pca[i, 1], X_test_pca[i, 2]))
    plt.show()

    plt.plot(sim_data)
    # plt.plot(X_test_pca[:, 0]) 

    plt.axvline(42)
    plt.show()

    print("")
