import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from time import time
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from FER.utils import get_label
from sklearn.mixture import GaussianMixture


def cos_sim(a, b):
    sim = dot(a, b) / (norm(a) * norm(b))
    return sim


def dot(A, B):
    return (sum(a * b for a, b in zip(A, B)))


def cosine_similarity(a, b):
    return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))


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


def get_prob(res , p_label):
    pp =[]
    for i in range(p_label):
        p_data = len(res[res==i])/len(res)
        pp.append(p_data)
    return pp

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

    # preprocessing
    # train_data = stage1_v1(train_data)
    # test_data = stage1_v1(test_data)
    train_data, mean, std = norm(train_data)
    test_data = (test_data - mean) / std

    # GMM
    n_comp = 3
    gm = GaussianMixture(n_components=n_comp, random_state=0).fit(train_data[:, 3:6].reshape(-1, 1))

    print(gm.means_)
    print()
    print(gm.covariances_)
    print()
    print(gm.weights_)


    res_data = []
    tpp = []

    for i in range(len(test_data)):
        # res = gm.predict(test_data[i, 3:6].reshape(-1,1))
        res = gm.predict(test_data[i, :].reshape(-1, 1))
        res_pp = get_prob(res, n_comp)
        res = cosine_similarity(a=gm.weights_, b=res_pp)
        res_data.append(res)
        tpp.append(res_pp)

    plt.close()
    x = np.linspace(-90, 90, len(res_data))
    plt.plot(x, res_data)
    plt.ylabel("Probability")
    plt.xlabel("Angle")
    plt.show()

    print()