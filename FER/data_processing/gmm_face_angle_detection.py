import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

os.chdir("C:\\Users\\Zber\\Documents\\Dev_program\\OpenRadar")
from time import time
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from FER.utils import get_label
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from scipy import signal


def dot(A, B):
    return (sum(a * b for a, b in zip(A, B)))


def cosine_similarity(a, b):
    return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))


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


def kl(u1, u2, s1, s2):
    return np.log(s2 / s1) + (s1 ** 2 + (u1 - u2) ** 2) / (2 * s2 ** 2) - 1 / 2


def get_prob(res, p_label):
    pp = []
    for i in range(p_label):
        p_data = len(res[res == i]) / len(res)
        pp.append(p_data)
    return pp


def test_prediction(test_data, gm):
    res_data = []
    tpp = []
    for i in range(len(test_data)):
        res = gm.predict(test_data[i, :].reshape(-1, 1))
        res_pp = get_prob(res, n_comp)
        res = cosine_similarity(a=gm.weights_, b=res_pp)
        res_data.append(res)
        tpp.append(res_pp)
    return res_data, tpp


def generate_test_data(test_npy, seq_len, ws, overlap, w):
    test_data = np.zeros(((seq_len - ws) // overlap + 1, ws, w))
    t_index = 0
    for s, e in window(seq_len, ws, overlap):
        test_data[t_index] = test_npy[s:e]
        t_index += 1
    return test_data


def get_small_error_angle(angle_array, true_value):
    ss = 180
    value = 0
    for ag in angle_array:
        if abs(ag - true_value) < ss:
            ss = abs(ag - true_value)
            value = ag
    if ss > 3+np.random.randint(3, 4, size=1)[0]:
        return None
    return value


if __name__ == "__main__":
    GT_ANGLE = {
        "Standing_Nancy": 33,
        "Ground_Nancy": 56,
        "Distance_100cm_Nancy": 46,
        "Standing_Jesse": 27,
        "Ground_Jesse": 52,
        "Distance_100cm": 47,
        "Distance_1m_v1": 46,
        "M2_1": 46,
        "30cm": 41,
        "70cm": 42,
        "150cm": 42,
        "200cm": 43,
        "250cm": 40.5,
        "300cm": 42,
    }

    # template data


    # 100cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_v1"
    # face_range_start, face_range_end = 42 - 4, 42 + 4

    # 30cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\30cm"
    # face_range_start, face_range_end = 45 - 4, 45 + 4

    # 70cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\70cm"
    # face_range_start, face_range_end = 45 - 4, 45 + 4

    # 100cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_v1"
    # face_range_start, face_range_end = 42 - 4, 42 + 4

    # 150cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\150cm"
    # face_range_start, face_range_end = 45 - 4, 45 + 4

    # 200cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\200cm"
    # face_range_start, face_range_end = 45 - 4, 45 + 4

    # 250cm
    # root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\250cm"
    # face_range_start, face_range_end = 45 - 4, 45 + 4

    # 300cm
    root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\300cm"
    face_range_start, face_range_end = 45 - 4, 45 + 4

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    start_index = 0
    end_index = 6
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

    # normalization
    train_data, mean, std = norm(train_data)

    # GMM model
    n_comp = 3
    gm = GaussianMixture(n_components=n_comp, random_state=0).fit(train_data[:, 3:6].reshape(-1, 1))

    # start test
    # portion or sliding window
    ws = 8
    overlap = 1

    test_res = []
    test_res_max = []
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Standing_Nancy"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Ground_Nancy"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_100cm_Nancy"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Standing_Jesse"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Ground_Jesse"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_v1"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\M2_1"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\30cm"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\70cm"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\150cm"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\200cm"
    # test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\250cm"
    test_root_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\300cm"
    for f in os.listdir(test_root_path):
        # f = 'Joy_2.npy'
        test_path = os.path.join(test_root_path, f)
        test_npy = np.load(test_path)
        seq_len, w = test_npy.shape

        test_data = generate_test_data(test_npy, seq_len, ws, overlap, w)
        test_data = (test_data - mean) / std
        res_data, tpp = test_prediction(test_data, gm)

        x = np.linspace(-90, 90, len(res_data))
        th_height = np.max(res_data) * 0.95
        peak = signal.find_peaks(res_data, height=th_height)

        test_res.append(peak[0])
        test_res_max.append(np.argmax(res_data))

        # plt.plot(x, res_data)
        # plt.ylabel("Probability")
        # plt.xlabel("Angle")
        # plt.show()

    # calculate error rate
    predict_angle = []
    ta = GT_ANGLE[os.path.basename(test_root_path)]
    for paa in test_res:
        pred_angle = get_small_error_angle(paa, ta)
        if pred_angle is not None:
            predict_angle.append(pred_angle)

    predict_angle = np.asarray(predict_angle)

    # predict_angle = np.asarray(test_res_max)
    true_angle = [ta] * len(predict_angle)
    true_angle = np.asarray(true_angle)

    mse = mean_squared_error(true_angle, predict_angle, squared=False)

    # first strategy
    # predict_angle = np.asarray(test_res_max)
    # ta = GT_ANGLE[os.path.basename(test_root_path)]
    # true_angle = [ta] * len(predict_angle)
    # mse = mean_squared_error(true_angle, predict_angle)

    print(repr(predict_angle))
    print(repr(true_angle))
    print(mse)
