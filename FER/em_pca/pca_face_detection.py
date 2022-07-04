import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from time import time
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

# get the data
root_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"

face_angle = 42
face_range_start, face_range_end = 42 - 4, 42 + 4
# emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
start_index = 0
end_index = 9
npy_format = "{}_{}.npy"

all_data = np.zeros((len(emotion_list) * (10), 8, 300))

# x_train
a_index = 0
for e in emotion_list:
    for i in range(start_index, end_index):
        bin_path = os.path.join(root_dir, npy_format.format(e, i))
        npy_data = np.load(bin_path)
        all_data[a_index] = npy_data[face_range_start:face_range_end]
        a_index += 1

# x_test
# emotion_list = ['Joy']
# start_index = 0
# end_index = 9
#emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

# test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"
test_dir = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"
test_data_path = os.path.join(test_dir, npy_format.format('Joy', 9))
test_data = np.load(test_data_path)
seq_len, w = test_data.shape
ws = 8
overlap = 1


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


all_test_data = np.zeros(((seq_len - 8) // 1 + 1, 8, 300))

t_index = 0
for s, e in window(seq_len, 8, 1):
    all_test_data[t_index] = test_data[s:e]
    t_index += 1

# principle component analysis

# 1. normaliztion
scaler = StandardScaler()
n_samples, h, w = all_data.shape
test_n_samples, test_h, test_w = all_test_data.shape
all_data = np.reshape(all_data, (n_samples, -1))
all_test_data = np.reshape(all_test_data, (all_test_data.shape[0], -1))
X_train = scaler.fit_transform(all_data)
X_test = scaler.fit_transform(all_test_data)

# reshape the data

X_train = np.reshape(X_train, (n_samples, h, w))
X_test = np.reshape(X_test, (test_n_samples, test_h, test_w))

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


X_train = np.sum(X_train, -1)
X_test = np.sum(X_test, -1)

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
        test_sim[i_train] = cor(test[:], train[:])
        # test_sim[i_train] = cos_sim(test[0:5], train[0:5])
    # sim_data[i_test] = 1/np.max(test_sim)
    # sim_data[i_test] = np.var(test_sim) * np.mean(test_sim)
    sim_data[i_test] = np.mean(test_sim)

plt.plot(sim_data)
plt.axvline(42)
plt.show()

print("")
