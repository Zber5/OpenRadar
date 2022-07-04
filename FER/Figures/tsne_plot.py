import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.colors as col

os.chdir("C:/Users/Zber/Documents/Dev_program/OpenRadar")
from FER.utils import MapRecord, get_label
from sklearn.model_selection import train_test_split
import matplotlib._color_data as mcd

tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
extra_color = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04']
tab_color = tab_color + extra_color


def check_is_in_badfile(bad_file_list, sub, emo, idx):
    for bf in bad_file_list:
        bfname = bf.replace('\n', '').split(',')
        if sub == bfname[0] and emo == bfname[1] and idx == int(bfname[2]):
            return True
    return False


def normalize(data, is_azi=True):
    azi_para = [73.505790, 3.681510]
    ele_para = [86.071959, 5.921158]
    if is_azi:
        return (data - azi_para[0]) / azi_para[1]
    else:
        return (data - ele_para[0]) / ele_para[1]


if __name__ == "__main__":
    target_subjects = ['S{}'.format(i) for i in range(0, 3)]
    npy_form = "{}.npy"
    ONSET = 31
    PEAK = 130
    data_index = [i for i in range(0, 10)]
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    root_path = "C:/Users/Zber/Desktop/Subjects_Heatmap"
    path_list = []
    labels = []
    subs = []
    format_string = "{}/{}_{}_{}"
    with open("C:/Users/Zber/Desktop/Subjects_Heatmap_new/bad_file_list.txt") as f:
        bad_file_list = f.readlines()

    for s in target_subjects:
        for e in emotion_list:
            label = get_label(e)
            for iid in data_index:
                if not check_is_in_badfile(bad_file_list, s, e, iid):
                    path_list.append(format_string.format(s, e, iid, npy_form))
                    labels.append(label)
                    subs.append(s.replace('S', ''))

    # read data into numpy
    npy_data = np.zeros((len(path_list), 91, 10))

    for i, path in enumerate(path_list):
        fp = os.path.join(root_path, path)
        azi_data = np.load(fp.format('azi'))
        azi_data = normalize(azi_data, is_azi=True)
        azi_data = np.mean(azi_data[ONSET:PEAK], axis=0)
        npy_data[i] = azi_data

    npy_data = npy_data.reshape((len(path_list), -1))
    cd = np.asarray(labels)
    subs = np.asarray(subs, dtype=np.int)

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(npy_data)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    markers = ["o", "^", "s"]

    for s in range(0, 3):
        sub_index = (subs == s)
        ax.scatter(X_tsne[sub_index, 0], X_tsne[sub_index, 1], marker=markers[s], c=cd[sub_index],
                   cmap=col.ListedColormap(tab_color))

    # plt.tick_params(left=False, bottom=False)
    plt.show()
