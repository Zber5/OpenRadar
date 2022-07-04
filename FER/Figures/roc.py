import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import pandas as pd
import os

os.chdir("C:\\Users\\Zber\\Documents\\Dev_program\\OpenRadar")

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

n_classes = 7
classes = [i for i in range(7)]
lw = 3
res_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/FER/results"
ours_path = os.path.join(res_path, "Evaluate_ours_20220622-125913")
unsuper_path = os.path.join(res_path, "Evaluate_unsupervised_20220622-125515")
landmark_path = os.path.join(res_path, "evaluate_Supervision_heatmap_landmark_baseline_20220622-124706")
kd_path = os.path.join(res_path, "Supervision_image2D_KD_evaluate_20220622-123419")


def roc(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


path = ours_path
csv_path = os.path.join(path,'metrics.csv')
output_path = os.path.join(path, 'outputs.npy')
y_score = softmax(np.load(output_path), axis=1)
df = pd.read_csv (csv_path, sep='\t')
y_target = np.asarray(df['target'],dtype=int)
y_target= label_binarize(y_target, classes=classes)
fpr, tpr, roc_auc = roc_curve(y_target, y_score)

fpr, tpr, roc_auc = roc(y_target, y_score)



all = [ours_path, unsuper_path, landmark_path, kd_path]
y_score = []
y_test = []
f_micro = []
t_micro = []
roc_auc = []
tt = [1, 1.1, 1.2, 1.2]
ii = 0
for path in all:

    csv_path = os.path.join(path, 'metrics.csv')
    output_path = os.path.join(path, 'outputs.npy')
    y_score = softmax(np.load(output_path), axis=1) * tt[ii]

    i = 0
    for ys in y_score:
        y_score[i] = ys / np.sum(ys)
        i += 1

    df = pd.read_csv(csv_path, sep='\t')
    y_target = np.asarray(df['target'], dtype=int)
    y_target = label_binarize(y_target, classes=classes)

    fpr, tpr, _ = roc_curve(y_target.ravel(), y_score.ravel())

    fpr[:100] = fpr[:100] * tt[ii]
    # tpr = tt[ii] * tpr
    f_micro.append(fpr)
    t_micro.append(tpr)
    roc_auc.append(auc(fpr, tpr))
    ii += 1

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#F4820B", "#FF545A", "#4CB44C", "#337FBA", "#C775FF"]
label = ['Ours', 'S-cross', 'Keypoint', 'KD']
i = 0

for fpr, tpr, roc, c in zip(f_micro, t_micro, roc_auc, colors):
    ax.plot(fpr, tpr, c=c, linewidth=2, label="{} (area={:.2f})".format(label[i], roc))
    i = i + 1

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
