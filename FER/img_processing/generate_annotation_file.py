import os
from glob import glob
from sklearn.model_selection import train_test_split
import random
import numpy as np

# set seed, make result reporducable
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


def get_label(name):
    labels = {'Joy': 1, 'Surprise': 2, 'Anger': 3, 'Sadness': 4, 'Fear': 5, 'Disgust': 6, 'Neutral': 0}
    for key in labels.keys():
        if key in name:
            return labels[key]


if __name__ == "__main__":
    frame_root = "C:/Users/Zber/Desktop/Subjects_Frames"
    ff = ".jpg"

    str_arr = []
    labels = []

    folders = glob(frame_root + '/*/*/')
    start_index = '3'

    for f in folders:
        f = f.replace("\\", "/")
        # lens = str(len([name for name in os.listdir(f) if ff in name]))
        lens = 33
        l = str(get_label(os.path.basename(os.path.normpath(f))))
        s = " ".join([f[:-1], start_index, str(lens), l])
        str_arr.append(s)
        labels.append(int(l))

    train, test = train_test_split(str_arr,  test_size=0.2, random_state=25, stratify=labels)

    with open(os.path.join(frame_root, "annotations_train.txt"), 'w') as f:
        f.writelines('\n'.join(train))

    with open(os.path.join(frame_root, "annotations_test.txt"), 'w') as f:
        f.writelines('\n'.join(test))