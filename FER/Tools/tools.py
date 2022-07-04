import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.lines import Line2D
import os
import json
import matplotlib.ticker as ticker
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from scipy.interpolate import make_interp_spline, BSpline

os.chdir("C:/Users/Zber/Documents/Dev_program/OpenRadar")
from FER.utils import MapRecord
from sklearn.model_selection import train_test_split


def rename(old_name):
    old_path = os.path.join(data_folder, old_name)
    new_path = old_path.replace('_2_', '_')
    os.rename(old_path, new_path)
    print("Rename from {} to {}".format(old_path, new_path))


def annotation_update(record_list, width=100, total_frame=300):
    for record in record_list:
        # if record.num_frames < width:
        #     pad = (width - record.num_frames)//2
        #     if record.onset < pad:
        #         record.peak += pad*2

        #     elif (total_frame - record.peak) < pad:
        #         record.onset -= pad*2
        #     else:
        #         record.onset -= pad
        #         record.peak += pad
        # else:
        #     pad = record.num_frames - width
        #     record.peak -= pad

        record.path = record.path.replace("Raw_0.bin", "{}.npy").replace("\\", "/")

        if record.num_frames != 100:
            record.peak += 1
        assert record.num_frames == 100, 'the num of frames must equal to 100!'
    return record_list


def annotation_attention(record_list, width=30):
    for record in record_list:
        record.onset = math.floor(record.onset * 3 / 10)
        record.peak = record.onset + width - 1
        record.path = record.relative_path.replace("_{}.npy", "")
    return record_list


def annotation_append(subs=['S6', 'S7']):
    str_arr = []
    str_format = "{} {} {} {} {} {} {} {}"
    npy_path = "{}_{}"
    emotion = 'Neutral'
    # subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']

    for sub in subs:
        for i in range(0, 30):
            path = (os.path.join(sub, npy_path.format(emotion, i, )) + '_{}.npy').replace("\\", "/")
            label = "0"
            onset = 31
            peak = 130
            offset = -1
            e1 = 0
            e2 = 0
            e3 = 0
            str_arr.append(str_format.format(path, label, onset, peak, offset, e1, e2, e3))
    return str_arr


def data_split(record_list):
    labels = [r.label for r in record_list]
    train, test = train_test_split(record_list, test_size=0.2, random_state=25, stratify=labels)
    return train, test


def hm_2_frame(root_path, hm_path, frame_path):
    record_list = [MapRecord(x.strip().split(), root_path) for x in open(hm_path)]

    new_record_list = annotation_attention(record_list)
    str_format = "{} {} {} {}\n"
    with open(frame_path, 'w') as f:
        for record in new_record_list:
            f.write(str_format.format(record.path, record.onset, record.peak, record.label))
    print("Write {} Records to txt file".format(len(new_record_list)))


def hm_2_landmark(root_path, hm_path, frame_path):
    record_list = [MapRecord(x.strip().split(), root_path) for x in open(hm_path)]

    new_record_list = annotation_attention(record_list)
    str_format = "{} {} {} {}\n"
    with open(frame_path, 'w') as f:
        for record in new_record_list:
            re_pa = record.path.replace("_{}", "") + ".npy"
            f.write(str_format.format(re_pa, record.onset, record.peak, record.label))
    print("Write {} Records to txt file".format(len(new_record_list)))


def append_record_to_file(record_list, file):
    str_format = "{} {} {} {} {} {} {} {}"
    with open(file, 'a') as f:
        for record in record_list:
            f.write(str_format.format(record.path+"_{}.npy", record.label, record.onset, record.peak,
                                      record.offset, record.width_err, record.height_err, record.index_err) + '\n')
    print("Write {} Records to txt file".format(len(record_list)))


def hm_2_landmark_v1(record_list, frame_path):
    record_list = record_list

    new_record_list = annotation_attention(record_list)
    str_format = "{} {} {} {}\n"
    with open(frame_path, 'w') as f:
        for record in new_record_list:
            re_pa = record.path.replace("_{}", "") + ".npy"
            f.write(str_format.format(re_pa, record.onset, record.peak, record.label))
    print("Write {} Records to txt file".format(len(new_record_list)))


if __name__ == "__main__":
    target_subjects = ["S0", "S1", "S2", "S3", "S4"]
    heatmap_dir = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap"
    full_anno_file = os.path.join(heatmap_dir, "heatmap_annotation_full_S8.txt")
    root_path = ""
    record_list = [MapRecord(x.strip().split(), root_path) for x in open(full_anno_file)]
    filter_record_list = []

    # filter_record
    for re in record_list:
        if re.path[:2] in target_subjects:
            filter_record_list.append(re)

    # train_test_splite
    train, test = data_split(filter_record_list)

    # write into landmark file
    landmark_dir = "C:/Users/Zber/Desktop/Subjects_Landmark_video"
    # write into landmark file
    landmark_train = os.path.join(landmark_dir, "landmark_S5_train.txt")
    landmark_test = os.path.join(landmark_dir, "landmark_S5_test.txt")
    hm_2_landmark_v1(train, landmark_train)
    hm_2_landmark_v1(test, landmark_test)

    # write into heatmap file
    heatmap_train = os.path.join(heatmap_dir, "heatmap_landmark_S5_train.txt")
    heatmap_test = os.path.join(heatmap_dir, "heatmap_landmark_S5_test.txt")
    append_record_to_file(train, heatmap_train)
    append_record_to_file(test, heatmap_test)


