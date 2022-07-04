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
            f.write(str_format.format(record.path + "_{}.npy", record.label, record.onset, record.peak,
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
    bad_file_path = "C:/Users/Zber/Desktop/Subjects_Heatmap_new/bad_file_list.txt"

    file_path = "C:/Users/Zber/Desktop/Subjects_Heatmap/heatmap_annotation_train_new.txt"
    target_file_path = "C:/Users/Zber/Desktop/Subjects_Heatmap_new/heatmap_annotation_train_s5.txt"

    # file_path = "C:/Users/Zber/Desktop/Subjects_Heatmap/heatmap_annotation_test_new.txt"
    # target_file_path = "C:/Users/Zber/Desktop/Subjects_Heatmap_new/heatmap_annotation_test_s5.txt"

    with open(bad_file_path, 'r') as bad_file:
        bad_file_list = bad_file.readlines()

    with open(file_path, 'r') as file:
        file_list = file.readlines()

    for f in file_list:
        sub = f[:2]
        ids = [i for i in range(len(f)) if f[i] == "_"]
        emo = f[3:ids[0]]
        idx = f[ids[0] + 1: ids[1]]

        matches = [sub, idx, emo]

        for bf in bad_file_list:
            bfname = bf.replace('\n','').split(',')
            if sub==bfname[0] and emo==bfname[1] and idx==bfname[2]:
                file_list.remove(f)
                print(f)
                continue

    # with open(target_file_path, 'w') as target_file:
    #     target_file.writelines(file_list)