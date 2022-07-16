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
from FER.utils import MapRecord, get_label
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


def check_is_in_badfile(bad_file_list, sub, emo, idx):
    for bf in bad_file_list:
        bfname = bf.replace('\n', '').split(',')
        if sub == bfname[0] and emo == bfname[1] and idx == int(bfname[2]):
            return True
    return False


if __name__ == "__main__":
    # train_form = "{}_train_Distance_v1.txt"
    # test_form = "{}_test_Distance_v1.txt"
    #
    # root_path = "C:/Users/Zber/Desktop/Subjects_Heatmap"
    # train_path = os.path.join(root_path, train_form.format('heatmap'))
    # test_path = os.path.join(root_path, test_form.format('heatmap'))
    #
    # frame_root_path = "C:/Users/Zber/Desktop/Subjects_Frames"
    # frame_train_path = os.path.join(frame_root_path, train_form.format('frames'))
    # frame_test_path = os.path.join(frame_root_path, test_form.format('frames'))

    # hm_2_frame("", train_path, frame_train_path)
    # hm_2_frame("", test_path, frame_test_path)

    # landmark
    # target_subjects = ["S0", "S1", "S2", "S3", "S4"]
    # heatmap_dir = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap"
    # full_anno_file = os.path.join(heatmap_dir, "heatmap_annotation_full_S8.txt")
    # root_path = ""
    # record_list = [MapRecord(x.strip().split(), root_path) for x in open(full_anno_file)]
    # filter_record_list = []
    #
    # # filter_record
    # for re in record_list:
    #     if re.path[:2] in target_subjects:
    #         filter_record_list.append(re)
    #
    # # train_test_splite
    # train, test = data_split(filter_record_list)
    #
    # # write into landmark file
    # landmark_dir = "C:/Users/Zber/Desktop/Subjects_Landmark_video"
    # # write into landmark file
    # landmark_train = os.path.join(landmark_dir, "landmark_S5_train.txt")
    # landmark_test = os.path.join(landmark_dir, "landmark_S5_test.txt")
    # hm_2_landmark_v1(train, landmark_train)
    # hm_2_landmark_v1(test, landmark_test)
    #
    # # write into heatmap file
    # heatmap_train = os.path.join(heatmap_dir, "heatmap_landmark_S5_train.txt")
    # heatmap_test = os.path.join(heatmap_dir, "heatmap_landmark_S5_test.txt")
    # append_record_to_file(train, heatmap_train)
    # append_record_to_file(test, heatmap_test)

    # heatmap
    ONSET = 31
    PEAK = 130
    npy_form = "{}.npy"
    format_string = "{}/{}_{}_{} {} {} {} -1 0 0 0"
    # target_subjects = ['S{}'.format(i) for i in range(0, 6)]
    # target_subjects = ['S{}'.format(i) for i in range(0, 2)] + ['S{}'.format(i) for i in range(3, 10)]
    # target_subjects = ['W{}'.format(i) for i in range(1, 4)]
    # target_subjects = ['W{}'.format(i) for i in range(1, 2)]
    # target_subjects = ['W3']
    # target_subjects = ['Standing_Jesse', 'Standing_Nancy', 'Ground_Jesse', 'Ground_Nancy', 'Distance_100cm']
    # target_subjects = ['Standing_Jesse']
    # target_subjects = ['Distance_250cm']  # , 'Distance_200cm', 'Distance_150cm']
    # target_subjects = ['Distance_300cm']  # , 'Distance_200cm', 'Distance_150cm']
    # target_subjects = ['M1_2', 'M2_2', 'M3_2']
    # target_subjects = ['M1_0', 'M1_1', 'M1_2', 'M2_0', 'M2_1', 'M2_2', 'Distance_100cm']
    # target_subjects = ['M1_2', 'M2_2', 'M3_2']
    # target_subjects = ['M3_2']
    # target_subjects = ['Camera']

    # subs = ['30cm_30d', '30cm_60d', '30cm_90d', '70cm_30d', '70cm_60d', '70cm_90d']
    # subs = ['100cm_30d', '100cm_60d', '100cm_90d', '150cm_30d', '150cm_60d', '150cm_90d']
    # subs = ['200cm_30d', '200cm_60d', '200cm_90d', '250cm_30d', '250cm_60d', '250cm_90d']
    # subs = ['300cm_30d', '300cm_60d', '300cm_90d']

    #'30cm_30d', '30cm_60d', '30cm_90d'
    target_subjectss = [['30cm_30d'],['70cm_30d'],['100cm_30d'],['150cm_30d'],['200cm_30d'],['250cm_30d'],['300cm_30d']]
    # target_subjects = ['70cm_30d']
    # target_subjects = ['100cm_30d']
    # target_subjects = ['150cm_30d']
    # target_subjects = ['200cm_30d']
    # target_subjects = ['250cm_30d']
    # target_subjects = ['300cm_30d']

    # target_subjects = ['300cm_30d', '300cm_60d', '300cm_90d']
    # target_subjects = ['200cm_30d', '200cm_60d', '200cm_90d', '250cm_30d', '250cm_60d', '250cm_90d',
    #                    '300cm_30d', '300cm_60d', '300cm_90d']

    # target_subjects = ['Ground_Jesse', 'Ground_Jesse']
    # target_subjects = ['Distance_100cm', 'Distance_100cm_Nancy']
    # target_subjects = ['S{}'.format(i) for i in range(0, 10)]
    # target_subjects = ['S0', 'S9']

    # train_index = [i for i in range(10)] + [i for i in range(20, 30)]
    # train_index = [i for i in range(0, 10)] + [i for i in range(20, 30)]
    # train_index = [i for i in range(0, 4)] + [i for i in range(7, 10)]
    # train_index = [i for i in range(0, 6)]
    # train_index = [i for i in range(0, 4)] + [i for i in range(7, 10)]
    # test_index = [i for i in range(10, 20)]
    # test_index = [i for i in range(8, 10)]
    # test_index = [i for i in range(30)]

    for target_subjects in target_subjectss:

        train_index = [i for i in range(0, 4)]
        test_index = [i for i in range(4, 10)]

        # test_index = [i for i in range(4, 7)]
        emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

        with open("C:/Users/Zber/Desktop/Subjects_Heatmap_new/bad_file_list.txt") as f:
            bad_file_list = f.readlines()

        # root_path = "C:/Users/Zber/Desktop/Subjects_Heatmap_new"
        root_path = "C:/Users/Zber/Desktop/Subjects_Heatmap"
        # train_form = "{}_train_S0_1_2.txt"
        # test_form = "{}_test_S0_1_2.txt"

        # train_form = "{}_train_S3_4_5.txt"
        # test_form = "{}_test_S3_4_5.txt"

        # train_form = "{}_train_landmark_S5.txt"
        # test_form = "{}_test_landmark_S5.txt"

        train_form = "{}_train_{}.txt".format('{}', target_subjects[0])
        test_form = "{}_test_{}.txt".format('{}', target_subjects[0])


        # train_form = "{}_train_M3_2.txt"
        # test_form = "{}_test_M3_2.txt"

        train_path = os.path.join(root_path, train_form.format('heatmap'))
        test_path = os.path.join(root_path, test_form.format('heatmap'))

        frame_root_path = "C:/Users/Zber/Desktop/Subjects_Frames"
        frame_train_path = os.path.join(frame_root_path, train_form.format('frames'))
        frame_test_path = os.path.join(frame_root_path, test_form.format('frames'))

        train_list = []
        test_list = []
        for s in target_subjects:
            for e in emotion_list:
                label = get_label(e)
                for train_id in train_index:
                    if not check_is_in_badfile(bad_file_list, s, e, train_id):
                        train_list.append(format_string.format(s, e, train_id, npy_form, label, ONSET, PEAK))
                for test_id in test_index:
                    if not check_is_in_badfile(bad_file_list, s, e, test_list):
                        test_list.append(format_string.format(s, e, test_id, npy_form, label, ONSET, PEAK))

        np.random.shuffle(train_list)
        np.random.shuffle(test_list)
        with open(train_path, 'w') as f:
            f.writelines('\n'.join(train_list))
        print("Write {} Records to txt file".format(len(train_list)))

        with open(test_path, 'w') as f:
            f.writelines('\n'.join(test_list))
        print("Write {} Records to txt file".format(len(test_list)))

        # heatmap to frame
        hm_2_frame("", train_path, frame_train_path)
        hm_2_frame("", test_path, frame_test_path)

    # cross ss
