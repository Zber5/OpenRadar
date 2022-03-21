import os
import math
from FER.utils import MapRecord

def rename(old_name):
    old_path = os.path.join(data_folder, old_name)
    new_path = old_path.replace('_2_', '_')
    os.rename(old_path, new_path)
    print("Rename from {} to {}".format(old_path, new_path))


def annotation_attention(record_list, width=30):
    for record in record_list:
        record.onset = math.floor(record.onset * 3 / 10)
        record.peak = record.onset + width - 1
        record.path = record.relative_path.replace("_{}.npy","")
    return record_list




if __name__ =="__main__":
    # Eliminate
    text_path = "C:\\Users\\Zber\Desktop\\invalid_data.txt"
    annotaton_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap\\heatmap_annotation_train_S8.txt"
    root_path = ""

    record_list = [MapRecord(x.strip().split(), root_path) for x in open(annotaton_path)]

    print(len(record_list))
    with open(text_path) as f:
        lines = f.readlines()

    count = 0
    for l in lines:
        l = l[:-1]
        for index, r in enumerate(record_list):
            if l in r.path:
                print(r.path)
                record_list.pop(index)
                count += 1
    print(count)





    root_path = "C:\\Users\\Zber\\Desktop\\Subjects_Frames"
    # annotaton_path = "D:\\Subjects\\annotations_v2.txt"
    # annotaton_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap\\heatmap_annotation.txt"
    annotaton_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap\\heatmap_annotation_train.txt"
    # annotaton_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap\\heatmap_annotation_test.txt"

    record_list = [MapRecord(x.strip().split(), root_path) for x in open(annotaton_path)]

    # new_record_list = annotation_update(record_list)
    #
    # # str format: path, label, onset, peak, offset, widthError, heightError, indexError
    # str_format = "{} {} {} {} {} {} {} {}"
    # with open(os.path.join(root_path, "heatmap_annotation.txt"), 'a') as f:
    #     for record in new_record_list:
    #         f.write(str_format.format(record.path, record.label, record.onset, record.peak, record.offset, record.width_err, record.height_err, record.index_err)+'\n')
    # print("Write {} Records to txt file".format(len(new_record_list)))

    # train, test = data_split(record_list)

    # str format: path, label, onset, peak, offset, widthError, heightError, indexError
    # str_format = "{} {} {} {} {} {} {} {}"
    # with open(os.path.join(root_path, "heatmap_annotation_train.txt"), 'a') as f:
    #     for record in train:
    #         f.write(str_format.format(record.path, record.label, record.onset, record.peak, record.offset, record.width_err, record.height_err, record.index_err)+'\n')
    # print("Write {} Records to txt file".format(len(train)))
    #
    # with open(os.path.join(root_path, "heatmap_annotation_test.txt"), 'a') as f:
    #     for record in test:
    #         f.write(str_format.format(record.path, record.label, record.onset, record.peak, record.offset, record.width_err, record.height_err, record.index_err)+'\n')
    # print("Write {} Records to txt file".format(len(test)))

    # str_arr = annotation_append()
    # with open(annotaton_path, 'a') as f:
    #     f.writelines('\n'.join(str_arr))
    # print("Write {} Records to txt file".format(len(str_arr)))

    new_record_list = annotation_attention(record_list)
    str_format = "{} {} {} {}\n"
    with open(os.path.join(root_path, "annotations_att_train.txt"), 'a') as f:
        for record in new_record_list:
            f.write(str_format.format(record.path, record.onset, record.peak, record.label))
    print("Write {} Records to txt file".format(len(new_record_list)))

"""
if __name__ == "__main__":

    start_index = 30
    end_index = 40
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']

    data_folder = 'D:\\Subjects\\S2'

    # rename files
    for emo in emotion_list:
        for idx in range(start_index, end_index):

            # svo
            svo_name = "{}_2_{}.svo".format(emo, idx)
            rename(svo_name)

            # bin
            bin_name = "{}_2_{}_Raw_0.bin".format(emo, idx)
            rename(bin_name)

            # log
            log_name = "{}_2_{}_Raw_LogFile.csv".format(emo, idx)
            rename(log_name)
"""