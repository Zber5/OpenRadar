import numpy as np
from utils import twin_sliding_window
import pandas as pd
from IPython.display import display


def format_dataset(sensor, imag, label, imag_window=12, sensor_window=40, imag_step=6, sensor_step=20,
                   intensity_unit=0.2):
    num_data = np.shape(sensor)[0]
    sensor_channels = np.shape(sensor)[1]
    sensor_bins = np.shape(sensor)[3]
    imag_channels = np.shape(imag)[1]
    sensor_length = np.shape(sensor)[2]
    imag_length = np.shape(imag)[2]

    num_segment = (imag_length - imag_window) // imag_step + 1

    x = np.zeros((num_segment * num_data, sensor_channels, sensor_window, sensor_bins))
    y = np.zeros((num_segment * num_data, imag_channels))
    l = np.zeros((num_segment * num_data))
    index = 0
    for i in range(num_data):
        label_content = label[i]
        for imag_idx, sensor_idx in twin_sliding_window(imag_length, sensor_length, imag_window, sensor_window,
                                                        imag_step, sensor_step):
            sensor_start, sensor_end = sensor_idx
            imag_start, imag_end = imag_idx
            x[index] = sensor[i, :, sensor_start:sensor_end]
            # imag_mean = np.mean(imag[i, :, imag_start:imag_end], axis=1)
            imag_diff = imag[i, :, imag_end] - imag[i, :, imag_start]
            cond = imag_diff > intensity_unit
            binary_label = cond.astype(int)
            y[index] = binary_label
            l[index] = label_content
            index += 1

    return x, y, l


if __name__ == "__main__":

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    RF_AU = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU15', 'AU17',
             'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43']

    JANNET_AU = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']

    # load data
    sensor_data_path = '//data/sensor_b8r3_c5_x.npy'
    flm_score_path_rf = '//data/aus_rf.npy'
    flm_score_path_jannet = '//data/aus_jannet.npy'
    label_path = '//data/sensor_b8r3_c5_y.npy'

    sensor_data_path_1 = '//data/sensor_b8r3_c5_x_s40_e80.npy'
    flm_score_path_rf_1 = '//data/aus_rf_s40_e80.npy'
    flm_score_path_jannet_1 = '//data/aus_jannet_s40_e80.npy'
    label_path_1 = '//data/sensor_b8r3_c5_y_s40_e80.npy'
    sensor = np.load(sensor_data_path)
    imag_rf = np.load(flm_score_path_rf)
    imag_jannet = np.load(flm_score_path_jannet)
    label = np.load(label_path)

    sensor_1 = np.load(sensor_data_path_1)
    imag_rf_1 = np.load(flm_score_path_rf_1)
    imag_jannet_1 = np.load(flm_score_path_jannet_1)
    label_1 = np.load(label_path_1)

    sensor = np.concatenate((sensor, sensor_1), axis=0)
    imag_rf = np.concatenate((imag_rf, imag_rf_1), axis=0)
    imag_jannet = np.concatenate((imag_jannet, imag_jannet_1), axis=0)
    label = np.concatenate((label, label_1), axis=0)

    imag_rf = imag_rf.transpose((0, 2, 1))
    imag_jannet = imag_jannet.transpose((0, 2, 1))

    x_r, y_r, label_r = format_dataset(sensor, imag_rf, label)
    x_j, y_j, label_j = format_dataset(sensor, imag_jannet, label)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # RF AUs
    auname = RF_AU
    np_count = np.zeros((6, len(auname)))
    for l in range(6):
        l_loc = (label_r == l)
        sum_num = []
        for i in range(len(auname)):
            l_where = (label_r == l)
            ss = np.sum(y_r[l_loc, i])
            sum_num.append(int(ss))
        np_count[l] = sum_num
    df = pd.DataFrame(data=np_count, columns=auname, dtype=np.int32)

    df.insert(loc=0, column='Expression', value=emotion_list)

    display(df)

    print("\n\n")

    # RF AUs
    auname = JANNET_AU
    np_count = np.zeros((6, len(auname)))
    for l in range(6):
        l_loc = (label_j == l)
        sum_num = []
        for i in range(len(auname)):
            ss = np.sum(y_j[l_loc, i])
            sum_num.append(int(ss))
        np_count[l] = sum_num
    df = pd.DataFrame(data=np_count, columns=auname, dtype=np.int32)

    df.insert(loc=0, column='Expression', value=emotion_list)

    display(df)
