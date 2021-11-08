import numpy as np
from sklearn.model_selection import train_test_split


def twin_sliding_window(imag_length, sensor_length, imag_window=6, sensor_window=20, imag_step=3, sensor_step=10):
    imag_start = 0
    sensor_start = 0

    while (imag_start + imag_window) <= imag_length and (sensor_start + sensor_window) <= sensor_length:
        yield (imag_start, imag_start + imag_window), (sensor_start, sensor_start + sensor_window)
        imag_start += imag_step
        sensor_start += sensor_step


def format_dataset(sensor, imag, label, imag_window=9, sensor_window=30, imag_step=3, sensor_step=10):
    num_data = np.shape(sensor)[0]
    sensor_channels = np.shape(sensor)[1]
    imag_channels = np.shape(imag)[1]
    sensor_length = np.shape(sensor)[2]
    imag_length = np.shape(imag)[2]

    intensity_unit = 0.1

    num_segment = (imag_length - imag_window) // imag_step + 1

    x = np.zeros((num_segment * num_data, sensor_channels, sensor_window))
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


if __name__ == '__main__':
    # emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # saved_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/ml_{}_b8_c5_{}.npy"
    # saved_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/ml_{}_b8_c5_rf_{}.npy"
    saved_path = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/ml_b8_c5_rf_{}.npy"

    # load data
    sensor_data_path = '/data/sensor_b8_c5_x.npy'
    label_path = '/data/sensor_b8_c5_y.npy'

    # aus data
    # flm_score_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/aus_JAANET1.npy'
    flm_score_path = '/data/aus_rf.npy'

    x = np.load(sensor_data_path)
    y = np.load(flm_score_path)
    label = np.load(label_path)

    # formatting data
    y = y.transpose((0, 2, 1))

    x_train, x_test, y_train, y_test, label_train, label_test = train_test_split(x, y, label, test_size=0.2,
                                                                                 random_state=25, stratify=label)

    # processing train data
    x_train, y_train, label_train = format_dataset(x_train, y_train, label_train)

    # processing test data
    x_test, y_test, label_test = format_dataset(x_test, y_test, label_test)

    # save sensor image x y label to npy
    np.save(saved_path.format('x_train'), x_train)
    np.save(saved_path.format('x_test'), x_test)

    np.save(saved_path.format('y_train'), y_train)
    np.save(saved_path.format('y_test'), y_test)

    np.save(saved_path.format('label_train'), label_train)
    np.save(saved_path.format('label_test'), label_test)
