import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from em_plot.antenna_array import parseConfigFile
from mmwave.dsp.utils import Window

# configure file
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange_modify.cfg'

# 1.get all data
# data_path = 'C:/ti/mySavedData/{}_{}_Raw_0.bin'
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
data_path = 'C:/Users/Zber/Desktop/SavedData_MIMO/{}_{}_Raw_0.bin'
emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
save_path = 'C:/Users/Zber/Desktop/SavedData_MIMO'
num_bin = 3

if __name__ == '__main__':

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    # emotion_list = ['Neutral', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/sensor_b{}_c{}_{}.npy'
    save_path = '//data/sensor_b{}r3_c{}_{}_s40_e80.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_8-10-5-3_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_7-9-6-4_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_7-9-6-4_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_all_bi{}_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff_segment_8-10-5-3_{}.npy'
    # save_path = 'C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/Emotion/data/emotion_3s_diff1_segment_{}.npy'

    # start index
    start_index = 40
    end_index = 80
    num_segment = 80
    num_data = len(emotion_list) * (end_index - start_index)

    # bool
    is_diff = True
    is_segment = False
    # num Antennas
    numTxAntennas = 3
    numRxAntennas = 4

    # load configure
    configParameters = parseConfigFile(configFileName, numTxAnt=numTxAntennas)

    # mmWave radar settings
    numFrames = configParameters['numFrames']
    numADCSamples = configParameters['numAdcSamples']

    numLoopsPerFrame = configParameters['numLoops']
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

    numRangeBins = numADCSamples
    numDopplerBins = numLoopsPerFrame
    numAngleBins = 64

    # virtual Antenna Array
    device_v = True
    # antenna all
    antenna_order = [i for i in range(1, 13)]
    # antenna_order = [8, 10, 7, 9, 6, 4, 5, 3] if device_v else [5, 6, 7, 8, 3, 4, 9, 10]
    # antenna_order = [8, 10, 7, 9, 6, 4, 5, 3] if device_v else [5, 6, 7, 8, 3, 4, 9, 10]
    # antenna_order = [7, 9, 6, 4] if device_v else [6, 7, 4, 9]
    # antenna_order = [8, 10, 5, 3] if device_v else [5, 8, 3, 10]
    virtual_array = []

    for tx in range(1, 4):
        for rx in range(1, 5):
            virtual_array.append((tx, rx))
    tx_map = {1: 0, 3: 1, 2: 2}

    # data processing parameter
    numDisplaySamples = 20
    bin_index = 8
    chirp_index = 5

    if is_diff:
        numSample = numFrames - 1
    else:
        numSample = numFrames

    if is_segment:
        numSample = num_segment

    data = np.zeros((num_data, len(antenna_order), numSample, num_bin))
    label = np.zeros(num_data)

    index = 0
    for l, e in enumerate(emotion_list):
        for i in range(start_index, end_index):
            bin_path = data_path.format(e, i)
            # load data
            adc_data = np.fromfile(bin_path, dtype=np.int16)
            adc_data = adc_data.reshape(numFrames, -1)
            adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                           num_rx=numRxAntennas, num_samples=numADCSamples)
            range_data = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)

            num_bin_side = num_bin // 2

            range_data = range_data[..., bin_index-num_bin_side:bin_index+num_bin_side+1]

            sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas, num_bin))
            sig = sig[:, :, chirp_index, :, :]

            c_data = np.zeros((len(antenna_order), numSample, num_bin))
            for c_i, o in enumerate(antenna_order):
                va_order = o - 1
                t, r = virtual_array[va_order]
                t, r = tx_map[t], r - 1
                va_sig = sig[:, t, r]
                va_phase = np.angle(va_sig)
                va_unwrap_phase = np.unwrap(va_phase, axis=0)

                if is_diff:
                    # c_data[c_i] = va_diff_phase[20:100]
                    va_diff_phase = np.diff(va_unwrap_phase, axis=0)
                    c_data[c_i] = va_diff_phase
                else:
                    c_data[c_i] = va_unwrap_phase

            # data[index] = c_data.reshape(-1)

            data[index] = c_data
            label[index] = l
            index += 1
            # va_diff_phase = va_unwrap_phase

        print("{} data loaded!".format(e))
    np.save(save_path.format(bin_index, chirp_index, 'x'), data)
    np.save(save_path.format(bin_index, chirp_index, 'y'), label)
    print("Finished")
