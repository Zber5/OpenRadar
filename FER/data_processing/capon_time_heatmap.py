import sys
import numpy as np
import mmwave.dsp as dsp
import mmwave.dsp.music as music
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize
from scipy import signal
import os
from mmwave.dsp.utils import Window
import math
from FER.utils import parseConfigFile, arange_tx, get_label

from itertools import accumulate
from operator import add
from mmwave.dsp.cfar import ca

from scipy.signal import find_peaks, peak_widths
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
from queue import Queue

DebugMode = True

if not DebugMode:
    import matplotlib

    matplotlib.use('Agg')

# plotting color
import matplotlib._color_data as mcd
import threading

tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
extra_color = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04']
tab_color = tab_color + extra_color

# figpath = "C:/Users/Zber/Desktop/mmWave_figure"
figpath = "C:/Users/Zber/Desktop/SavedFigure"

# 3s
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'


def capon_heatmap(adc_data_path, saved_path):
    offset = 5

    # (1) Reading in adc data
    adc_data = np.fromfile(adc_data_path, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize_cli, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)
    print("Data Loaded!")

    # (1) processing range data
    # window types : Bartlett, Blackman p, Hanning p and Hamming
    # range_data = dsp.range_processing(adc_data)
    range_data = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)
    range_data = arange_tx(range_data, num_tx=numTxAntennas)

    # (2) doppler processing frame
    det_matrix, aoa_input = dsp.doppler_processing_frame(range_data, num_tx_antennas=numTxAntennas,
                                                         clutter_removal_enabled=True,
                                                         window_type_2d=Window.HAMMING,
                                                         accumulate=True)

    det_matrix_vis = np.fft.fftshift(det_matrix, axes=2)
    # det_matrix_vis_mean = np.mean(det_matrix_vis[:, :, :], axis=0)
    det_matrix_vis_mean = np.mean(det_matrix_vis[:, :, :], axis=0)
    # det_matrix_vis_mean = det_matrix_vis[54, :, :]
    bin_data = det_matrix_vis_mean[:, 17] + det_matrix_vis_mean[:, 15]
    peak_data = ca(bin_data, guard_len=2, noise_len=4, l_bound=8)
    # get peak_data
    peak_data = peak_data[offset:offset + 200]
    detect_pos = np.where(peak_data == True)[0]
    detect_pos = detect_pos + offset
    print("Detect Position: {} -> {}".format(saved_path, detect_pos))
    if len(detect_pos) == 0:
        return -1
    # detect_pos_fixed = [29, 30, 31]
    # detect_pos_fixed = [9, 10, 11, 12, 13] #30cm
    # detect_pos_fixed = [19, 20, 21, 22, 23] #70cm
    # detect_pos_fixed = [41, 42, 43] #150cm
    # detect_pos_fixed = [49, 50, 51] #200cm
    # detect_pos_fixed = [59, 60 , 61] #250cm
    # detect_pos_fixed = [69, 70] #300cm
    detect_pos_fixed = [21, 22, 23, 24, 25, 26]


    aoa_input = np.transpose(aoa_input, (0, 3, 2, 1))
    aoa_input = np.concatenate((aoa_input[:, :16, :, :], aoa_input[:, 17:, :, :]), axis=1)

    num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE_AZI, ANGLE_RES_AZI, VIRT_ANT_AZI)
    time_angle = np.zeros((ANGLE_BINS_AZI, 300))

    for i in range(0, 300):
        for r in detect_pos_fixed:
            # chirp_data = np.mean(va_data[5, :, VIRT_ANT_AZI_INDEX, i:i + 1], axis=-1)
            chirp_data = aoa_input[i, :, VIRT_ANT_AZI_INDEX, r]
            # chirp_data = np.angle(chirp_data)
            # steering_vec = np.angle(steering_vec)
            # chirp_data = va_data[50, :, VIRT_ANT_AZI_INDEX, i]
            # time_angle[:, i] += music.aoa_music_1D(steering_vec, chirp_data, 1)

            # capon beamformer
            capon_angle, beamWeights = dsp.aoa_capon(chirp_data, steering_vec, magnitude=True)
            time_angle[:, i] = time_angle[:, i] + capon_angle

    np.save(saved_path, time_angle)


def thread_job(queue, bin_path, heatmap_out_path):
    while not queue.empty():
        q = queue.get()
        bpath = os.path.join(bin_path, q)
        hpath = os.path.join(heatmap_out_path, q.replace("_Raw_0.bin", ""))
        capon_heatmap(bpath, hpath)
        queue.task_done()


def check_is_in_badfile(bad_file_list, sub, emo, idx):
    for bf in bad_file_list:
        bfname = bf.replace('\n', '').split(',')
        if sub == bfname[0] and emo == bfname[1] and idx == int(bfname[2]):
            return True
    return False


if __name__ == "__main__":
    # num Antennas
    numTxAntennas = 3
    numRxAntennas = 4
    # load configure parameters
    config = parseConfigFile(configFileName)

    # mmWave radar settings
    numFrames = config['numFrames']
    numADCSamples = config['numAdcSamples']
    numLoopsPerFrame = config['numLoops']
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
    numAngleBins = 64

    # aoa related
    VIRT_ELE_PAIRS = [[8, 2], [9, 3], [10, 4], [11, 5]]
    VIRT_AZI_PAIRS = [[i for i in range(0, 8)]]

    # azimuth
    ANGLE_RES_AZI = 2
    ANGLE_RANGE_AZI = 90
    ANGLE_BINS_AZI = (ANGLE_RANGE_AZI * 2) // ANGLE_RES_AZI + 1
    VIRT_ANT_AZI = 8

    # elevation
    ANGLE_RES_ELE = 5
    ANGLE_RANGE_ELE = 30
    ANGLE_BINS_ELE = (ANGLE_RANGE_ELE * 2) // ANGLE_RES_ELE + 1
    VIRT_ANT_ELE = 2

    BIN_RANG_S = 0
    BIN_RANG_E = 256
    BINS_PROCESSED = BIN_RANG_E - BIN_RANG_S
    VIRT_ANT_AZI_INDEX = [i for i in range(0, 8)]
    VIRT_ANT_ELE_INDEX = VIRT_ELE_PAIRS[2]

    # data processing parameter
    range_resolution, bandwidth = dsp.range_resolution(config['numAdcSamples'],
                                                       dig_out_sample_rate=config['digOutSampleRate'],
                                                       freq_slope_const=config['freqSlopeConst'])

    doppler_resolution = dsp.doppler_resolution(bandwidth, start_freq_const=config['startFreq'],
                                                ramp_end_time=config['rampEndTime'],
                                                idle_time_const=config['idleTime'],
                                                num_loops_per_frame=config['numLoops'],
                                                num_tx_antennas=numTxAntennas)

    print('Range Resolution: {:.2f}cm, Bandwidth: {:.2f}Ghz, Doppler Resolution: {:.2f}m/s'.format(
        range_resolution * 100, bandwidth / 1000000000, doppler_resolution))

    # root_path = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m"
    # root_path = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m_stand"
    # root_path = "D:\\Subjects\\Distance_300cm"
    root_path = "D:\\Subjects\\M2_1"
    # root_path = "C:\\Users\\Zber\\Desktop\\Subjects\\Distance_1m_ground"
    data_path = '{}_{}_Raw_0.bin'
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m"
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_v1"
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_stand"
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_ground"
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\Distance_1m_stand_v1"
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\300cm"
    output_data_path = "C:\\Users\\Zber\\Desktop\\Capon_Heatmap\\M2_1"
    # emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    # emotion_list = ['Joy', 'Anger', 'Sadness', 'Fear', 'Disgust']
    # emotion_list = ['Surprise']
    emotion_list = ['Joy']
    start_index = 0
    end_index = 1
    # end_index = 10

    # baf files
    with open("C:/Users/Zber/Desktop/Subjects_Heatmap_new/bad_file_list.txt") as f:
        bad_file_list = f.readlines()

    queue = Queue()
    for l, e in enumerate(emotion_list):
        for i in range(start_index, end_index):
            bin_path = os.path.join(root_path, data_path.format(e, i))
            relative_path = os.path.join(data_path.format(e, i))
            if not check_is_in_badfile(bad_file_list, os.path.basename(output_data_path), e, i):
                queue.put(relative_path)

    thread_job(queue, root_path, output_data_path)

    # NUM_THREADS = 12
    # for i in range(NUM_THREADS):
    #     worker = threading.Thread(target=thread_job, args=(queue, root_path, output_data_path))
    #     worker.start()
    #
    # print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    # print('This can take an hour or two depending on dataset size')
    # queue.join()
    # print('all done')
