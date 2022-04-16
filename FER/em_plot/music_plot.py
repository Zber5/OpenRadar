import sys
import numpy as np
import mmwave.dsp as dsp
import mmwave.dsp.music as music
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize
from data_processing.mediapipe_facemesh_one import flm_detector, distance
from scipy import signal
import os
from mmwave.dsp.utils import Window
import math
from FER.utils import parseConfigFile, arange_tx

from itertools import accumulate
from operator import add
from mmwave.dsp.cfar import ca

from scipy.signal import find_peaks, peak_widths
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

DebugMode = True

if not DebugMode:
    import matplotlib

    matplotlib.use('Agg')

# plotting color
import matplotlib._color_data as mcd

tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
extra_color = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04']
tab_color = tab_color + extra_color

# figpath = "C:/Users/Zber/Desktop/mmWave_figure"
figpath = "C:/Users/Zber/Desktop/SavedFigure"

# Data and ConfigFile
# 64 chirps
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_20s_50fps_2.cfg'
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Chirps_0_Raw_0.bin"


# 3s
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OpenEyes_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OpenMouth_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/RaiseCheek_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OnlySurprise_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OnlyBodyMotion_0_Raw_0.bin"
# adc_data_path = "D:\\Subjects\\S2\\Neutral_10_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Surprise0.5m_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Standing_1_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/empty_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing&surprise_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing&alwaysmove_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_sit_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/sit_1m_move_1_Raw_0.bin"
adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/stand_1m_move_1_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_ground_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/ground_1m_1_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Surprise1m_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/SurpriseAndBodyMotion_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing2ground_0_Raw_0.bin"

# adc_data_path = "D:\\Subjects\\S2\\Joy_33_Raw_0.bin"


if __name__ == '__main__':
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
    va_data = arange_tx(range_data, num_tx=numTxAntennas)

    # range_data = va_data[5, :, VIRT_ANT_AZI_INDEX, :]
    # range_data = np.transpose(range_data, (1, 0, 2))
    #
    # range_doppler = np.fft.fft(range_data, axis=0)
    # range_doppler = np.fft.fftshift(range_doppler, axes=0)
    #
    # padding = ((0, 0), (0, numAngleBins - range_doppler.shape[1]), (0, 0))
    # range_azimuth = np.pad(range_doppler, padding, mode='constant')
    # range_azimuth = np.fft.fft(range_azimuth, axis=1)
    # range_azimuth_plot = range_azimuth[:,:,26].sum(0)
    # plt.plot(np.log(np.abs(range_azimuth_plot)))
    # plt.show()

    # plt.imshow(np.log(np.abs(range_azimuth).sum(0).T))
    # plt.xlabel('Azimuth (Angle) Bins')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Azimuth')
    # plt.show()

    # chirp_data = np.mean(range_data[5, :, VIRT_ANT_AZI_INDEX, :], axis=-1)
    # chirp_data = va_data[5, :, VIRT_ANT_AZI_INDEX, 8]

    num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE_AZI, ANGLE_RES_AZI, VIRT_ANT_AZI)

    # plt.imshow(np.angle(steering_vec))
    # plt.show()
    # bin_start = 27
    # bin_end = 32

    # standing and surprise
    # bin_start = 25
    # bin_end = 30

    # standing and always move
    # bin_start = 10
    # bin_end = 11

    # standing
    bin_start = 25
    bin_end = 30

    # bin_start = 10
    # bin_end = 11

    # bin_end = 26

    # bin_start = 25
    # bin_end = 32

    # bin_start = 28
    # bin_end = 32

    # bin_start = 25
    # bin_end = 28

    # ground
    # bin_start = 30
    # bin_end = 34

    fig, axes = plt.subplots(1, 1, figsize=(160, 90))

    # chirp_data = va_data[5, :, VIRT_ANT_AZI_INDEX, bin_start]



    # 2Dfft
    # range_doppler = va_data[5, :, VIRT_ANT_AZI_INDEX, :]
    # range_doppler = np.transpose(range_doppler, (1, 0, 2))
    #
    # padding = ((0, 0), (0, numAngleBins - range_doppler.shape[1]), (0, 0))
    # range_azimuth = np.pad(range_doppler, padding, mode='constant')
    # range_azimuth = np.fft.fft(range_azimuth, axis=1)
    # range_azimuth = range_azimuth[:,:,25].sum(0)
    # plt.plot(np.log(np.abs(range_azimuth)))
    # plt.show()
    #
    # plt.imshow(np.log(np.abs(range_azimuth).sum(0).T))
    # plt.xlabel('Azimuth (Angle) Bins')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Azimuth')
    # plt.show()

    # time_angle_fft = np.zeros((64, 300))
    # for i in range(300):
    #     for r in range(bin_start, bin_end):
    #         range_data = va_data[i, :, VIRT_ANT_AZI_INDEX, :]
    #         range_data = np.transpose(range_data, (1, 0, 2))
    #         range_doppler = np.fft.fft(range_data, axis=0)
    #         range_doppler = np.fft.fftshift(range_doppler, axes=0)
    #         padding = ((0, 0), (0, numAngleBins - range_doppler.shape[1]), (0, 0))
    #         range_azimuth = np.pad(range_doppler, padding, mode='constant')
    #         range_azimuth = np.fft.fft(range_azimuth, axis=1)
    #         time_angle_fft[:, i] += np.log(np.abs(range_azimuth).sum(0)[:, r])
    #
    #         # range_azimuth = range_azimuth[:,:,25].sum(0)
    #         # plt.plot(np.log(np.abs(range_azimuth)))
    #
    # plt.imshow(time_angle_fft, cmap=plt.cm.jet)


    # angle_fft = np.fft.fft(va_data[5, :, VIRT_ANT_AZI_INDEX, bin_start], axis=0)

    time_angle = np.zeros((ANGLE_BINS_AZI, 300))

    for i in range(0, 300):
        for r in range(bin_start, bin_end):
            # chirp_data = np.mean(va_data[5, :, VIRT_ANT_AZI_INDEX, i:i + 1], axis=-1)
            chirp_data = va_data[i, :, VIRT_ANT_AZI_INDEX, r]
            # chirp_data = np.angle(chirp_data)
            # steering_vec = np.angle(steering_vec)
            # chirp_data = va_data[50, :, VIRT_ANT_AZI_INDEX, i]
            time_angle[:, i] += music.aoa_music_1D(steering_vec, chirp_data, 1)

            # time_angle[:, i] += music.aoa_root_music_1D(steering_vec, chirp_data, 1)

            # capon_angle, beamWeights = dsp.aoa_capon(chirp_data, steering_vec, magnitude=True)
            # time_angle[:, i] = time_angle[:, i] + capon_angle

        # music_specturm = music.aoa_root_music_1D(steering_vec, chirp_data, 1)
        # music_specturm = music.aoa_esprit(steering_vec.T, chirp_data, 2, 2)

    #
    # np.save("C:/Users/Zber/Desktop/Subjects/Test/empty_data", time_angle)
    # empty_angle = np.load("C:/Users/Zber/Desktop/Subjects/Test/empty_data.npy")

    # new_angle = time_angle-empty_angle

    # fig, axes = plt.subplots(1, 1, figsize=(50, 90))
    # axes.imshow(new_angle, cmap=plt.cm.jet)

    plt.imshow(time_angle, cmap=plt.cm.jet)

    # angle = np.abs(60 - np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle.npy"))
    # angle = np.abs(60 - np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle_standing&surprise.npy"))
    # image_angle = np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle_1m_sit.npy")
    # image_angle = np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle_sit_1m_move.npy")
    image_angle = np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle_stand_1m_move.npy")
    # image_angle = np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle_ground_1m_1.npy")

    # image_angle = np.load("C:/Users/Zber/Desktop/Subjects/Test/image_angle_standing_1.npy")
    image_angle = (image_angle - 35 + 90) / ANGLE_RES_AZI
    angle = image_angle
    x = np.linspace(0, 300, len(angle))
    plt.plot(x, angle, c="black", lw=5)

    plt.show()

    # ca from top to bottom
    detect_point = ca(time_angle[:, 10], guard_len=2, noise_len=4, mode='wrap', l_bound=1)

    # music plot
    # for i, color in zip(range(0, 300), tab_color):
    #     # chirp_data = np.mean(va_data[5, :, VIRT_ANT_AZI_INDEX, i:i + 1], axis=-1)
    #     chirp_data = va_data[i, :, VIRT_ANT_AZI_INDEX, 27]
    #     # chirp_data = va_data[50, :, VIRT_ANT_AZI_INDEX, i]
    #     music_specturm = music.aoa_music_1D(steering_vec, chirp_data, 1)
    #     # music_specturm = music.aoa_root_music_1D(steering_vec, chirp_data, 1)
    #     # music_specturm = music.aoa_esprit(steering_vec.T, chirp_data, 2, 2)
    #     axes.plot(music_specturm, c=color, label='Step {}'.format(i))
    # axes.legend(loc='upper right', fontsize=30)
    # plt.show()

    # for i, color in zip(range(0, 300, 50), tab_color):
    #     # chirp_data = np.mean(va_data[5, :, VIRT_ANT_AZI_INDEX, i:i + 1], axis=-1)
    #     chirp_data = va_data[i, :, VIRT_ANT_AZI_INDEX, 27]
    #     # chirp_data = va_data[50, :, VIRT_ANT_AZI_INDEX, i]
    #     music_specturm = music.aoa_music_1D(steering_vec, chirp_data, 1)
    #     # music_specturm = music.aoa_root_music_1D(steering_vec, chirp_data, 1)
    #     # music_specturm = music.aoa_esprit(steering_vec.T, chirp_data, 2, 2)
    #     axes.plot(music_specturm, c=color, label='Range {}'.format(i))
    # axes.legend(loc='upper right', fontsize=30)
    # plt.show()

    # chirp_data = np.mean(va_data[5, :, VIRT_ANT_AZI_INDEX, bin_start:bin_end], axis=-1)
    chirp_data = np.sum(va_data[:, :, VIRT_ANT_AZI_INDEX, 28], axis=0).T
    music_specturm = music.aoa_music_1D(steering_vec, chirp_data, 1)

    # scan from top to bottom

    # music_specturm = music.aoa_root_music_1D(steering_vec, chirp_data, 1)
    # music_specturm = music.aoa_esprit(steering_vec.T, chirp_data, 2, 2)
    plt.plot(music_specturm)
    plt.show()

    # music_specturm = np.zeros((61, 50))
    # for i in range(50):
    #     chirp_data = va_data[5, :, VIRT_ANT_AZI_INDEX, i]
    #     music_specturm[:, i] = music.aoa_music_1D(steering_vec, chirp_data, 1)
    # plt.imshow(music_specturm, cmap=plt.cm.jet)

    # chirp_data = np.mean(va_data[5, :, VIRT_ANT_AZI_INDEX, bin_start:bin_end], axis=-1)
    chirp_data = np.sum(va_data[:, :, VIRT_ANT_AZI_INDEX, 28], axis=0).T
    ang_est_vector = np.zeros((num_vec))
    num_max, doa_specturm = dsp.angle_estimation.aoa_est_bf_multi_peak_det(gamma=0.2, sidelobe_level=0.2,
                                                                           sig_in=chirp_data, steering_vec=steering_vec,
                                                                           steering_vec_size=num_vec,
                                                                           ang_est=ang_est_vector)
    # music_specturm = music.aoa_root_music_1D(steering_vec, chirp_data, 1)
    # music_specturm = music.aoa_esprit(steering_vec.T, chirp_data, 2, 2)
    plt.plot(doa_specturm[:, 1])
    plt.imshow(doa_specturm)
    plt.show()

    print("")
