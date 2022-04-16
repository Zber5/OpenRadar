# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize
from FER.utils import parseConfigFile, arange_tx

DebugMode = True

if not DebugMode:
    import matplotlib

    matplotlib.use('Agg')

figpath = "C:/Users/Zber/Desktop/SavedFigure"

# mmWave studio configure
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OpenEyes_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OnlySurprise_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/OnlyBodyMotion_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/SurpriseAndBodyMotion_0_Raw_0.bin"


# Subject
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S1/Surprise_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S3/Surprise_30_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S3/Surprise_30_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/constant_moving_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Joy_41_Raw_0.bin"
# adc_data_path = "C:\\Users\\Zber\\Desktop\\mmWave_plot\\Neutral_0_Raw_0.bin"
# adc_data_path = "C:\\Users\\Zber\\Desktop\\mmWave_plot\\Neutral_11_Raw_0.bin"
# adc_data_path = "C:\\Users\\Zber\\Desktop\\mmWave_plot\\Neutral_4_Raw_0.bin"
# adc_data_path = "D:\\Subjects\\S2\\Joy_26_Raw_0.bin"
# adc_data_path = "D:\\Subjects\\S2\\Neutral_10_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Surprise.5m_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Surprise1m_0_Raw_0.bin"

# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_standing_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_sit_0_Raw_0.bin"
adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/1m_ground_0_Raw_0.bin"

# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Notable_1_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Enable_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Enable_0_Raw_0.bin"


def plot_heatmap(adc_data_path, bin_start=4, bin_end=14, diff=False, remove_clutter=True, cumulative=False):
    num_bins = bin_end - bin_start
    npy_azi = np.zeros((numFrames, ANGLE_BINS, num_bins))
    npy_ele = np.zeros((numFrames, ANGLE_BINS, num_bins))

    adc_data = np.fromfile(adc_data_path, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)

    # Start DSP processing
    range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED))
    range_elevation = np.zeros((ANGLE_BINS, BINS_PROCESSED))
    num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT_AZI)
    num_vec_ele, steering_vec_ele = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT_ELE)

    cum_azi = np.zeros((ANGLE_BINS, num_bins))
    cum_ele = np.zeros((ANGLE_BINS, num_bins))
    pre_h = np.zeros((ANGLE_BINS, num_bins))
    pre_e = np.zeros((ANGLE_BINS, num_bins))

    for i in range(numFrames):

        """ 1 (Range Processing) """

        frame = adc_data[i]

        # --- range fft
        radar_cube = dsp.range_processing(frame)

        """ 2 (Capon Beamformer) """

        # --- static clutter removal
        # --- Do we need ?
        if remove_clutter:
            mean = radar_cube.mean(0)
            radar_cube = radar_cube - mean

        # --- capon beamforming
        beamWeights = np.zeros((VIRT_ANT_AZI, BINS_PROCESSED), dtype=np.complex_)
        radar_cube_azi = np.concatenate((radar_cube[0::numTxAntennas, ...], radar_cube[1::numTxAntennas, ...]), axis=1)

        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        for i in range(BINS_PROCESSED):
            range_azimuth[:, i], beamWeights[:, i] = dsp.aoa_capon(radar_cube_azi[:, :, i].T, steering_vec,
                                                                   magnitude=True)

        # --- capon beamforming elevation (3,5)
        beamWeights_ele = np.zeros((VIRT_ANT_ELE, BINS_PROCESSED), dtype=np.complex_)
        radar_cube_ele = np.concatenate(
            (radar_cube[0::numTxAntennas, 2:3, ...], radar_cube[2::numTxAntennas, 0:1, ...]), axis=1)
        for i in range(BINS_PROCESSED):
            range_elevation[:, i], beamWeights_ele[:, i] = dsp.aoa_capon(radar_cube_ele[:, :, i].T, steering_vec_ele,
                                                                         magnitude=True)

        """ 3 (Object Detection) """
        heatmap_log = np.log2(range_azimuth[:, bin_start:bin_end])
        heatmap_log_ele = np.log2(range_elevation[:, bin_start:bin_end])

        if cumulative:
            cum_azi += heatmap_log
            cum_ele += heatmap_log_ele

        heatmap_azi = heatmap_log - pre_h
        heatmap_ele = heatmap_log_ele - pre_e

        if diff:
            pre_h = heatmap_azi
            pre_e = heatmap_ele

        # normalize
        heatmap_azi = heatmap_azi / heatmap_azi.max()
        heatmap_ele = heatmap_ele / heatmap_ele.max()

        npy_azi[i] = heatmap_azi
        npy_ele[i] = heatmap_ele

        return npy_azi, npy_ele


if __name__ == '__main__':
    # DSP processing parameters for heatmap generation
    numAngleBins = 64
    VIRT_ELE_PAIRS = [[8, 2], [9, 3], [10, 4], [11, 5]]
    VIRT_AZI_PAIRS = [[i for i in range(0, 8)]]
    SKIP_SIZE = 4

    # azimuth
    ANGLE_RES_AZI = 2
    ANGLE_RANGE_AZI = 60
    ANGLE_BINS_AZI = (ANGLE_RANGE_AZI * 2) // ANGLE_RES_AZI + 1

    # elevation
    ANGLE_RES_ELE = 5
    ANGLE_RANGE_ELE = 30
    ANGLE_BINS_ELE = (ANGLE_RANGE_ELE * 2) // ANGLE_RES_ELE + 1

    BIN_RANG_S = 0
    BIN_RANG_E = 256
    BINS_PROCESSED = BIN_RANG_E - BIN_RANG_S
    VIRT_ANT_AZI_INDEX = [i for i in range(0, 8)]
    VIRT_ANT_ELE_INDEX = VIRT_ELE_PAIRS[2]

    # heatmap configuration
    mode = "capon"
    # mode = "bartlett"
    coherent = False

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

    if not coherent:
        VIRT_ANT_AZI = 8
        VIRT_ANT_ELE = 2
    else:
        VIRT_ANT_AZI = len(VIRT_ANT_AZI_INDEX)
        VIRT_ANT_ELE = len(VIRT_ANT_ELE_INDEX)

    static_clutter_removal = True
    diff = False
    z_score = False
    is_log = True

    ims = []
    max_size = 0

    # Start DSP processing
    num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE_AZI, ANGLE_RES_AZI, VIRT_ANT_AZI)
    num_vec_ele, steering_vec_ele = dsp.gen_steering_vec(ANGLE_RANGE_ELE, ANGLE_RES_ELE, VIRT_ANT_ELE)

    # fig, axes = plt.subplots(1, 4, figsize=(ANGLE_BINS // 5, BINS_PROCESSED // 5 * 4))
    fig, axes = plt.subplots(1, 2, figsize=(ANGLE_BINS_AZI // 5, BINS_PROCESSED // 5 * 2))
    frame_index = 0

    # stored np array
    cum_h = np.zeros((ANGLE_BINS_AZI, BINS_PROCESSED))
    cum_e = np.zeros((ANGLE_BINS_ELE, BINS_PROCESSED))

    # cum_h = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)
    # cum_e = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)

    pre_h = np.zeros((ANGLE_BINS_AZI, BINS_PROCESSED))
    pre_e = np.zeros((ANGLE_BINS_ELE, BINS_PROCESSED))

    maxl = 20

    queue = deque(maxlen=maxl)

    # for frame_index in range(1, 150):
    # for frame_index in range(numFrames):
    # surprise and body movement
    # for frame_index in range(100, 150):
    # surprise
    # for frame_index in range(88, 105):
    # neutral
    for frame_index in range(106, 134):

        # for frame_index in range(300):
        # eye
        # for frame_index in range(81, 98):
        # mouth
        # for frame_index in range(97, 127):
        # cheeck
        # for frame_index in range(66, 81):
        # frame_index += 1
        """ 1 (Range Processing) """

        frame = adc_data[frame_index]

        # --- range fft
        radar_cube = dsp.range_processing(frame, axis=-1)

        if static_clutter_removal:
            mean = radar_cube.mean(axis=2, keepdims=True)
            radar_cube = radar_cube - mean

        # virtual antenna arrangement
        radar_cube = arange_tx(radar_cube, num_tx=numTxAntennas, vx_axis=1, axis=0)

        """ 2 (Beamformer Processing) """
        # --- static clutter removal
        # --- Do we need ?
        # if static_clutter_removal:
        #     mean = radar_cube.mean(axis=2, keepdims=True)
        #     radar_cube = radar_cube - mean

        # --- capon beamforming

        if coherent:
            radar_cube_azi = []
            radar_cube_ele = []
            for azi_index in VIRT_AZI_PAIRS:
                radar_cube_azi.append(radar_cube[:, azi_index, :])
            for ele_index in VIRT_ELE_PAIRS:
                radar_cube_ele.append(radar_cube[:, ele_index, :])
        else:
            # radar_cube_azi = radar_cube[:, VIRT_ANT_AZI_INDEX, :]
            # radar_cube_ele = radar_cube[:, VIRT_ANT_ELE_INDEX, :]

            radar_cube_azi = [radar_cube[:, VIRT_ANT_AZI_INDEX, :]]
            radar_cube_ele = [radar_cube[:, VIRT_ANT_ELE_INDEX, :]]

        stack_azi = []
        stack_ele = []
        if mode == "capon":
            for cube_azi in radar_cube_azi:
                range_azimuth = np.zeros((ANGLE_BINS_AZI, BINS_PROCESSED))
                beamWeights = np.zeros((VIRT_ANT_AZI, BINS_PROCESSED), dtype=np.complex_)

                for i in range(BIN_RANG_S, BIN_RANG_E):
                    r_i = i - BIN_RANG_S
                    range_azimuth[:, r_i], beamWeights[:, r_i] = dsp.aoa_capon(cube_azi[:, :, i].T, steering_vec,
                                                                               magnitude=True)
                stack_azi.append(range_azimuth)

            for cube_ele in radar_cube_ele:
                range_elevation = np.zeros((ANGLE_BINS_ELE, BINS_PROCESSED))
                beamWeights_ele = np.zeros((VIRT_ANT_ELE, BINS_PROCESSED), dtype=np.complex_)
                for i in range(BIN_RANG_S, BIN_RANG_E):
                    r_i = i - BIN_RANG_S
                    range_elevation[:, r_i], beamWeights_ele[:, r_i] = dsp.aoa_capon(cube_ele[:, :, i].T,
                                                                                     steering_vec_ele, magnitude=True)
                stack_ele.append(range_elevation)

        elif mode == 'bartlett':
            for radar_cube_azi in radar_cube_azi:
                doa_spectrum_azi = dsp.aoa_bartlett(steering_vec, radar_cube_azi[..., BIN_RANG_S:BIN_RANG_E], axis=1)
                # stack_azi.append(np.mean(doa_spectrum_azi, axis=0))
                stack_azi.append(np.sum(doa_spectrum_azi, axis=0))
            for radar_cube_ele in radar_cube_ele:
                doa_spectrum_ele = dsp.aoa_bartlett(steering_vec_ele, radar_cube_ele[..., BIN_RANG_S:BIN_RANG_E],
                                                    axis=1)
                # stack_ele.append(np.mean(doa_spectrum_ele, axis=0))
                stack_ele.append(np.sum(doa_spectrum_ele, axis=0))

        else:
            print("not supported yet!")
            sys.exit(0)

        range_azimuth = np.sum(stack_azi, axis=0)
        range_elevation = np.sum(stack_ele, axis=0)

        """ 3 (Object Detection) """
        if is_log:
            range_azimuth = 20 * np.log10(range_azimuth)
            # range_azimuth = np.log2(range_azimuth)
            range_elevation = 20 * np.log10(range_elevation)
            # range_elevation = np.log2(range_elevation)

            # range_azimuth = np.log2(range_azimuth)
            # range_elevation = np.log2(range_elevation)

        cum_h += range_azimuth
        cum_e += range_elevation

        if diff:
            range_azimuth = range_azimuth - pre_h
            range_elevation = range_elevation - pre_e

            pre_h = range_azimuth
            pre_e = range_elevation

        if z_score:
            range_azimuth = (range_azimuth - np.mean(range_azimuth)) / np.std(range_azimuth)
            range_elevation = (range_elevation - np.mean(range_elevation)) / np.std(range_elevation)

        axes[0].set_xlabel('Range')
        axes[0].set_ylabel('Azimuth')

        axes[1].set_xlabel('Range')
        axes[1].set_ylabel('Elevation')

        # if len(queue) == maxl:
        #     list_heatmap = list(queue)
        #     sum_heatmap = np.sum(list_heatmap, axis=0)/maxl
        #     axes[0].imshow(sum_heatmap, interpolation='nearest', aspect='auto', cmap=parula_map)
        #     peaks, mask = cago_cfar(sum_heatmap, l_bound=14)
        #     axes[1].imshow(mask, interpolation='nearest', aspect='auto', cmap=parula_map)
        #
        # queue.append(range_azimuth)

        # axes[0].imshow(range_azimuth / range_azimuth.max(), interpolation='nearest', aspect='auto', cmap=parula_map)
        axes[0].imshow(range_azimuth[:, 0:55], interpolation='nearest', aspect='auto', cmap=plt.cm.jet)
        # axes[0].imshow(range_azimuth, interpolation='nearest', aspect='auto', cmap='coolwarm')

        # axes[0].imshow(np.angle(range_azimuth), interpolation='nearest', aspect='auto')

        # cago
        # peaks = cago_cfar(range_azimuth)

        # axes[1].set_xlabel('Range')
        # axes[1].set_ylabel('Elevation')
        # axes[1].imshow(range_elevation / range_elevation.max(), interpolation='nearest', aspect='auto', cmap=parula_map)
        # axes[1].imshow(range_elevation - range_elevation.min(), interpolation='nearest', aspect='auto', cmap=parula_map)
        # axes[1].imshow(range_elevation, interpolation='nearest', aspect='auto', cmap='coolwarm')

        axes[1].imshow(cum_h[:, 0:55], interpolation='nearest', aspect='auto', cmap=plt.cm.jet)
        # peaks, mask = cago_cfar(range_azimuth)
        # axes[1].imshow(mask, interpolation='nearest', aspect='auto', cmap=parula_map)

        # peaks, mask = cago_cfar(range_elevation)

        # axes[0].imshow(cum_h/np.max(cum_h), interpolation='nearest', aspect='auto', cmap='coolwarm')
        # axes[0].imshow(cum_e/np.max(cum_e), interpolation='nearest', aspect='auto', cmap='coolwarm')

        # peaks, mask = cago_cfar(cum_h/np.max(cum_h))
        # peaks, mask = cago_cfar_ele(cum_e/np.max(cum_e))

        # axes[1].imshow(mask, interpolation='nearest', aspect='auto', cmap='coolwarm')
        # axes[1].imshow(np.angle(range_elevation), interpolation='nearest', aspect='auto')
        #
        # axes[2].set_xlabel('Range')
        # axes[2].set_ylabel('Azimuth')
        # axes[2].imshow(cum_h / cum_h.max(), interpolation='nearest', aspect='auto', cmap='coolwarm')
        #
        # axes[3].set_xlabel('Range')
        # axes[3].set_ylabel('Elevation')
        # axes[3].imshow(cum_e / cum_e.max(), interpolation='nearest', aspect='auto', cmap='coolwarm')

        plt.title("Range-Angle Heatmap " + str(frame_index), loc='center')
        if frame_index == 133:
            print("")
        plt.pause(0.02)
        axes[0].clear()
        axes[1].clear()
        # axes[2].clear()
        # axes[3].clear()
