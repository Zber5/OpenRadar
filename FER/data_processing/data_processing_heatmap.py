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
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import os
from FER.utils import parseConfigFile, arange_tx, get_label

# mmWave studio configure
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'


# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s_10fps.cfg'


# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s_50fps.cfg'


def plot_heatmap_capon(adc_data_path, bin_start=4, bin_end=14, diff=False, is_log=False, remove_clutter=True,
                       cumulative=False):
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

    if cumulative:
        cum_azi = np.zeros((ANGLE_BINS, num_bins))
        cum_ele = np.zeros((ANGLE_BINS, num_bins))
    if diff:
        pre_h = np.zeros((ANGLE_BINS, num_bins))
        pre_e = np.zeros((ANGLE_BINS, num_bins))

    for frame_index in range(numFrames):

        frame = adc_data[frame_index]

        radar_cube = dsp.range_processing(frame)

        # virtual antenna arrangement
        radar_cube = arange_tx(radar_cube, num_tx=numTxAntennas, vx_axis=1, axis=0)

        # --- static clutter removal
        if remove_clutter:
            mean = radar_cube.mean(0)
            radar_cube = radar_cube - mean

        # --- capon beamforming
        radar_cube_azi = radar_cube[:, VIRT_ANT_AZI_INDEX, :]
        radar_cube_ele = radar_cube[:, VIRT_ANT_ELE_INDEX, :]

        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        for i in range(BINS_PROCESSED):
            range_azimuth[:, i], _ = dsp.aoa_capon(radar_cube_azi[:, :, i].T, steering_vec,
                                                   magnitude=True)
            range_elevation[:, i], _ = dsp.aoa_capon(radar_cube_ele[:, :, i].T, steering_vec_ele,
                                                     magnitude=True)

        """ 3 (Object Detection) """
        if is_log:
            heatmap_azi = 20 * np.log10(range_azimuth[:, bin_start:bin_end])
            heatmap_ele = 20 * np.log10(range_elevation[:, bin_start:bin_end])
        else:
            heatmap_azi = range_azimuth[:, bin_start:bin_end]
            heatmap_ele = range_elevation[:, bin_start:bin_end]

        if cumulative:
            cum_azi += heatmap_azi
            cum_ele += heatmap_ele

        if diff:
            heatmap_azi = heatmap_azi - pre_h
            heatmap_ele = heatmap_ele - pre_e
            pre_h = heatmap_azi
            pre_e = heatmap_ele

        # normalize
        # heatmap_azi = heatmap_azi / heatmap_azi.max()
        # heatmap_ele = heatmap_ele / heatmap_ele.max()

        npy_azi[frame_index] = heatmap_azi
        npy_ele[frame_index] = heatmap_ele

    return npy_azi, npy_ele


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

    if cumulative:
        cum_azi = np.zeros((ANGLE_BINS, num_bins))
        cum_ele = np.zeros((ANGLE_BINS, num_bins))
    if diff:
        pre_h = np.zeros((ANGLE_BINS, num_bins))
        pre_e = np.zeros((ANGLE_BINS, num_bins))

    for frame_index in range(numFrames):

        """ 1 (Range Processing) """

        frame = adc_data[frame_index]

        # --- range fft
        radar_cube = dsp.range_processing(frame)

        # range_bin_idx = 5

        # radar_cube to

        """ 2 (Capon Beamformer) """

        # --- static clutter removal
        # --- Do we need ?
        if remove_clutter:
            mean = radar_cube.mean(0)
            radar_cube = radar_cube - mean

        # --- capon beamforming
        beamWeights = np.zeros((VIRT_ANT_AZI, BINS_PROCESSED), dtype=np.complex_)
        radar_cube_azi = np.concatenate((radar_cube[0::numTxAntennas, ...], radar_cube[1::numTxAntennas, ...]), axis=1)
        # 4 virtual antenna
        # radar_cube_azi = radar_cube[0::numTxAntennas, ...]

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
        heatmap_azi = np.log2(range_azimuth[:, bin_start:bin_end])
        heatmap_ele = np.log2(range_elevation[:, bin_start:bin_end])

        if cumulative:
            cum_azi += heatmap_azi
            cum_ele += heatmap_ele

        if diff:
            heatmap_azi = heatmap_azi - pre_h
            heatmap_ele = heatmap_ele - pre_e
            pre_h = heatmap_azi
            pre_e = heatmap_ele

        # normalize
        heatmap_azi = heatmap_azi / heatmap_azi.max()
        heatmap_ele = heatmap_ele / heatmap_ele.max()

        npy_azi[frame_index] = heatmap_azi
        npy_ele[frame_index] = heatmap_ele

    return npy_azi, npy_ele


if __name__ == '__main__':

    # num Antennas
    numTxAntennas = 3
    numRxAntennas = 4

    # load configure parameters
    configParameters = parseConfigFile(configFileName)

    # mmWave radar settings
    numFrames = configParameters['numFrames']
    numADCSamples = configParameters['numAdcSamples']
    numLoopsPerFrame = configParameters['numLoops']
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
    numRangeBins = numADCSamples
    numDopplerBins = numLoopsPerFrame

    # DSP processing parameters
    VIRT_ELE_PAIRS = [[8, 2], [9, 3], [10, 4], [11, 5]]
    VIRT_AZI_PAIRS = [[i for i in range(0, 4)], [i for i in range(4, 8)], [i for i in range(8, 12)]]
    ANGLE_RES = 1
    ANGLE_RANGE = 45
    ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
    BINS_PROCESSED = 20

    # VIRT_ANT_AZI_INDEX = [i for i in range(8)]
    VIRT_ANT_AZI_INDEX = [i for i in range(0, 8)]
    VIRT_ANT_ELE_INDEX = VIRT_ELE_PAIRS[2]

    VIRT_ANT_AZI = len(VIRT_ANT_AZI_INDEX)
    VIRT_ANT_ELE = len(VIRT_ANT_ELE_INDEX)

    # heatmap configuration
    mode = "capon"
    # mode = "bartlett"
    non_coherent = False

    if non_coherent:
        VIRT_ANT_AZI = 8
        VIRT_ANT_ELE = 2

    static_clutter_removal = True
    is_diff = False
    is_log = False

    # range resolutiona and doppler resolution
    range_resolution, bandwidth = dsp.range_resolution(numADCSamples,
                                                       dig_out_sample_rate=configParameters['digOutSampleRate'],
                                                       freq_slope_const=configParameters['freqSlopeConst'])

    doppler_resolution = dsp.doppler_resolution(bandwidth, start_freq_const=configParameters['startFreq'],
                                                ramp_end_time=configParameters['rampEndTime'],
                                                idle_time_const=configParameters['idleTime'],
                                                num_loops_per_frame=configParameters['numLoops'],
                                                num_tx_antennas=numTxAntennas)
    print(
        'Range Resolution: {:.2f} cm, Bandwidth: {:.2f} Ghz, Doppler Resolution: {:.2f}'.format(range_resolution * 100,
                                                                                                bandwidth / 1000000000,
                                                                                                doppler_resolution))

    # Test data
    bin_start = 4
    bin_end = 14
    bin_path = "C:\\Users\\Zber\\Desktop\\Subjects\\Test\\LR_100fps_0_Raw_0.bin"
    output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects\\Test"
    npy_azi, npy_ele = plot_heatmap_capon(bin_path, bin_start, bin_end, diff=is_diff, is_log=is_log,
                                          remove_clutter=static_clutter_removal, cumulative=True)

    cum_azi = np.mean(npy_azi[100:200], 0)
    cum_ele = np.mean(npy_ele[100:200], 0)

    import seaborn as sns
    import matplotlib.pylab as plt

    ax = sns.heatmap(cum_azi)
    plt.show()

    uniform_data = npy_ele
    ax = sns.heatmap(cum_ele)
    plt.show()

    # ######  #
    root_path = "D:\\Subjects\\"
    data_path = '{}_{}_Raw_0.bin'
    output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap"

    # D Differences (current - pre), S (static clutter removal), L (log2 calculation),
    # N (Normalization), B (Bin index from # to #), I (Data Index from # to #)
    # A Angle Range, AR Angle Resolution, CO coherent

    # start index
    subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
    start_index = 0
    end_index = 1
    num_records = (end_index - start_index) * len(emotion_list) * len(subs)
    save_npy = True
    save_txt = True

    # data
    bin_start = 4
    bin_end = 14
    num_bins = bin_end - bin_start
    data_azi = np.zeros((num_records, numFrames, ANGLE_BINS, num_bins))
    data_ele = np.zeros((num_records, numFrames, ANGLE_BINS, num_bins))
    index = 0

    str_arr = []

    # saved file name
    # file_prefix = "heatmap_D0_S0_L0_N0_B4-14_I0-80_A-45->45_AR_1_CO_0"
    file_prefix = "Heatmap_D{}_S{}_L{}_B{}->{}_I{}->{}_A{}->{}_AR_{}_CO_{}".format(int(is_diff),
                                                                                   int(static_clutter_removal),
                                                                                   int(is_log), bin_start, bin_end,
                                                                                   start_index,
                                                                                   end_index, -ANGLE_RANGE, ANGLE_RANGE,
                                                                                   ANGLE_RES, non_coherent)

    for sub in subs:
        for l, e in enumerate(emotion_list):
            for i in range(start_index, end_index):
                bin_path = os.path.join(root_path, sub, data_path.format(e, i))
                relative_path = os.path.join(sub, data_path.format(e, i))
                label = get_label(data_path.format(e, i))

                npy_azi, npy_ele = plot_heatmap_capon(bin_path, bin_start, bin_end, diff=is_diff, is_log=is_log,
                                                      remove_clutter=static_clutter_removal)
                str_arr.append("{} {}".format(relative_path, label))

                data_azi[index] = npy_azi
                data_ele[index] = npy_ele
                index += 1
                print("{} Complete".format(bin_path))

    # save npy file
    if save_npy:
        save_path = os.path.join(output_data_path, file_prefix)
        save_path_azi = save_path + "_azi"
        save_path_ele = save_path + "_ele"
        np.save(save_path_azi, data_azi)
        np.save(save_path_ele, data_ele)
        print("Npy file saved")

    if save_txt:
        with open(os.path.join(root_path, "heatmap_annotation.txt"), 'a') as f:
            f.writelines('\n'.join(str_arr))
        print("Write {} Records to txt file".format(len(str_arr)))
