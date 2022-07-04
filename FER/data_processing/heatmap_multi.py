import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import os
from FER.utils import parseConfigFile, arange_tx, get_label
from queue import Queue
import threading
from mmwave.dsp.utils import Window

# mmWave studio configure
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'


def plot_heatmap_capon(adc_data_path, save_path, bin_start=4, bin_end=14, diff=False, is_log=False,
                       remove_clutter=True,
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
            # heatmap_azi = 20 * np.log10(range_azimuth[:, bin_start:bin_end])
            heatmap_azi = 20 * np.log10(range_azimuth[:, bin_start:bin_end] + 1)
            # heatmap_ele = 20 * np.log10(range_elevation[:, bin_start:bin_end])
            heatmap_ele = 20 * np.log10(range_elevation[:, bin_start:bin_end] + 1)
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

    save_path_azi = save_path + "_azi"
    save_path_ele = save_path + "_ele"
    np.save(save_path_azi, npy_azi)
    np.save(save_path_ele, npy_ele)
    print("{} npy file saved!".format(save_path))


def plot_heatmap_capon_v1(adc_data_path, save_path, bin_start=4, bin_end=14, is_log=False, remove_clutter=True):
    num_bins = bin_end - bin_start
    npy_azi = np.zeros((numFrames, ANGLE_BINS, num_bins))
    npy_ele = np.zeros((numFrames, ANGLE_BINS, num_bins))

    adc_data = np.fromfile(adc_data_path, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize_cli, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)

    # Data Preparation
    num_vec_azi, steering_vec_azi = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT_AZI)
    num_vec_ele, steering_vec_ele = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT_ELE)

    # Data Processing
    radar_cube = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)
    # radar_cube = dsp.range_processing(adc_data)
    # virtual antenna arrangement
    radar_cube = arange_tx(radar_cube, num_tx=numTxAntennas)

    # --- static clutter removal
    if remove_clutter:
        mean = radar_cube.mean(2, keepdims=True)
        radar_cube = radar_cube - mean

    # --- capon beamforming
    for i in range(0, 300):
        rb = 0
        for r in range(bin_start, bin_end):
            chirp_data_azi = radar_cube[i, :, VIRT_ANT_AZI_INDEX, r]
            # capon beamformer
            capon_angle_azi, _ = dsp.aoa_capon(chirp_data_azi, steering_vec_azi, magnitude=True)
            npy_azi[i, :, rb] = capon_angle_azi

            chirp_data_ele = radar_cube[i, :, VIRT_ANT_ELE_INDEX, r]
            # capon beamformer
            capon_angle_ele, _ = dsp.aoa_capon(chirp_data_ele, steering_vec_ele, magnitude=True)
            npy_ele[i, :, rb] = capon_angle_ele
            rb += 1

    if is_log:
        npy_azi = 20 * np.log10(npy_azi + 1)
        npy_ele = 20 * np.log10(npy_ele + 1)

    # if is_pose:
    #     npy_azi = npy_azi[:, ang_start:ang_end, :]
    #     npy_ele = npy_ele[:, 60:90, :]

    save_path_azi = save_path + "_azi"
    save_path_ele = save_path + "_ele"
    np.save(save_path_azi, npy_azi)
    np.save(save_path_ele, npy_ele)
    print("{} npy file saved!".format(save_path))


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


def thread_job(queue, bin_path, heatmap_out_path):
    while not queue.empty():
        q = queue.get()
        bpath = os.path.join(bin_path, q)
        hpath = os.path.join(heatmap_out_path, q.replace("_Raw_0.bin", ""))
        plot_heatmap_capon(bpath, hpath, bin_start=bin_start, bin_end=bin_end, diff=is_diff, is_log=is_log,
                           remove_clutter=static_clutter_removal)

        # plot_heatmap_capon_v1(bpath, hpath, bin_start=bin_start, bin_end=bin_end, is_log=is_log,
        #                       remove_clutter=static_clutter_removal)

        queue.task_done()


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
    BINS_PROCESSED = 80
    #     9 10 11 12
    # 1 2 3 4  5  6  7 8
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
    is_log = True

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

    root_path = "D:\\Subjects\\"
    data_path = '{}_{}_Raw_0.bin'
    output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap"
    # output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap"
    json_file_name = "config.json"

    # D Differences (current - pre), S (static clutter removal), L (log2 calculation),
    # N (Normalization), B (Bin index from # to #), I (Data Index from # to #)
    # A Angle Range, AR Angle Resolution, CO coherent

    # start index
    # subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    # subs = ['S8', 'S9']
    # subs = ['W2', 'W3']
    # subs = ['W2', 'W3']
    # subs = ['Standing_Jesse', 'Standing_Nancy', 'Ground_Jesse', 'Ground_Nancy', 'Distance_100cm']
    # subs = ['Distance_250cm']
    # subs = ['Distance_300cm']
    # subs = ['M1_0', 'M1_1', 'M2_0', 'M2_1', 'M3_0', 'M3_1']
    # subs = ['M1_2', 'M2_2', 'M3_2']
    # subs = ['Standing_Jesse',  'Ground_Jesse']
    # subs = ['S7']
    # subs = ['W1']
    # subs = ['S4']

    # subs = ['30cm_30d', '30cm_60d', '30cm_90d']
    # subs = ['70cm_30d', '70cm_60d', '70cm_90d']
    # subs = ['100cm_30d', '100cm_60d', '100cm_90d']
    # subs = ['150cm_30d', '150cm_60d', '150cm_90d']
    # subs = ['200cm_30d', '200cm_60d', '200cm_90d']
    # subs = ['250cm_30d', '250cm_60d', '250cm_90d']
    subs = ['300cm_30d', '300cm_60d', '300cm_90d']

    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']
    # emotion_list = ['Joy']
    # emotion_list = ['Neutral']
    start_index = 0
    end_index = 10
    save_txt = False
    save_config = False

    # data
    # bin_start = 4
    # bin_end = 14

    # Jesse
    # is_pose = True
    # bin_start = 26
    # bin_end = 31
    # ang_start = 5
    # ang_end = 35

    # Nancy
    # is_pose = True
    # bin_start = 23
    # bin_end = 28
    # ang_start = 10
    # ang_end = 40

    # 30_d cm
    # bin_start = 5
    # bin_end = 15

    # 70_d cm
    # bin_start = 13
    # bin_end = 23

    # 100_d cm
    # bin_start = 20
    # bin_end = 30

    # 150_d cm
    # bin_start = 31
    # bin_end = 41

    # 200_d cm
    # bin_start = 43
    # bin_end = 53

    # 250_d cm
    # bin_start = 55
    # bin_end = 65

    # 300_d cm
    bin_start = 66
    bin_end = 76

    # 100cm
    # bin_start = 18
    # bin_end = 28

    # 150 cm
    # bin_start = 38
    # bin_end = 48

    # 200 cm
    # bin_start = 45
    # bin_end = 55

    # 250 cm
    # bin_start = 55
    # bin_end = 65

    # 300 cm
    # bin_start = 67
    # bin_end = 77

    num_bins = bin_end - bin_start
    index = 0
    queue = Queue()

    str_arr = []

    for sub in subs:
        for l, e in enumerate(emotion_list):
            for i in range(start_index, end_index):
                bin_path = os.path.join(root_path, sub, data_path.format(e, i))
                relative_path = os.path.join(sub, data_path.format(e, i))
                queue.put(relative_path)
                label = get_label(data_path.format(e, i))
                str_arr.append("{} {}".format(relative_path, label))

    if save_txt:
        with open(os.path.join(output_data_path, "heatmap_annotation.txt"), 'a') as f:
            f.writelines('\n'.join(str_arr) + '\n')
        print("Write {} Records to txt file".format(len(str_arr)))

    if save_config:
        import json

        config = {
            "Different": str(is_diff),
            "Static Clutter Removal": str(static_clutter_removal),
            "Noncoherent": str(non_coherent),
            "Log10": str(is_log),
            "Bin Start": bin_start,
            "Bin End": bin_end,
            "Data Start Index": start_index,
            "Data End Index": end_index,
            "Angle Start": -ANGLE_RANGE,
            "Angle End": ANGLE_RANGE,
            "Angle Bins": ANGLE_BINS,
            "Angle Resolution": ANGLE_RES,
        }

        with open(os.path.join(output_data_path, json_file_name), 'w') as f:
            json.dump(config, f, indent=4)

    # thread_job(queue, root_path, output_data_path)

    NUM_THREADS = 12
    for i in range(NUM_THREADS):
        worker = threading.Thread(target=thread_job, args=(queue, root_path, output_data_path))
        worker.start()

    print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    print('This can take an hour or two depending on dataset size')
    queue.join()
    print('all done')
