import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import os
from mmwave.dsp.utils import Window
import math
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from FER.utils import parseConfigFile, get_label, arange_tx, MapRecord

plt.close('all')

DebugMode = True

if not DebugMode:
    import matplotlib

    matplotlib.use('Agg')

# Configure file
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'


def peak_detection(range_data, half=True, is_diff=True, loop_index=5, plot=False):
    onset = peak = offset = -1
    e1 = e2 = e3 = 0
    start_seg = 10
    end_seg = 150

    sig = range_data[:, loop_index, :]

    # num_va = numTxAntennas * numRxAntennas
    va_list = [2, 3, 4, 5, 8, 9, 10, 11]

    sig = np.angle(sig)
    sig = np.unwrap(sig, axis=0)
    sig = sig[:, va_list]

    if is_diff:
        sig = np.diff(sig, axis=0)
        sig = np.abs(sig)
        sig = np.mean(sig, axis=1)
    else:
        sig = np.mean(sig, axis=1)

    sig = sig / max(sig)

    peak_threshold_weight = 0.95
    peak_threshold = max(sig) * peak_threshold_weight

    peaks = []

    num_peaks = 1 if half else 2

    sig_seg = sig[start_seg:end_seg]

    while len(peaks) < num_peaks:
        peaks, pp = find_peaks(sig_seg, height=peak_threshold)

        if len(peaks) < num_peaks:
            peak_threshold_weight -= 0.02
            peak_threshold = max(sig_seg) * peak_threshold_weight

    heights = pp['peak_heights']
    index = sorted(sorted(range(len(heights)), key=lambda i: heights[i])[-num_peaks:])
    peaks = peaks[index] + start_seg

    results_full = peak_widths(sig, peaks, rel_height=0.98)
    results_full = np.asarray(results_full)

    if results_full[0, 0] > 130 or results_full[0, 0] < 30:
        e1 = 1
        print("==== Width Problem ====")
    if results_full[1, 0] > 0.25:
        e2 = 1
        print("==== Height Problem ====")
    if results_full[2, 0] > 120 or results_full[3, 0] > 220:
    # if results_full[2, 0] > 150 or results_full[3, 0] > 250:
        e3 = 1
        print("==== Index Problem ====")

    # onset -> peak
    if half:
        onset = math.floor(results_full[2, 0])
        peak = math.ceil(results_full[3, 0])

    # onset -> offset
    if not half:
        offset = math.ceil(results_full[3, 1])
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(sig)
        ax.plot(peaks, sig[peaks], "x")
        ax.hlines(*results_full[1:, 0], color="C3")
        if not half:
            ax.hlines(*results_full[1:, 1], color='C2')
        plt.show()

    return onset, peak, offset, e1, e2, e3


def annotation_update(record_list, width=100, total_frame=300):
    for record in record_list:
        if record.num_frames < width:
            pad = (width - record.num_frames)//2
            if record.onset < pad:
                record.peak += pad*2

            elif (total_frame - record.peak) < pad:
                record.onset -= pad*2
            else:
                record.onset -= pad
                record.peak += pad
        else:
            pad = record.num_frames - width
            record.peak -= pad

        record.path = record.path.replace("Raw_0.bin","{}.npy").replace("\\", "/")

        if record.num_frames != 100:
            record.peak += 1
        assert record.num_frames == 100, 'the num of frames must equal to 100!'
    return record_list


def annotation_attention(record_list, width=30):
    for record in record_list:
        record.onset = math.floor(record.onset * 3 / 10)
        record.peak = record.onset + width - 1
        record.path = record.path.replace("_{}.npy","")
    return record_list


def annotation_append():
    str_arr = []
    str_format = "{} {} {} {} {} {} {} {}"
    npy_path = "{}_{}"
    emotion = 'Neutral'
    subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']

    for sub in subs:
        for i in range(0,30):
            path = (os.path.join(sub, npy_path.format(emotion, i,)) + '_{}.npy').replace("\\", "/")
            label = "0"
            onset = 51
            peak = 150
            offset = -1
            e1 = 0
            e2 = 0
            e3 = 0
            str_arr.append(str_format.format(path, label, onset, peak, offset, e1, e2 , e3))
    return str_arr


def data_split(record_list):
    labels = [r.label for r in record_list]
    train, test = train_test_split(record_list, test_size=0.2, random_state=25, stratify=labels)
    return train, test



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

    # numDopplerBins = numLoopsPerFrame
    numDopplerBins = numLoopsPerFrame

    numAngleBins = 64

    # data processing parameter
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

    # loading data configure
    # root_path = "D:\\Subjects"
    # root_path = "C:\\Users\\Zber\\Desktop\\Subjects_Heatmap"
    root_path = "C:\\Users\\Zber\\Desktop\\Subjects_Frames"
    annotaton_path = "D:\\Subjects\\annotations.txt"

    record_list = [MapRecord(x.strip().split(), root_path) for x in open(annotaton_path)]

    half_attention = True
    plot_debug = True

    for record in record_list:

        if record.index_err == 1:
            adc_data = np.fromfile(record.path, dtype=np.int16)
            adc_data = adc_data.reshape(numFrames, -1)
            adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                           num_rx=numRxAntennas, num_samples=numADCSamples)
            print("{} >> Data Loaded!".format(record.path))

            # processing range data
            range_data = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)

            # reshape frame data
            range_data = arange_tx(range_data, num_tx=numTxAntennas)

            b_index = 8
            out = peak_detection(range_data[..., b_index], half=half_attention,
                                                             plot=plot_debug)
