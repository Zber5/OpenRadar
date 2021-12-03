import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import os
from mmwave.dsp.utils import Window
import math
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

from FER.utils import parseConfigFile, get_label, arange_tx, colors

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
    root_path = "D:\\Subjects\\"
    output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects_Variance\\S2"
    str_arr = []

    # str format: path, label, onset, peak, offset, widthError, heightError, indexError
    str_format = "{} {} {} {} {} {} {} {}"

    adc_path = "{}_{}_Raw_0.bin"
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']
    # emotion_list = ['Anger']
    subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    # subs = ['S1']

    start_index = 0
    end_index = 30
    half_attention = True
    plot_debug = True

    for sub in subs:
        for e in emotion_list:
            for i in range(start_index, end_index):
                adc_data_path = os.path.join(root_path, sub, adc_path.format(e, i))
                relative_path = os.path.join(sub, adc_path.format(e, i))

                # (1) Reading in adc data
                adc_data = np.fromfile(adc_data_path, dtype=np.int16)
                adc_data = adc_data.reshape(numFrames, -1)
                adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                               num_rx=numRxAntennas, num_samples=numADCSamples)
                print("{} >> Data Loaded!".format(adc_data_path))

                # processing range data
                range_data = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)

                # reshape frame data
                range_data = arange_tx(range_data, num_tx=numTxAntennas)

                b_index = 8
                out = peak_detection(range_data[..., b_index], half=half_attention,
                                                                 plot=plot_debug)
                label = get_label(e)
                str_arr.append(str_format.format(relative_path, label, *out))

    with open(os.path.join(root_path, "annotations_v2txt"), 'a') as f:
        f.writelines('\n'.join(str_arr))
    print("Write {} Records to txt file".format(len(str_arr)))