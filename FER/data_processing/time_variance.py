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

import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from em_plot.mediapipe_facemesh_one import flm_detector, distance
from scipy import signal
import os
from mmwave.dsp.utils import Window
import math

from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

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

# QOL settings
loadData = True

figpath = "C:/Users/Zber/Desktop/SavedFigure"

# Configure file
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'

antenna_order = [i for i in range(0, 4)] + [i for i in range(8, 12)] + [i for i in range(4, 8)]


def parseConfigFile(configFileName, numTxAnt=3):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('/r/n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        # numRxAnt = numRxAntennas
        # numTxAnt = numTxAntennas

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            adcStartTime = float(splitWords[4])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(splitWords[11])

            configParameters['startFreq'] = startFreq
            configParameters['idleTime'] = idleTime
            configParameters['adcStartTime'] = adcStartTime
            configParameters['rampEndTime'] = rampEndTime
            configParameters['numAdcSamples'] = numAdcSamples
            configParameters['digOutSampleRate'] = digOutSampleRate
            configParameters['freqSlopeConst'] = freqSlopeConst

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

            configParameters['numLoops'] = numLoops
            configParameters['numFrames'] = numFrames

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
            2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


def butter_bandpass_fs(sig, lowcut, highcut, fs, order=5, output='sos'):
    sos = signal.butter(order, [lowcut, highcut], btype='bandpass', output=output, fs=fs)
    filtered = signal.sosfilt(sos, sig)
    return filtered


def butter_lowpass_fs(sig, highcut, fs, order=5, output='sos'):
    sos = signal.butter(order, [highcut], btype='lowpass', output=output, fs=fs)
    filtered = signal.sosfilt(sos, sig)
    return filtered


def butter_highpass_fs(sig, lowcut, fs, order=5, output='sos'):
    sos = signal.butter(order, [lowcut], btype='highpass', output=output, fs=fs)
    filtered = signal.sosfilt(sos, sig)
    return filtered


def get_phase_change_npy(range_data, bin_index=0, is_diff=True, loop_index=5):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # fig1, axes1 = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data[:, loop_index, :]

    # num_va = numTxAntennas * numRxAntennas
    num_va_list = [2, 3, 4, 5, 8, 9, 10, 11]

    sig = np.angle(sig)
    sig = np.unwrap(sig, axis=0)

    sii = 20

    sig = sig[sii:, num_va_list]

    if is_diff:
        sig = np.abs(np.diff(sig, axis=0))
        sig = np.mean(sig, axis=1)
        sig = sig / np.max(sig)
    else:
        sig = np.mean(sig, axis=1)
    x_sig = np.linspace(0, 10, len(sig))

    # landmark generation
    all_FLms = flm_detector(video_path, None, output_as_video=False, output_flm_video=False)
    key_score = distance(all_FLms)
    lm_difference = np.sum(key_score, axis=1)

    si = 6
    lm = lm_difference[si:] / np.max(lm_difference[si:])

    x_lm = np.linspace(0, 10, len(lm))

    ax.plot(x_sig, sig, linewidth=2, c='b', label='Phase', zorder=5)
    ax.plot(x_lm, lm, linewidth=2, c='g', label='Landmark', zorder=5)

    # ax.set_ylim([-4, 4]) if is_diff else None
    plt.legend(bbox_to_anchor=(0.837, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    plt.show() if DebugMode else None
    if is_diff:
        fig.savefig("{}_feasibility_phase_landmark_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    else:
        fig.savefig("{}_feasibility_phase_landmark_{}_unwrap.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def arange_tx(signal, num_tx, vx_axis=2, axis=1):
    """Separate interleaved radar data from separate TX along a certain axis to account for TDM radars.

    Args:
        signal (ndarray): Received signal.
        num_tx (int): Number of transmit antennas.
        vx_axis (int): Axis in which to accumulate the separated data.
        axis (int): Axis in which the data is interleaved.

    Returns:
        ndarray: Separated received data in the

    """
    # Reorder the axes
    reordering = np.arange(len(signal.shape))
    reordering[0] = axis
    reordering[axis] = 0
    signal = signal.transpose(reordering)

    out = np.concatenate([signal[i::num_tx, ...] for i in range(num_tx)], axis=vx_axis)

    return out.transpose(reordering)


def correction_factor(win_type='Hanning'):
    WIN_EC = {"Hanning": 1.63, "Flattop": 2.26, "Blackman": 1.97, "Hamming": 1.59}

    numBits = 16
    cf = 20 * math.log10(2 ** (numBits - 1)) + 20 * math.log10(numADCSamples * WIN_EC[win_type]) - 20 * math.log10(
        math.sqrt(2))
    return cf


def phase_temporal_attentionv2(range_data, is_diff=True, loop_index=5):
    # fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # fig1, axes1 = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
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

    peak_threshold_weight = 0.8
    peak_threshold = max(sig) * peak_threshold_weight

    peaks = []

    while len(peaks) < 2:
        peaks, pp = find_peaks(sig, height=peak_threshold)

        if len(peaks) < 2:
            peak_threshold_weight -= 0.02
            peak_threshold = max(sig) * peak_threshold_weight

    heights = pp['peak_heights']
    index = sorted(sorted(range(len(heights)), key=lambda i: heights[i])[-2:])
    peaks = peaks[index]

    results_full = peak_widths(sig, peaks, rel_height=0.98)

    # onset -> peak
    mn0 = results_full[2][0]
    mx0 = results_full[3][0]

    # onset -> offset
    mn = results_full[2][0]
    mx = results_full[3][1]

    # ax.plot(sig)
    # ax.plot(peaks, sig[peaks], "x")
    # ax.hlines(*results_full[1:], color="C3")
    # ax.hlines(h, mn, mx, color='C2')
    # plt.show()

    onset_peak = mx0 - mn0
    onset_offset = mx - mn

    return onset_peak, onset_offset


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
    root_path = "C:\\Users\\Zber\\Desktop\\Subjects\\S2"
    output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects_Variance\\S2"
    file_prefix = "Variance"
    save_npy = True

    adc_path = "{}_2_{}_Raw_0.bin"
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust']

    start_index = 30
    end_index = 40

    num_records = (end_index - start_index) * len(emotion_list)

    data_npy = np.zeros((num_records, 2))

    record_index = 0
    for l, e in enumerate(emotion_list):
        for i in range(start_index, end_index):

            adc_data_path = os.path.join(root_path, adc_path.format(e, i))

            # (1) Reading in adc data
            if loadData:
                adc_data = np.fromfile(adc_data_path, dtype=np.int16)
                adc_data = adc_data.reshape(numFrames, -1)
                adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                               num_rx=numRxAntennas, num_samples=numADCSamples)
                print("{} >> Data Loaded!".format(adc_data_path))

            # (1) processing range data
            # window types : Bartlett, Blackman p, Hanning p and Hamming
            range_data = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)

            # reshape frame data
            range_data = arange_tx(range_data, num_tx=numTxAntennas)

            b_index = 8
            onset_peak, onset_offset = phase_temporal_attentionv2(range_data[..., b_index], is_diff=True)
            data_npy[record_index] = [onset_peak, onset_offset]


            # start_bin_index = 8
            # end_bin_index = 9

            # plot range profile
            # for i in range(start_bin_index, end_bin_index):
            #     b_index = i
            #     phase_temporal_attentionv2(range_data[..., b_index], b_index, is_diff=True)

    # save npy file
    if save_npy:
        save_path = os.path.join(output_data_path, file_prefix)
        np.save(save_path, data_npy)
        print("Npy file saved")