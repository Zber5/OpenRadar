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
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize
from data_processing.mediapipe_facemesh_one import flm_detector, distance
from scipy import signal
import os
from mmwave.dsp.utils import Window
import math

from itertools import accumulate
from operator import add

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

# QOL settings
loadData = True

# figpath = "C:/Users/Zber/Desktop/mmWave_figure"
figpath = "C:/Users/Zber/Desktop/SavedFigure"

# Configure file
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_10s.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx1_rx4_2.cfg'
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/radarProfile.cfg'
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_5s.cfg'

# pad cover face v
device_v = True

# bin index
bin_index = 7

# adc_base_path = 'C:/Users/Zber/Desktop/SavedData_MIMO/'
# adc_base_path = 'D:/SavedData_MIMO/'

# FER
# adc_data_name = "Anger_0_Raw_0.bin"
# adc_data_name = "Joy_41_Raw_0.bin"
# adc_data_name = "Sadness_0_Raw_0.bin"
# adc_data_name = "Surprise_0_Raw_0.bin"
# adc_data_name = "Fear_0_Raw_0.bin"
# adc_data_name = "Disgust_70_Raw_0.bin"

# adc_data_path = os.path.join(adc_base_path, adc_data_name)

# Material
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/Block_foilpad_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/Block_none_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/Block_mirror_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/Block_ipad_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/static_test_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/static_red_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/move_red_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/static_yellow_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/SavedData_Eyes/move_yellow_0_Raw_0.bin"

# adc_data_path = "C:/Users/Zber/Desktop/mmWave_plot/Joy_Test_0_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/mmWave_plot/Joy_Test_1_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/mmWave_plot/Joy_Test_2_Raw_0.bin"


# Subject
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S1/Joy_2_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S1/Anger_1_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S2/Joy_3_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/S2/Joy_2_31_Raw_0.bin"
# adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Test4_0_Raw_0.bin"
# adc_data_path = "D:/Subjects/S2/Joy_31_Raw_0.bin"
adc_data_path = "D:/Subjects/S4/Joy_20_Raw_0.bin"

# video_path = "C:/Users/Zber/Desktop/Subjects_Video/S2/Joy_31/Joy_31.avi"
video_path = "C:/Users/Zber/Desktop/Subjects_Video/S4/Joy_20/Joy_20.avi"

plotRangeDopp = True
plot2DscatterXY = False
plot2DscatterXZ = False
plot2DscatterYZ = False
plot3Dscatter = False
plotCustomPlt = False

plotMakeMovie = False
makeMovieTitle = ""
makeMovieDirectory = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/visualizer/movie/test_plotRDM.mp4"

# antenna_order = [8, 10, 7, 9, 6, 4, 5, 3] if device_v else [5, 6, 7, 8, 3, 4, 9, 10]
antenna_order = [i for i in range(0, 4)] + [i for i in range(8, 12)] + [i for i in range(4, 8)]

visTrigger = plot2DscatterXY + plot2DscatterXZ + plot3Dscatter + plotRangeDopp + plotCustomPlt
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

singFrameView = False

# constant frequency
LM_LOWCUT = 0.2
LM_HIGHCUT = 3

fig_prefix = os.path.basename(adc_data_path)[:-4]

# virtual Antenna Array
virtual_array = []
tx_map = {0: 0, 2: 1, 1: 2}
for tx in range(0, 3):
    tx = tx_map[tx]
    for rx in range(0, 4):
        virtual_array.append((tx, rx))

phase_only = True


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
    configParameters["numDopplerBins"] = int(numChirpsPerFrame / numTxAnt)
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


def movieMaker(fig, ims, title, save_dir):
    import matplotlib.animation as animation
    import matplotlib as mpl
    mpl.rcParams['animation.ffmpeg_path'] = r"C://Users//Zber//Documents//ffmpeg//bin//ffmpeg.exe"

    # Set up formatting for the Range Azimuth heatmap movies
    # Writer = animation.writers['ffmpeg']
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    plt.title(title)
    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    print('Complete')


def velocity_processing(x):
    return 0
    # calculate velocity for all range bin
    # print out velocity on specific range bin
    # look up the change


# def butter_bandpass(sig, lowcut, highcut, fs, order=5, output='ob'):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     sos = signal.butter(order, [low, high], btype='bandpass', output=output)
#     filtered = signal.sosfreqz(sos, sig)
#     return filtered


# def butter_lowpass(sig, lowcut, highcut, fs, order=5, output='ob'):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     sos = signal.butter(order, [high], btype='lowpass', output='ba')
#     filtered = signal.sosfreqz(sos, sig)
#     # pp = plt.semilogx((fs * 0.5 / np.pi) * w, abs(h), label=label)
#     return filtered


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


def plot_amplitude_change(range_data, bin_index):
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = np.sum(np.abs(sig), axis=(1, 2, 3))
    axes.plot(sig, linewidth=1.5, c='b')
    fig.tight_layout()
    plt.show() if DebugMode else None
    fig.savefig("{}_amp_change_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def plot_amplitude_change_multi(range_data, bin_index, is_diff=True):
    if device_v:
        fig, axes = plt.subplots(4, 2, figsize=(50, 90))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(90, 50))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    for ax, o, color in zip(fig.axes, antenna_order, tab_color):
        va_order = o - 1
        t, r = virtual_array[va_order]
        # t, r = tx_map[t], r - 1
        va_sig = sig[:, t, r]
        va_sum = np.abs(va_sig)
        if is_diff:
            va_sum = np.diff(va_sum)
        ax.plot(va_sum, linewidth=15, c=color)
        ax.set_title('Virtual Antenna {}'.format(o), fontdict={'fontsize': 80, 'fontweight': 25})
        # ax.set_ylim([-4, 4])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.97, bottom=0.03, left=0.03)

    plt.show() if DebugMode else None
    fig.savefig("{}_amp_change_multi_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def plot_amplitude_change_multi_in_one(range_data, bin_index, is_diff=True):
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    for o, color in zip(antenna_order, tab_color):
        va_order = o - 1
        t, r = virtual_array[va_order]
        # t, r = tx_map[t], r - 1
        va_sig = sig[:, t, r]
        va_sum = np.abs(va_sig)
        if is_diff:
            va_sum = np.diff(va_sum)
        axes.plot(va_sum, linewidth=2, c=color)
        axes.set_title('Virtual Antenna {}'.format(o), fontdict={'fontsize': 80, 'fontweight': 25})
        # ax.set_ylim([-4, 4])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.97, bottom=0.03, left=0.03)

    plt.show() if DebugMode else None
    if is_diff:
        fig.savefig("{}_amp_change_multi_in_one_diff{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    else:
        fig.savefig("{}_amp_change_multi_in_one_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def plot_freq_change_multi(range_data, is_diff=True):
    if device_v:
        fig, axes = plt.subplots(4, 2, figsize=(50, 90))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(90, 50))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    # sig = sig[:, :, 5, :]

    for ax, o, color in zip(fig.axes, antenna_order, tab_color):
        va_order = o - 1
        t, r = virtual_array[va_order]
        # t, r = tx_map[t], r - 1
        va_sig = sig[:, t, :, r]
        va_freq = va_sig.reshape((-1))
        ax.plot(va_freq, linewidth=15, c=color)
        ax.set_title('Virtual Antenna {}'.format(o), fontdict={'fontsize': 80, 'fontweight': 25})
        # ax.set_ylim([-4, 4])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.97, bottom=0.03, left=0.03)

    plt.show() if DebugMode else None
    fig.savefig("{}_freq_multi.pdf".format(os.path.join(figpath, fig_prefix)))


def plot_virtual_antenna(range_data, bin_index, is_diff=True):
    if device_v:
        fig, axes = plt.subplots(4, 2, figsize=(50, 90))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(90, 50))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    for ax, o, color in zip(fig.axes, antenna_order, tab_color):
        va_order = o - 1
        t, r = virtual_array[va_order]
        # t, r = tx_map[t], r - 1
        va_sig = sig[:, t, r]
        va_phase = np.angle(va_sig)
        va_unwrap_phase = np.unwrap(va_phase)

        if is_diff:
            va_diff_phase = np.diff(va_unwrap_phase)
        else:
            va_diff_phase = va_unwrap_phase

        ax.plot(va_diff_phase, linewidth=15, c=color)
        ax.set_title('Virtual Antenna {}'.format(o), fontdict={'fontsize': 80, 'fontweight': 25})
        ax.set_ylim([-4, 4])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.97, bottom=0.03, left=0.03)

    plt.show() if DebugMode else None

    if is_diff:
        fig.savefig("{}_multiphase_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    else:
        fig.savefig("{}_multiphase_unwrap_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def plot_phase_change_in_one(range_data, bin_index=0, is_diff=True, loop_index=5):
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))

    # fig1, axes1 = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data[:, loop_index, :]

    # num_va = numTxAntennas * numRxAntennas
    num_va = 5
    ax = axes
    for o, color in zip(range(0, 12), tab_color):
        va_index = antenna_order[o]
        # t, r = tx_map[t], r - 1
        va_sig = sig[:, o]
        va_phase = np.angle(va_sig)
        va_unwrap_phase = np.unwrap(va_phase)

        if is_diff:
            va_diff_phase = np.diff(va_unwrap_phase)
        else:
            va_diff_phase = va_unwrap_phase

        max_i = np.argmax(np.abs(va_diff_phase))
        # plot_max_x = np.arange(max_i - 1, max_i + 2)
        # plot_max_y = va_diff_phase[max_i - 1:max_i + 2]

        plot_max_x = np.arange(max_i, max_i + 1)
        plot_max_y = va_diff_phase[max_i:max_i + 1]

        # va_diff_phase_filtered = butter_lowpass_fs(va_diff_phase, 40, 100)
        va_diff_phase_filtered = butter_bandpass_fs(va_diff_phase, 10, 40, 100)

        ax.plot(va_diff_phase, linewidth=2, c=color, label='Virtual Antenna {}'.format(va_index), zorder=5)
        # ax.plot(va_diff_phase_filtered, linewidth=1.5, linestyle='dashed', c=color,
        #         label='Virtual Antenna Filtered{}'.format(va_index), zorder=6)

        # ax.scatter(plot_max_x, plot_max_y, c=color, s=5, zorder=10)
        ax.set_ylim([-4, 4]) if is_diff else None
    plt.legend(bbox_to_anchor=(0.837, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    plt.show() if DebugMode else None
    if is_diff:
        fig.savefig("{}_multiphase_in_one_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    else:
        fig.savefig("{}_multiphase_in_one_{}_unwrap.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def get_phase_change_npy(range_data, bin_index=0, is_diff=True, loop_index=5):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # fig1, axes1 = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data[:, loop_index, :, :]

    # num_va = numTxAntennas * numRxAntennas
    num_va_list = [2, 3, 4, 5, 8, 9, 10, 11]

    sig = np.angle(sig)
    sig = np.unwrap(sig, axis=0)

    sii = 0

    sig = sig[sii:, num_va_list]

    if is_diff:
        sig = np.abs(np.diff(sig, axis=0))
        sig = np.mean(sig, axis=(1, 2))
        sig = sig / np.max(sig)
    else:
        sig = np.mean(sig, axis=1)
    x_sig = np.linspace(0, 10, len(sig))

    # landmark generation
    all_FLms = flm_detector(video_path, None, output_as_video=False, output_flm_video=False)
    key_score = distance(all_FLms, normalise=False)
    lm_difference = np.sum(key_score, axis=1)

    si = 0
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


def plot_virtual_antenna_point(range_data, bin_index):
    if device_v:
        fig, axes = plt.subplots(4, 2, figsize=(50, 90))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(90, 50))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    for ax, o, color in zip(fig.axes, antenna_order, tab_color):
        h = color[1:]
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

        va_order = o - 1
        t, r = virtual_array[va_order]
        # t, r = tx_map[t], r - 1
        va_sig = sig[:, t, r]
        va_real = va_sig.real
        va_imag = va_sig.imag

        # va_real = va_real[20:120]
        # va_imag = va_imag[20:120]

        alpha_scale = np.linspace(0, 1, len(va_imag))
        colors = [[rgb[0], rgb[1], rgb[2], a] for a in alpha_scale]

        # va_phase = np.angle(va_sig)
        # va_unwrap_phase = np.unwrap(va_phase)

        # va_diff_phase = np.diff(va_unwrap_phase)
        # va_diff_phase = va_unwrap_phase

        ax.scatter(va_imag, va_real, s=60, c=colors, label='Virtual Antenna {}'.format(o))
        ax.legend(loc='upper right', fontsize=30)
        # ax.set_title('Virtual Antenna {}'.format(o), fontdict= {'fontsize': 20, 'fontweight' : 15} )
        # ax.set_ylim([-4, 4])
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    plt.show() if DebugMode else None
    fig.savefig("{}_scatter_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def plot_phase_map(range_data, num_bins=None, v_antenna=None, loop_index=None, is_phase=False, is_diff=True):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]

    if num_bins is None:
        num_bins = numADCSamples

    # focus range
    # sig = range_data[..., :num_bins]
    sig = range_data

    # sig = sig.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas, num_bins))

    # phase processing
    if is_phase:
        sig_phase = np.angle(sig)
        sig_phase = np.unwrap(sig_phase)
        name = "phase"
    else:
        sig_phase = np.abs(sig)
        name = "amplitude"
        sig_phase = 20 * np.log10(sig_phase) - correction_factor() + 10
    # sig_phase = np.unwrap(sig_phase)
    # sig_phase = sig

    sig_phase = sig_phase[..., 5:num_bins]

    if loop_index:
        sig_phase = sig_phase[:, loop_index, :, :]
    else:
        sig_phase = np.mean(sig_phase, axis=1)

    if v_antenna:
        # t, r = virtual_array[loop_index]
        sig_phase = sig_phase[:, v_antenna, :]
    else:
        sig_phase = np.mean(sig_phase, axis=1)

    # sig_phase = np.angle(sig_phase)
    # sig_phase = np.unwrap(sig_phase)

    if is_diff:
        sig_phase = np.diff(sig_phase, axis=0)

    # matlib 'viridis'
    if is_phase and is_diff:
        img = ax.imshow(np.abs(sig_phase.T), cmap='bwr', vmin=-5, vmax=5, interpolation='none', aspect='auto')
    else:
        img = ax.imshow(sig_phase.T, cmap='bwr', interpolation='none', aspect='auto')

    cbar = fig.colorbar(img, ax=ax)
    cbar.minorticks_on()

    fig.tight_layout()
    plt.show() if DebugMode else None
    if is_diff:
        fig.savefig("{}_{}_map_{}_diff.pdf".format(os.path.join(figpath, fig_prefix), name, bin_index))
    else:
        fig.savefig("{}_{}_map_{}_unwrap.pdf".format(os.path.join(figpath, fig_prefix), name, bin_index))


def plot_range_profile(range_data, bin_index, manual_bin=True):
    numDisplaySamples = 20
    select_chirp = 5
    select_loop = 3

    # figure preparation
    fig, axes = plt.subplots(3, 2, figsize=(120, 60))

    # (1) processing range data
    # window types : Bartlett, Blackman p, Hanning p and Hamming
    # range_data = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)
    selected_range_data = range_data[:, :, select_loop, :numDisplaySamples]

    # plot range profile
    range_profile = selected_range_data.reshape((-1, numDisplaySamples))
    axes[0, 0].imshow(np.abs(range_profile).T, interpolation='nearest', aspect='auto')

    # (2) variance magnitude determine argmax
    var = np.var(np.abs(selected_range_data), axis=(0, 1))
    if not manual_bin:
        bin_index = np.argmax(var, axis=0)

    # plot variance
    axes[0, 1].bar(np.arange(0, numDisplaySamples), var)

    # (3) Select the range bin corresponding to the user
    select_bin_data = selected_range_data[..., bin_index]

    # chirp selection
    # select_bin_data = select_bin_data[:, 0::3]
    select_bin_data = select_bin_data[:, select_chirp]

    # (4) subtract mean value of related raw signal
    bin_data = select_bin_data

    # (5) extract phase data
    bin_data = bin_data.reshape(-1)
    phase_data = np.angle(bin_data)

    # plot phase data
    axes[1, 0].plot(phase_data)

    # (6) unwrapping phase
    unwrap_phase_data = np.unwrap(phase_data)

    # plot unwrapped phase data
    axes[1, 1].plot(unwrap_phase_data)

    # (7) difference unwrapped phase data
    diff_phase_data = np.diff(unwrap_phase_data)

    # plot difference of phase data
    axes[2, 0].set_ylim([-3, 3])
    axes[2, 0].plot(diff_phase_data)

    # (8) band pass filter signal
    # x1 = butter_bandpass_fs(sig=x, lowcut=1.5, highcut=3, fs=50, output='sos')
    # x1 = butter_lowpass_fs(sig=x, highcut=3, fs=50, output='sos')
    filter_data = butter_highpass_fs(sig=unwrap_phase_data, lowcut=20, fs=50, output='sos')

    # plot filtered data
    axes[2, 1].set_ylim([-4, 4])
    axes[2, 1].plot(np.abs(filter_data[1]))

    plt.show() if DebugMode else None
    plt.savefig("{}_phase_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


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


def plot_range_heatmap(range_data, is_plot=True):
    # rd_map = np.zeros((numFrames, numTxAntennas * numRxAntennas, numADCSamples, numDopplerBins))

    for i, radar_cube in enumerate(range_data[10:15:5]):
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas,
                                                       clutter_removal_enabled=True,
                                                       window_type_2d=Window.HAMMING,
                                                       interleaved=True, accumulate=False)

        det_matrix = det_matrix.transpose((1, 0, 2))

        if is_plot:
            fig, axes = plt.subplots(3, 4, figsize=(120, 90))
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=2)
            det_matrix_vis = det_matrix_vis[:, :32, :]
            mean = np.mean(det_matrix_vis, axis=0)
            std = np.std(det_matrix_vis, axis=0)

            for ax, virtual_index in zip(fig.axes, range(numTxAntennas * numRxAntennas)):
                plt_matrix = det_matrix_vis[virtual_index]
                img = ax.imshow(plt_matrix / plt_matrix.max(), aspect='auto')
                # img = ax.imshow((plt_matrix - mean) / std, cmap='Blues', aspect='auto')
                # cbar = fig.colorbar(img, ax=ax)
                # cbar.minorticks_on()
                fig.tight_layout()
                plt.show() if DebugMode else None
                plt.savefig("{}_range_velocity_{}.pdf".format(os.path.join(figpath, fig_prefix), i))

                # det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
                # rd_map[i] = det_matrix

    return aoa_input


def plot_micro_doppler(range_data, bin_index=5, va=5):
    # get bin
    range_data = range_data[:, :, bin_index]
    range_data = range_data[:, :, va]
    doppler = np.fft.fft(range_data, axis=1)
    # doppler = np.log2(np.abs(doppler))
    doppler = np.fft.fftshift(doppler, axes=1)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(np.abs(doppler.T), cmap='bwr', aspect='auto')


def plot_range_velocity(range_data, is_plot=True):
    # rd_map = np.zeros((numFrames, numTxAntennas * numRxAntennas, numADCSamples, numDopplerBins))

    for i, radar_cube in enumerate(range_data[10:15:5]):
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas,
                                                       clutter_removal_enabled=True,
                                                       window_type_2d=Window.HAMMING,
                                                       interleaved=True, accumulate=False)

        det_matrix = det_matrix.transpose((1, 0, 2))

        if is_plot:
            fig, axes = plt.subplots(3, 4, figsize=(120, 90))
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=2)
            # det_matrix_vis = det_matrix_vis[:, :32, :]
            mean = np.mean(det_matrix_vis, axis=0)
            std = np.std(det_matrix_vis, axis=0)

            for ax, virtual_index in zip(fig.axes, range(numTxAntennas * numRxAntennas)):
                plt_matrix = det_matrix_vis[virtual_index]
                # img = ax.imshow(plt_matrix / plt_matrix.max(), cmap='viridis', aspect='auto')
                img = ax.imshow((plt_matrix - mean) / std, cmap='bwr', aspect='auto')
                # cbar = fig.colorbar(img, ax=ax)
                # cbar.minorticks_on()
                fig.tight_layout()
                plt.show() if DebugMode else None
                plt.savefig("{}_range_velocity_{}.pdf".format(os.path.join(figpath, fig_prefix), i))

                # det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
                # rd_map[i] = det_matrix

    return aoa_input


def plot_range_azimuth_heatmap(range_data):
    return 0


def correction_factor(win_type='Hanning'):
    WIN_EC = {"Hanning": 1.63, "Flattop": 2.26, "Blackman": 1.97, "Hamming": 1.59}

    numBits = 16
    cf = 20 * math.log10(2 ** (numBits - 1)) + 20 * math.log10(numADCSamples * WIN_EC[win_type]) - 20 * math.log10(
        math.sqrt(2))
    return cf


def phase_temporal_attention(range_data, bin_index=0, is_diff=True, loop_index=5):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # fig1, axes1 = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data[:, loop_index, :]

    # num_va = numTxAntennas * numRxAntennas
    num_va = 12

    sig = np.angle(sig)
    sig = np.unwrap(sig, axis=0)

    if is_diff:
        sig = np.diff(sig, axis=0)
        sig = np.mean(sig, axis=1)
    else:
        sig = np.mean(sig, axis=1)

    sig_grad = np.gradient(sig, edge_order=1)
    sig_grad2 = np.gradient(sig, edge_order=2)
    sig_grad2_abs = np.abs(sig_grad2)

    sig_grad2_abs_acc = list(accumulate(sig_grad2_abs, func=add))
    sig_grad2_acc = list(accumulate(sig_grad2, func=add))

    peak_threshold_weight = 0.5511886431

    peak_threshold = max(sig_grad2_abs) * peak_threshold_weight
    peaks, heights = find_peaks(sig_grad2_abs, height=peak_threshold)

    ax.plot(sig, linewidth=2, c='b', label='Mean', zorder=5)
    ax.plot(sig_grad, linewidth=2, c='r', label='Grad1', zorder=6)
    ax.plot(sig_grad2, linewidth=2, c='g', label='Grad2', zorder=5)
    ax.plot(sig_grad2_abs, linewidth=2, c='c', label='Grad2_ABS', zorder=5)
    # ax.plot(sig_grad2_abs_acc, linewidth=2, c='y', label='Grad2_ABS_ACC', zorder=5)
    # ax.plot(sig_grad2_acc, linewidth=2, c='m', label='Grad2_ACC', zorder=5)
    ax.scatter(peaks, sig_grad2_abs[peaks], c='black')

    # ax.set_ylim([-4, 4]) if is_diff else None
    plt.legend(bbox_to_anchor=(0.837, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    plt.show() if DebugMode else None
    if is_diff:
        fig.savefig("{}_multiphase_in_one_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    else:
        fig.savefig("{}_multiphase_in_one_{}_unwrap.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def phase_temporal_attentionv2(range_data, bin_index=0, is_diff=True, loop_index=5):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

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

    peaks, _ = find_peaks(sig, height=peak_threshold)

    results_full = peak_widths(sig, peaks, rel_height=0.98)

    h = [results_full[1][1]]
    mn = [results_full[2][0]]
    mx = [results_full[3][1]]

    mn0 = [results_full[2][0]]
    mx0 = [results_full[3][0]]

    ax.plot(sig)
    ax.plot(peaks, sig[peaks], "x")
    ax.hlines(*results_full[1:], color="C3")
    ax.hlines(h, mn, mx, color='C2')
    plt.show()

    print("")

    # sig_grad = np.gradient(sig, edge_order=1)
    # sig_grad2 = np.gradient(sig, edge_order=2)
    # sig_grad2_abs = np.abs(sig_grad2)
    #
    # sig_grad2_abs_acc = list(accumulate(sig_grad2_abs, func=add))
    # sig_grad2_acc = list(accumulate(sig_grad2, func=add))
    #
    # peak_threshold_weight = 0.5511886431
    #
    # peak_threshold = max(sig_grad2_abs) * peak_threshold_weight
    # peaks, heights = find_peaks(sig_grad2_abs, height=peak_threshold)
    #
    # ax.plot(sig, linewidth=2, c='b', label='Mean', zorder=5)
    # ax.plot(sig_grad, linewidth=2, c='r', label='Grad1', zorder=6)
    # ax.plot(sig_grad2, linewidth=2, c='g', label='Grad2', zorder=5)
    # ax.plot(sig_grad2_abs, linewidth=2, c='c', label='Grad2_ABS', zorder=5)
    # # ax.plot(sig_grad2_abs_acc, linewidth=2, c='y', label='Grad2_ABS_ACC', zorder=5)
    # # ax.plot(sig_grad2_acc, linewidth=2, c='m', label='Grad2_ACC', zorder=5)
    # ax.scatter(peaks, sig_grad2_abs[peaks], c='black')
    #
    # # ax.set_ylim([-4, 4]) if is_diff else None
    # plt.legend(bbox_to_anchor=(0.837, 1), loc='upper left', borderaxespad=0.)
    # fig.tight_layout()
    # plt.show() if DebugMode else None
    # if is_diff:
    #     fig.savefig("{}_multiphase_in_one_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    # else:
    #     fig.savefig("{}_multiphase_in_one_{}_unwrap.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def micro_doppler():
    return 0


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

    print('Range Resolution: {:.2f}cm, Bandwidth: {:.2f}Ghz, Doppler Resolution: {:.2f}m/s'.format(
        range_resolution * 100, bandwidth / 1000000000, doppler_resolution))

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile(adc_data_path, dtype=np.int16)
        adc_data = adc_data.reshape(numFrames, -1)
        # adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
        #                                num_rx=numRxAntennas, num_samples=numADCSamples)

        adc_data = np.apply_along_axis(DCA1000.organize_cli, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)

        dataCube = np.copy(adc_data)
        print("Data Loaded!")

    # (1) processing range data
    # window types : Bartlett, Blackman p, Hanning p and Hamming
    # range_data = dsp.range_processing(adc_data)
    range_data = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)
    range_data = arange_tx(range_data, num_tx=numTxAntennas)

    det_matrix, aoa_input = dsp.doppler_processing_frame(range_data, num_tx_antennas=numTxAntennas,
                                                         clutter_removal_enabled=True,
                                                         window_type_2d=Window.HAMMING,
                                                         accumulate=True)

    det_matrix_vis = np.fft.fftshift(det_matrix, axes=2)

    print("")
    # for i in range(numFrames):
    #     radar_frame = range_data[i]
    #
    #     det_matrix, aoa_input = dsp.doppler_processing(radar_frame, num_tx_antennas=numTxAntennas,
    #                                                    clutter_removal_enabled=True, interleaved=False,
    #                                                    window_type_2d=Window.HAMMING,
    #                                                    accumulate=True)

    # numRangeBins, numVirtualAntennas, num_doppler_bins

    # range_data = range_data.reshape((-1,32,3,4,256))

    # plot_phase_map(range_data, num_bins=30, v_antenna=5, loop_index=5, is_phase=True, is_diff=True)

    # plot_micro_doppler(range_data)

    # plot_range_velocity(range_data)

    # range_data = arange_tx(range_data, num_tx=numTxAntennas)
    # plot phase map
    # plot_phase_map(range_data, num_bins=30, v_antenna=2, loop_index=5, is_phase=False, is_diff=False)

    # plot_range_heatmap(range_data)

    # range profile test
    # sig = range_data.reshape((numFrames * numTxAntennas * numLoopsPerFrame, numRxAntennas, numADCSamples))
    #
    # sig = sig[:, 0, 7:13]
    #
    # plt.imshow(np.abs(sig.T), cmap='bwr', interpolation='nearest', aspect='auto')
    # plt.show()

    # range_data = range_data.reshape((numFrames , numTxAntennas, numLoopsPerFrame, numRxAntennas, numADCSamples))
    # fft2d_out = np.fft.fft(range_data, axis=2)
    #
    # fft2d_log_abs = np.log2(np.abs(fft2d_out))
    #
    # # Accumulate
    # if accumulate:
    #     return np.sum(fft2d_log_abs, axis=1)
    #
    # det_matrix, aoa_input = dsp.doppler_processing(range_data, num_tx_antennas=numTxAntennas,
    #                                                clutter_removal_enabled=True,
    #                                                window_type_2d=Window.HAMMING,
    #                                                interleaved=True, accumulate=True)

    start_bin_index = 8
    end_bin_index = 11

    # plot range profile
    for i in range(start_bin_index, end_bin_index):
        b_index = i
        # plot_range_profile(range_data, b_index)
        # plot change of amplitude
        # plot_amplitude_change_multi_in_one(range_data[..., b_index], b_index)
        # plot_amplitude_change_multi_in_one(range_data[..., b_index], b_index, is_diff=False)

        # phase change
        # plot_phase_change_in_one(range_data[..., b_index], b_index, is_diff=False)
        # plot_phase_change_in_one(range_data[..., b_index], b_index, is_diff=False)
        # phase_temporal_attention(range_data[..., b_index], b_index, is_diff=True)
        # phase_temporal_attentionv2(range_data[..., b_index], b_index, is_diff=True)
        get_phase_change_npy(range_data[..., 8:11], b_index, is_diff=True)

    if phase_only:
        sys.exit(0)

    ims = []
    max_size = 0
    # (1.5) Required Plot Declarations
    if plot2DscatterXY or plot2DscatterXZ:
        fig, axes = plt.subplots(1, 2)
    elif plot3Dscatter and plotMakeMovie:
        fig = plt.figure()
        nice = Axes3D(fig)
    elif plot3Dscatter:
        fig = plt.figure()
    elif plotRangeDopp:
        fig = plt.figure()
    elif plotCustomPlt:
        print("Using Custom Plotting")

    # Doppler plot
    for i, frame in enumerate(dataCube):
        #        print(i,end=',') # Frame tracker
        # (2) Range Processing
        from mmwave.dsp.utils import Window

        cluster = None

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        # det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas,
        #                                                clutter_removal_enabled=True,
        #                                                window_type_2d=Window.HAMMING,
        #                                                interleaved=True, accumulate=True)

        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas,
                                                       clutter_removal_enabled=True,
                                                       window_type_2d=Window.HAMMING,
                                                       interleaved=True, accumulate=True)

        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            # doppler_data[i] = det_matrix_vis
            if plotMakeMovie:
                ims.append((plt.imshow(det_matrix_vis / det_matrix_vis.max(), interpolation='nearest', aspect='auto'),))
            else:
                plt.imshow(det_matrix_vis / det_matrix_vis.max(), interpolation='nearest', aspect='auto')
                plt.title("Range-Doppler plot " + str(i))
                plt.pause(0.05)
                plt.clf()
                plt.show()

        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)

        # plt.imshow(fft2d_sum / fft2d_sum.max(), interpolation='nearest', aspect='auto')
        # plt.show()

        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=4,
                                                                  noise_len=16)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=16)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        # snr_data[i] = fft2d_sum - noiseFloorDoppler

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        # plot_raw_detection(detObj2DRaw, type='peakVal')
        # plot_raw_detection(detObj2DRaw, type='SNR')

        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numLoopsPerFrame, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numLoopsPerFrame)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])

        peakThreshold = np.mean(detObj2DRaw['peakVal']) - 1 * np.std(detObj2DRaw['peakVal'])
        peakValThresholds2 = np.array([[1, peakThreshold], ])

        # detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5,
        #                                    range_resolution)

        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, max_range=9, min_range=5,
                                           range_resolution=range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

        x, y, z = dsp.naive_xyz(azimuthInput.T, num_tx=numTxAntennas, num_rx=numRxAntennas, fft_size=numADCSamples)

        xyzVecN = np.zeros((3, x.shape[0]))
        xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
        xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']

        # retain range bin from 6 to 8
        # xyzVecN = retain_range(xyzVecN, range_resolution)

        Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
                                                                     range_resolution, method='Bartlett')

        # retain range bin from 6 to 8
        # xyzVec = retain_range(xyzVec, range_resolution)

        # (5) 3D-Clustering
        # detObj2D must be fully populated and completely accurate right here
        numDetObjs = detObj2D.shape[0]
        dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                        'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
        detObj2D_f = detObj2D.astype(dtf)
        detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 7)

        # Fully populate detObj2D_f with correct info
        for i, currRange in enumerate(Ranges):
            if i >= (detObj2D_f.shape[0]):
                # copy last row
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
            if currRange == detObj2D_f[i][0]:
                detObj2D_f[i][3] = xyzVec[0][i]
                detObj2D_f[i][4] = xyzVec[1][i]
                detObj2D_f[i][5] = xyzVec[2][i]
            else:  # Copy then populate
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
                detObj2D_f[i][3] = xyzVec[0][i]
                detObj2D_f[i][4] = xyzVec[1][i]
                detObj2D_f[i][5] = xyzVec[2][i]

                # radar_dbscan(epsilon, vfactor, weight, numPoints)
                # cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)

        if len(detObj2D_f) > 0:
            cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=True)

            cluster_np = np.array(cluster['size']).flatten()
            if cluster_np.size != 0:
                if max(cluster_np) > max_size:
                    max_size = max(cluster_np)

        # (6) Visualization
        if plotRangeDopp:
            continue
        if plot2DscatterXY or plot2DscatterXZ:

            if plot2DscatterXY:
                xyzVec = xyzVec[:, (np.abs(xyzVec[2]) < 1.5)]
                xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]
                axes[0].set_ylim(bottom=0, top=10)
                axes[0].set_ylabel('Range')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=0, top=10)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)

            elif plot2DscatterXZ:
                axes[0].set_ylim(bottom=-0.5, top=0.5)
                axes[0].set_ylabel('Elevation')
                axes[0].set_xlim(left=-0.4, right=0.4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=-0.5, top=0.5)
                axes[1].set_xlim(left=-0.4, right=0.4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)

            if plotMakeMovie and plot2DscatterXY:
                ims.append((axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=2),
                            axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=2)))
            elif plotMakeMovie and plot2DscatterXZ:
                ims.append((axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=2),
                            axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=2)))
            elif plot2DscatterXY:
                axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
            elif plot2DscatterXZ:
                axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=3)
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
        elif plot3Dscatter and plotMakeMovie:
            nice.set_zlim3d(bottom=-5, top=5)
            nice.set_ylim(bottom=0, top=10)
            nice.set_xlim(left=-4, right=4)
            nice.set_xlabel('X Label')
            nice.set_ylabel('Y Label')
            nice.set_zlabel('Z Label')

            ims.append((nice.scatter(xyzVec[0], xyzVec[1], xyzVec[2], c='r', marker='o', s=2),))

        elif plot3Dscatter and cluster is not None:
            if singFrameView:
                ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
            else:
                ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
                plt.pause(0.1)
                plt.clf()
        else:
            # sys.exit("Unknown plot options.")
            print("Unknown plot options.")

    if visTrigger and plotMakeMovie:
        movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)
