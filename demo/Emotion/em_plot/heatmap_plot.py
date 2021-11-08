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
import scipy
from scipy import signal
import os

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

DebugMode = True

if not DebugMode:
    import matplotlib

    matplotlib.use('Agg')

# QOL settings
loadData = True

figpath = "C:/Users/Zber/Desktop/SavedFigure"

# mmWave studio configure
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
device_v = True
# Test
adc_data_path = "C:/Users/Zber/Desktop/SavedData_MIMO/Joy_41_Raw_0.bin"

plotRangeDopp = True
plot2DscatterXY = False
plot2DscatterXZ = False
plot3Dscatter = False
plotCustomPlt = False

plotMakeMovie = True
makeMovieTitle = ""
makeMovieDirectory = "C:/Users/Zber/Documents/Dev_program/OpenRadar/demo/visualizer/movie/test_plotRDM.mp4"

antenna_order = [8, 10, 7, 9, 6, 4, 5, 3] if device_v else [5, 6, 7, 8, 3, 4, 9, 10]

visTrigger = plot2DscatterXY + plot2DscatterXZ + plot3Dscatter + plotRangeDopp + plotCustomPlt
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

singFrameView = False

# constant frequency
LM_LOWCUT = 0.2
LM_HIGHCUT = 3

fig_prefix = os.path.basename(adc_data_path)[:-4]

# virtual Antenna Array
virtual_array = []
for tx in [1, 3, 2]:
    for rx in range(1, 5):
        virtual_array.append((tx, rx))

tx_map = {1: 0, 3: 1, 2: 2}


def parseConfigFile(configFileName, numTxAnt=3):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
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


def movieMaker(fig, ims, title, save_dir):
    import matplotlib.animation as animation
    import matplotlib as mpl
    mpl.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\Zber\\Documents\\ffmpeg\\bin\\ffmpeg.exe"

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


def butter_bandpass(sig, lowcut, highcut, fs, order=5, output='ob'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='bandpass', output=output)
    filtered = signal.sosfreqz(sos, sig)
    return filtered


def butter_lowpass(sig, lowcut, highcut, fs, order=5, output='ob'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [high], btype='lowpass', output='ba')
    filtered = signal.sosfreqz(sos, sig)
    # pp = plt.semilogx((fs * 0.5 / np.pi) * w, abs(h), label=label)
    return filtered


def butter_bandpass_fs(sig, lowcut, highcut, fs, order=5, output='ba'):
    sos = signal.butter(order, [lowcut, highcut], btype='bandpass', output=output, fs=fs)
    filtered = signal.sosfreqz(sos, worN=sig)
    return filtered


def butter_lowpass_fs(sig, highcut, fs, order=5, output='ba'):
    sos = signal.butter(order, [highcut], btype='lowpass', output=output, fs=fs)
    filtered = signal.sosfreqz(sos, worN=sig)
    return filtered


def butter_highpass_fs(sig, lowcut, fs, order=5, output='ba'):
    sos = signal.butter(order, [lowcut], btype='highpass', output=output, fs=fs)
    filtered = signal.sosfreqz(sos, worN=sig)
    return filtered


def plot_virtual_antenna(range_data, is_diff=True):
    import matplotlib._color_data as mcd
    if device_v:
        fig, axes = plt.subplots(4, 2, figsize=(50, 90))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(90, 50))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]

    for ax, o, color in zip(fig.axes, antenna_order, tab_color):
        va_order = o - 1
        t, r = virtual_array[va_order]
        t, r = tx_map[t], r - 1
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
        fig.savefig("{}_multiphase.pdf".format(os.path.join(figpath, fig_prefix)))
    else:
        fig.savefig("{}_multiphase_unwrap.pdf".format(os.path.join(figpath, fig_prefix)))


def plot_virtual_antenna_in_one(range_data, bin_index=0, is_diff=True):
    import matplotlib._color_data as mcd
    if device_v:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]

    ax = axes
    for o, color in zip(antenna_order, tab_color):
        va_order = o - 1
        t, r = virtual_array[va_order]
        t, r = tx_map[t], r - 1
        va_sig = sig[:, t, r]
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

        ax.plot(va_diff_phase, linewidth=1, c=color, label='Virtual Antenna {}'.format(o), zorder=5)
        ax.scatter(plot_max_x, plot_max_y, c=color, s=5, zorder=10)
        ax.set_ylim([-4, 4]) if is_diff else None
    plt.legend(bbox_to_anchor=(0.837, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    plt.show() if DebugMode else None
    if is_diff:
        fig.savefig("{}_multiphase_in_one_{}.pdf".format(os.path.join(figpath, fig_prefix), bin_index))
    else:
        fig.savefig("{}_multiphase_in_one_{}_unwrap.pdf".format(os.path.join(figpath, fig_prefix), bin_index))


def plot_virtual_antenna_point(range_data):
    import matplotlib._color_data as mcd
    if device_v:
        fig, axes = plt.subplots(4, 2, figsize=(50, 90))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(90, 50))
    # v_order = [8, 10, 7, 9, 6, 4, 5, 3]
    sig = range_data.reshape((-1, numTxAntennas, numLoopsPerFrame, numRxAntennas))
    sig = sig[:, :, 5, :]

    tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]

    for ax, o, color in zip(fig.axes, antenna_order, tab_color):
        h = color[1:]
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

        va_order = o - 1
        t, r = virtual_array[va_order]
        t, r = tx_map[t], r - 1
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
    fig.savefig("{}_scatter.pdf".format(os.path.join(figpath, fig_prefix)))


def range_elevation_generator(virtual_antenna_pair, radar_cube):
    return 0


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
    SKIP_SIZE = 4
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
        VIRT_ANT_AZI = 4
        VIRT_ANT_ELE = 2

    static_clutter_removal = True
    diff = False
    z_score = False
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

    ims = []
    max_size = 0

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile(adc_data_path, dtype=np.int16)
        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        dataCube = np.copy(adc_data)
        print("Data Loaded!")

    # Start DSP processing
    num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT_AZI)
    num_vec_ele, steering_vec_ele = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT_ELE)

    fig, axes = plt.subplots(1, 4, figsize=(ANGLE_BINS // 5, BINS_PROCESSED // 5 * 4))
    frame_index = 0

    # stored np array
    cum_h = np.zeros((ANGLE_BINS, BINS_PROCESSED))
    cum_e = np.zeros((ANGLE_BINS, BINS_PROCESSED))

    # cum_h = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)
    # cum_e = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)

    pre_h = np.zeros((ANGLE_BINS, BINS_PROCESSED))
    pre_e = np.zeros((ANGLE_BINS, BINS_PROCESSED))

    for frame_index in range(80, 150):
        # for frame_index in range(numFrames):
        # frame_index += 1
        """ 1 (Range Processing) """

        frame = adc_data[frame_index]

        # --- range fft
        radar_cube = dsp.range_processing(frame)

        # virtual antenna arrangement
        radar_cube = arange_tx(radar_cube, num_tx=numTxAntennas, vx_axis=1, axis=0)

        """ 2 (Beamformer Processing) """
        # --- static clutter removal
        # --- Do we need ?
        if static_clutter_removal:
            mean = radar_cube.mean(0)
            radar_cube = radar_cube - mean

        # --- capon beamforming

        if non_coherent:
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
                range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED))
                # range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)
                beamWeights = np.zeros((VIRT_ANT_AZI, BINS_PROCESSED), dtype=np.complex_)

                for i in range(BINS_PROCESSED):
                    range_azimuth[:, i], beamWeights[:, i] = dsp.aoa_capon(cube_azi[:, :, i].T, steering_vec,
                                                                           magnitude=True)
                stack_azi.append(range_azimuth)

            for cube_ele in radar_cube_ele:
                range_elevation = np.zeros((ANGLE_BINS, BINS_PROCESSED))
                # range_elevation = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)
                beamWeights_ele = np.zeros((VIRT_ANT_ELE, BINS_PROCESSED), dtype=np.complex_)
                for i in range(BINS_PROCESSED):
                    range_elevation[:, i], beamWeights_ele[:, i] = dsp.aoa_capon(cube_ele[:, :, i].T,
                                                                                 steering_vec_ele, magnitude=True)
                stack_ele.append(range_elevation)

            range_azimuth = np.sum(stack_azi, axis=0)
            range_elevation = np.sum(stack_ele, axis=0)

        elif mode == 'bartlett':

            doa_spectrum_azi = dsp.aoa_bartlett(steering_vec, radar_cube_azi[..., :BINS_PROCESSED], axis=1)
            doa_spectrum_ele = dsp.aoa_bartlett(steering_vec_ele, radar_cube_ele[..., :BINS_PROCESSED], axis=1)
            range_azimuth = np.sum(doa_spectrum_azi, axis=0)
            range_elevation = np.sum(doa_spectrum_ele, axis=0)

        else:
            print("not supported yet!")
            sys.exit(0)

        """ 3 (Object Detection) """
        if is_log:
            range_azimuth = np.log2(range_azimuth)
            range_elevation = np.log2(range_elevation)

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

        axes[0].imshow(range_azimuth / range_azimuth.max(), interpolation='nearest', aspect='auto')
        # axes[0].imshow(np.angle(range_azimuth), interpolation='nearest', aspect='auto')

        axes[1].set_xlabel('Range')
        axes[1].set_ylabel('Elevation')
        axes[1].imshow(range_elevation / range_elevation.max(), interpolation='nearest', aspect='auto')
        # axes[1].imshow(np.angle(range_elevation), interpolation='nearest', aspect='auto')

        axes[2].set_xlabel('Range')
        axes[2].set_ylabel('Azimuth')
        axes[2].imshow(cum_h / cum_h.max(), interpolation='nearest', aspect='auto')

        axes[3].set_xlabel('Range')
        axes[3].set_ylabel('Elevation')
        axes[3].imshow(cum_e / cum_e.max(), interpolation='nearest', aspect='auto')

        plt.title("Range-Angle Heatmap " + str(frame_index), loc='center')
        plt.pause(0.02)
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[3].clear()
