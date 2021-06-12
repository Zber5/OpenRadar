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

DebugMode = False

if not DebugMode:
    import matplotlib

    matplotlib.use('Agg')

# QOL settings
loadData = True

figpath = "C:/ti/mySavedFig"

# Configure file
# configFileName = "C:/Users/Zber/Desktop/Tx3_bestRangeResolution.cfg"
# configFileName = "C:/Users/Zber/Desktop/bestRangeResolution.cfg"
# configFileName = 'C:/Users/Zber/Desktop/vod_vs_18xx_10fps.cfg'
# configFileName ='C:/Users/Zber/Desktop/profile_2021_04_04T06_04_18_277.cfg'


# range resolution 0.8cm, tx1 and rx1
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_12345_r8_0.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_static_r8_0.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_12345_r8_1.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_123_r8_0.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_123_r8_1.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1234_r8_1.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_r8_0.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_static_r8_0.bin', dtype=np.int16)
# adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_static_r8_1.bin', dtype=np.int16)
# configFileName ='C:/Users/Zber/Desktop/mmWave Configuration/tx1_rx4_1.cfg'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_1234_r8_0.bin'

# 100 fps
# configFileName ='C:/Users/Zber/Desktop/mmWave Configuration/100fps_estimator.cfg'
# configFileName ='C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange.cfg'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_12_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_12345_Raw_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_1_Raw_0.bin'

# npy data
# npy_data_path = 'C:/ti/parser_scripts/adcdata.npy'

# tx3_rx4_bestRange 50fps 300frames
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx3_rx4_bestRange.cfg'

# frame 300
# adc_data_path = 'C:/ti/mySavedData/LipMotion_tx3_rx4_static_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_tx3_rx4_1234_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_tx3_rx4_1234_1.bin'
# frame 150
# adc_data_path = 'C:/ti/mySavedData/LipMotion_flame_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_ear_sound_0.bin'

# frame 600
# adc_data_path = 'C:/ti/mySavedData/LipMotion_ear_sound_600_0.bin' #1.5m
# adc_data_path = 'C:/ti/mySavedData/LipMotion_ear_sound_600_1.bin' #1m
# adc_data_path = 'C:/ti/mySavedData/LipMotion_ear_sound_600_2.bin' #30cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_ear_sound_600_3.bin' #10cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_eardrum_10cm.bin' #10cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_ear_sound_static_0.bin' # static

# vocal cord
# adc_data_path = 'C:/ti/mySavedData/LipMotion_vocalcord.bin' # 20cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_vocalcord_2m.bin' # 200 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_vocalcord_3m.bin' # 300 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_vcv_2m.bin' # 300 cm

# adc_data_path = 'C:/ti/mySavedData/LipMotion_lipmotion_2m.bin'  # 200 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_lipmotion_3m.bin' # 200 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_2m.bin' # 200 cm

# long distance
# adc_data_path = 'C:/ti/mySavedData/LipMotion_vcv_4m_1.bin' # 200 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_lipmotion_4m_0.bin'  # 200 cm

# emotion
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion.bin'


# happy
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_happy_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_happy_1.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_happy_v_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_happy_v_1.bin'

# anger
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_anger_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_anger_1.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_anger_v_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_anger_v_1.bin'

# superise
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_suprise_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_suprise_1.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_surprise_v_0.bin'
# adc_data_path = 'C:/ti/mySavedData/LipMotion_emotion_surprise_v_1.bin'

# fans
# adc_data_path = 'C:/ti/mySavedData/LipMotion_fan_0.bin' # 200 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_fan_static.bin' # 200 cm

# disk
# adc_data_path = 'C:/ti/mySavedData/LipMotion_disk_0.bin' # 200 cm
# adc_data_path = 'C:/ti/mySavedData/LipMotion_disk_static.bin' # 200 cm


# stereo
# adc_data_path = 'C:/ti/mySavedData/LipMotion_stereo_0.bin' # 200 cm
adc_data_path = 'C:/ti/mySavedData/LipMotion_stereo_static_0.bin' # 200 cm


plotRangeDopp = True
plot2DscatterXY = False
plot2DscatterXZ = False
plot3Dscatter = False
plotCustomPlt = False

plotMakeMovie = True
makeMovieTitle = ""
makeMovieDirectory = "./test_plot3Dscatter.mp4"

visTrigger = plot2DscatterXY + plot2DscatterXZ + plot3Dscatter + plotRangeDopp + plotCustomPlt
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

singFrameView = False

# constant frequency
LM_LOWCUT = 0.2
LM_HIGHCUT = 3

fig_prefix = os.path.basename(adc_data_path)[:-4]


def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = numRxAntennas
        numTxAnt = numTxAntennas

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

    # Set up formatting for the Range Azimuth heatmap movies
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

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
    numAngleBins = 64

    # data processing parameter
    numDisplaySamples = 150

    range_resolution, bandwidth = dsp.range_resolution(numADCSamples,
                                                       dig_out_sample_rate=configParameters['digOutSampleRate'],
                                                       freq_slope_const=configParameters['freqSlopeConst'])
    doppler_resolution = dsp.doppler_resolution(bandwidth, start_freq_const=configParameters['startFreq'],
                                                ramp_end_time=configParameters['rampEndTime'],
                                                idle_time_const=configParameters['idleTime'],
                                                num_loops_per_frame=configParameters['numLoops'],
                                                num_tx_antennas=numTxAntennas)
    print(f'Range Resolution: {range_resolution}, Bandwidth: {bandwidth}, Doppler Resolution: {doppler_resolution}')

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

    from mmwave.dsp.utils import Window

    # figure preparation
    fig, axes = plt.subplots(3, 2, figsize=(120, 60))

    # (1) processing range data
    # window types : Bartlett, Blackman p, Hanning p and Hamming
    range_data = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)
    range_data_copy = np.copy(range_data)
    selected_range_data = range_data[:, :, 3, :numDisplaySamples]

    # plot range profile
    range_profile = selected_range_data.reshape((-1, numDisplaySamples))
    axes[0, 0].imshow(np.abs(range_profile).T, interpolation='nearest', aspect='auto')

    # (2) variance magnitude determine argmax
    var = np.var(np.abs(selected_range_data), axis=(0, 1))
    bin_index = np.argmax(var, axis=0)
    bin_index = 103
    # bin_index = 6

    # plot variance
    axes[0, 1].bar(np.arange(0, numDisplaySamples), var)

    # (3) Select the range bin corresponding to the user
    select_bin_data = selected_range_data[..., bin_index]

    # chirp selection
    # select_bin_data = select_bin_data[:, 0::3]
    select_bin_data = select_bin_data[:, 5]

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
    plt.savefig("{}_phase.pdf".format(os.path.join(figpath, fig_prefix)))
    print("")

    # (1.5) Required Plot Declarations
    #     if plot2DscatterXY or plot2DscatterXZ:
    #         fig, axes = plt.subplots(1, 2)
    #     elif plot3Dscatter and plotMakeMovie:
    #         fig = plt.figure()
    #         nice = Axes3D(fig)
    #     elif plot3Dscatter:
    #         fig = plt.figure()
    #     elif plotRangeDopp:
    #         fig = plt.figure()
    #     elif plotCustomPlt:
    #         print("Using Custom Plotting")

    # (1.6) Optional single frame view
    # if singFrameView:
    #     dataCube = np.zeros((1, numChirpsPerFrame, 8, numADCSamples), dtype=complex)
    #     dataCube[0, :, :, :] = adc_data[299]
    # else:
    #     dataCube = adc_data

    # dataCube = adc_data
    doppler_data = np.zeros((numFrames, numADCSamples, numChirpsPerFrame // numTxAntennas))
    snr_data = np.zeros((numFrames, numADCSamples, numChirpsPerFrame // numTxAntennas))

    for i, frame in enumerate(dataCube):
        #        print(i,end=',') # Frame tracker
        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas,
                                                       clutter_removal_enabled=True,
                                                       window_type_2d=Window.HAMMING)

        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            doppler_data[i] = det_matrix_vis
            if plotMakeMovie:
                ims.append((plt.imshow(det_matrix_vis / det_matrix_vis.max()),))
            else:
                plt.imshow(det_matrix_vis / det_matrix_vis.max())
                plt.title("Range-Doppler plot " + str(i))
                # plt.pause(0.05)
                # plt.clf()
                plt.show()

        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
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

        snr_data[i] = fft2d_sum - noiseFloorDoppler

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)
        #
        # azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        #
        # x, y, z = dsp.naive_xyz(azimuthInput.T, fft_size=16)
        # xyzVecN = np.zeros((3, x.shape[0]))
        # xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        # xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
        # xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']
        #
        # Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
        #                                                              range_resolution, method='Bartlett')
        #
        # # (5) 3D-Clustering
        # # detObj2D must be fully populated and completely accurate right here
        # numDetObjs = detObj2D.shape[0]
        # dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
        #                 'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
        # detObj2D_f = detObj2D.astype(dtf)
        # detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 7)
        #
        # # Fully populate detObj2D_f with correct info
        # for i, currRange in enumerate(Ranges):
        #     if i >= (detObj2D_f.shape[0]):
        #         # copy last row
        #         detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
        #     if currRange == detObj2D_f[i][0]:
        #         detObj2D_f[i][3] = xyzVec[0][i]
        #         detObj2D_f[i][4] = xyzVec[1][i]
        #         detObj2D_f[i][5] = xyzVec[2][i]
        #     else:  # Copy then populate
        #         detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
        #         detObj2D_f[i][3] = xyzVec[0][i]
        #         detObj2D_f[i][4] = xyzVec[1][i]
        #         detObj2D_f[i][5] = xyzVec[2][i]
        #
        #         # radar_dbscan(epsilon, vfactor, weight, numPoints)
        # #        cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)
        # if len(detObj2D_f) > 0:
        #     cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=True)
        #
        #     cluster_np = np.array(cluster['size']).flatten()
        #     if cluster_np.size != 0:
        #         if max(cluster_np) > max_size:
        #             max_size = max(cluster_np)

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
                axes[0].set_ylim(bottom=-5, top=5)
                axes[0].set_ylabel('Elevation')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=-5, top=5)
                axes[1].set_xlim(left=-4, right=4)
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

        # elif plot3Dscatter:
        #     if singFrameView:
        #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
        #     else:
        #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
        #         plt.pause(0.1)
        #         plt.clf()
        # else:
        #     sys.exit("Unknown plot options.")

    # plot doppler
    fig, ax = plt.subplots(1, 1, figsize=(45, 30))
    ax.imshow(doppler_data[:, :, 13].T)
    # ax.imshow(doppler_data[:, bin_index, :].T)
    plt.show() if DebugMode else None
    fig.savefig("{}_doppler.pdf".format(os.path.join(figpath, fig_prefix)))

    # plot doppler variance
    fig, ax = plt.subplots(1, 1, figsize=(45, 30))
    # doppler_var = np.std(doppler_data[:, :, bin_index], axis=1)
    doppler_var = np.std(doppler_data[:, bin_index, :], axis=1)
    # doppler_var = np.std(doppler_data[:, :, 13], axis=1)
    # plt.plot(doppler_var)
    # plt.savefig("{}_dopplervar.pdf".format(os.path.join(figpath, fig_prefix)))

    # find doppler variance peaks
    peaks, _ = signal.find_peaks(doppler_var, height=7.5)
    ax.plot(doppler_var)
    ax.plot(peaks, doppler_var[peaks], "x", markersize=20)
    plt.show() if DebugMode else None
    fig.savefig("{}_dopplervar.pdf".format(os.path.join(figpath, fig_prefix)))
    # plt.show()

    # plot snr
    fig, ax = plt.subplots(1, 1, figsize=(45, 30))
    # ax.imshow(snr_data[:, :, 13].T)
    ax.imshow(snr_data[:, bin_index, :].T)
    plt.show() if DebugMode else None
    fig.savefig("{}_snr.pdf".format(os.path.join(figpath, fig_prefix)))
    # plt.show()

    # plos snr line
    fig, ax = plt.subplots(1, 1, figsize=(45, 30))
    # var = np.std(snr_data[:, bin_index, 5], axis=1)
    # avg = np.mean(snr_data[:, bin_index, :], axis=1)
    avg = snr_data[:, bin_index, 5]
    # bin_index = np.argmax(var, axis=0)
    # ax.plot(var)
    ax.plot(avg)
    # plt.show()
    fig.savefig("{}_snravg.pdf".format(os.path.join(figpath, fig_prefix)))

    print("")
"""

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

        # (1.6) Optional single frame view
        if singFrameView:
            dataCube = np.zeros((1, numChirpsPerFrame, 8, numADCSamples), dtype=complex)
            dataCube[0, :, :, :] = adc_data[299]
        else:
            dataCube = adc_data

        for i, frame in enumerate(dataCube):
            #        print(i,end=',') # Frame tracker
            # (2) Range Processing
            from mmwave.dsp.utils import Window

            radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
            assert radar_cube.shape == (
                numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

            # (3) Doppler Processing
            det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=2, clutter_removal_enabled=True,
                                                           window_type_2d=Window.HAMMING)

            # --- Show output
            if plotRangeDopp:
                det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
                if plotMakeMovie:
                    ims.append((plt.imshow(det_matrix_vis / det_matrix_vis.max()),))
                else:
                    plt.imshow(det_matrix_vis / det_matrix_vis.max())
                    plt.title("Range-Doppler plot " + str(i))
                    plt.pause(0.05)
                    plt.clf()

            # (4) Object Detection
            # --- CFAR, SNR is calculated as well.
            fft2d_sum = det_matrix.astype(np.int64)
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

            dtype_location = '(' + str(numTxAntennas) + ',)<f4'
            dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                       'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
            detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
            detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
            detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
            detObj2DRaw['peakVal'] = peakVals.flatten()
            detObj2DRaw['SNR'] = snr.flatten()

            # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
            detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

            # --- Peak Grouping
            # detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
            # SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
            # peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
            # detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)
            #
            # azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
            #
            # x, y, z = dsp.naive_xyz(azimuthInput.T, fft_size=16)
            # xyzVecN = np.zeros((3, x.shape[0]))
            # xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
            # xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
            # xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']
            #
            # Psi, Theta, Ranges, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
            #                                                              range_resolution, method='Bartlett')
            #
            # # (5) 3D-Clustering
            # # detObj2D must be fully populated and completely accurate right here
            # numDetObjs = detObj2D.shape[0]
            # dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
            #                 'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
            # detObj2D_f = detObj2D.astype(dtf)
            # detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 7)
            #
            # # Fully populate detObj2D_f with correct info
            # for i, currRange in enumerate(Ranges):
            #     if i >= (detObj2D_f.shape[0]):
            #         # copy last row
            #         detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
            #     if currRange == detObj2D_f[i][0]:
            #         detObj2D_f[i][3] = xyzVec[0][i]
            #         detObj2D_f[i][4] = xyzVec[1][i]
            #         detObj2D_f[i][5] = xyzVec[2][i]
            #     else:  # Copy then populate
            #         detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
            #         detObj2D_f[i][3] = xyzVec[0][i]
            #         detObj2D_f[i][4] = xyzVec[1][i]
            #         detObj2D_f[i][5] = xyzVec[2][i]
            #
            #         # radar_dbscan(epsilon, vfactor, weight, numPoints)
            # #        cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)
            # if len(detObj2D_f) > 0:
            #     cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=True)
            #
            #     cluster_np = np.array(cluster['size']).flatten()
            #     if cluster_np.size != 0:
            #         if max(cluster_np) > max_size:
            #             max_size = max(cluster_np)

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
                    axes[0].set_ylim(bottom=-5, top=5)
                    axes[0].set_ylabel('Elevation')
                    axes[0].set_xlim(left=-4, right=4)
                    axes[0].set_xlabel('Azimuth')
                    axes[0].grid(b=True)

                    axes[1].set_ylim(bottom=-5, top=5)
                    axes[1].set_xlim(left=-4, right=4)
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

            # elif plot3Dscatter:
            #     if singFrameView:
            #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
            #     else:
            #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
            #         plt.pause(0.1)
            #         plt.clf()
            else:
                sys.exit("Unknown plot options.")

        if visTrigger and plotMakeMovie:
            movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)
    
"""
