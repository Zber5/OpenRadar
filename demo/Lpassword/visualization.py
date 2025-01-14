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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

# QOL settings
loadData = True

# Configure file
# configFileName = "C:/Users/Zber/Desktop/Tx3_bestRangeResolution.cfg"
# configFileName = "C:/Users/Zber/Desktop/bestRangeResolution.cfg"
# configFileName = 'C:/Users/Zber/Desktop/vod_vs_18xx_10fps.cfg'

# caputure raw data
# configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/tx1_rx4_0.cfg'

# range resolution 8cm
configFileName ='C:/Users/Zber/Desktop/mmWave Configuration/tx1_rx4_1.cfg'
# configFileName ='C:/Users/Zber/Desktop/vocal_print_config.cfg'

# numFrames = 300
# numADCSamples = 128
# total_lenght = 230162432 // 2
# numFrames = 8
# numFrames = 439

numFrames = 100
numADCSamples = 256
numTxAntennas = 1
numRxAntennas = 1
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

range_resolution, bandwidth = dsp.range_resolution(numADCSamples)
doppler_resolution = dsp.doppler_resolution(bandwidth)

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
            configParameters['framePeriodicity'] = framePeriodicity

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


# def phase(z): # Calculates the phase of a complex number
#   r = np.absolute(z)
#   return (z.real/r + 1j * z.imag/r)

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


if __name__ == '__main__':
    # num Antennas
    numTxAntennas = 1
    numRxAntennas = 1
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

    range_profile = []

    # (1) Reading in adc data
    if loadData:
        # adc_data1 = np.fromfile('./data/1_person_walking_128loops.bin', dtype=np.uint16)
        # adc_data = np.fromfile('C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data.bin',dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/eye_blinking_Raw_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/lipmotion_F100_1_Raw_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/lipmotion_F100_2_Raw_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/lipmotion_F500_1_Raw_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/lipmotion_F500_2_Raw_0.bin', dtype=np.int16)

        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_Raw_0_A_2.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_Raw_0_A_3.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_12345_12345_2.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_Raw_0_E_1.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_Raw_0_E_2.bin', dtype=np.int16)

        # adc_data = np.load("C:/ti/parser_scripts/adcdata.npy")
        # adc_data = adc_data[:total_lenght]

        # range resolution 0.5cm LipMotion_1_12345_12345_20cm.bin
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_12345_12345_20cm.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_12345_12345_20cm_1.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_1234_1234_20cm_0.bin', dtype=np.int16)

        # range resolution 0.8cm
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_12345_r8_0.bin', dtype=np.int16)
        adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_static_r8_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_12345_r8_1.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_123_r8_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_123_r8_1.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1234_r8_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1234_r8_1.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_r8_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_static_r8_0.bin', dtype=np.int16)
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_static_r8_1.bin', dtype=np.int16)



        # static
        # adc_data = np.fromfile('C:/ti/mySavedData/LipMotion_1_static_1.bin', dtype=np.int16)

        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Data Loaded!")

    # data segmentation
    # adc_data = adc_data[200:300, :, 0, :]

    """
    numDisplaySamples = 20
    from mmwave.dsp.utils import Window

    # processing data
    # window types : Bartlett, Blackman p, Hanning p and Hamming
    adc_data = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)
    # adc_data = dsp.range_processing(adc_data)
    range_data = np.copy(adc_data)
    # adc_data = adc_data[:, 0::2, 0, :numDisplaySamples]
    # adc_data = adc_data[:, 0::numTxAntennas, 0, :numDisplaySamples]
    adc_data = adc_data[:, :, 0, :numDisplaySamples]

    # figure
    fig, axes = plt.subplots(3, 2, figsize=(30, 60))

    frame_data = adc_data.reshape((-1, numDisplaySamples))
    # plt.imshow(np.abs(frame_data).T, interpolation='nearest', aspect='auto')
    axes[0, 0].imshow(np.abs(frame_data).T, interpolation='nearest', aspect='auto')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Range')
    # plt.show()
    # print("")

    # get the duration of eye blinking
    # adc_data = adc_data[290:291, :, 1, :]
    # buffer = []
    # for frame in adc_data:
    #     radar_cube = dsp.range_processing(frame)
    # range_plot = adc_data.reshape((-1, numADCSamples))
    # fig, axes = plt.subplots(1, 1, figsize=(80,100))
    # range_plot = radar_cube[:, 0, :]
    # buffer.append(range_plot)

    # plt.imshow(np.abs(range_plot).T, interpolation='nearest', aspect='auto')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Range')
    # plt.show()
    # print("")
    # break
    # total_frame = np.asarray(buffer)

    # total_frame = total_frame.reshape((-1,numADCSamples))
    # plt.imshow(np.abs(total_frame).T, interpolation='nearest', aspect='auto')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Range')
    # plt.show()
    # print("")

    # radar_cube = dsp.range_processing(adc_data)
    # range_plot = radar_cube[0, :, 0, :]
    # plt.imshow(np.abs(range_plot).T, interpolation='nearest', aspect='auto')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Range')
    # plt.show()
    # print("")

    # radar_cube = dsp.range_processing(adc_data)
    # radar_cube = radar_cube[:, :, 0, :]
    # range_plot = radar_cube.reshape((-1, numADCSamples))
    # plt.imshow(np.abs(range_plot).T, interpolation='nearest', aspect='auto')
    # plt.ylabel('Range Bins')
    # plt.title('Interpreting a Single Frame - Range')
    # plt.show()

    # obtain the range bin with the max value
    # bin_index = np.argmax(adc_data.mean((0, 1, 2)), axis=0)
    # bin_index = 8
    # bin_index = 248

    # variance magnitude determine argmax
    var = np.var(adc_data, axis=(0, 1))
    # #
    # bin_index = np.argmax(var, axis=0)
    # plt.bar(np.arange(numADCSamples),var)
    bin_index = np.argmax(var, axis=0)
    # bin_index = 6
    axes[0, 1].bar(np.arange(0, numDisplaySamples), var)
    # plt.tight_layout()
    # plt.bar(np.arange(0, numDisplaySamples), var)
    # plt.title('Range Bin Variance')
    # plt.show()
    # bin_index = 3
    # for frame in adc_data:

    # (2) Range Processing)
    # --- range fft
    # radar_cube = dsp.range_processing(adc_data)
    # radar_cube = dsp.doppler_processing(radar_cube)
    # radar_cube = radar_cube[:,0,0,:]

    # (3) Select the range bin corresponding to the user
    select_range_bin = adc_data[..., bin_index]

    # (4) subtract mean value of related raw signal
    # mean = select_range_bin.mean(0, keepdims=True)
    # x = select_range_bin - mean
    x = select_range_bin

    # (5) unwrapping phase
    # extract phase
    x = np.angle(x)
    x = x.reshape(-1)
    unwrap = np.copy(x)

    # x = x[:, 5]

    axes[1, 0].plot(x)
    # plt.plot(x)
    # plt.title('Phase Radians')
    # plt.show()

    x = np.unwrap(x)

    axes[1, 1].plot(x)
    # plt.plot(x)
    # plt.show()
    # plt.title("Phase Unwrapped")
    # plt.show()

    x = np.diff(x)
    axes[2, 0].set_ylim([-3,3])
    # axes[2, 0].plot(x[1100:1600])
    axes[2, 0].plot(x)

    # x1 = np.diff(unwrap)
    # x1 = np.diff(unwrap)
    # axes[2, 1].plot(x1)

    # x1 = butter_bandpass(sig=unwrap, lowcut=LM_LOWCUT, highcut=LM_HIGHCUT, fs=30)
    # x1 = butter_bandpass_fs(sig=unwrap, lowcut=LM_LOWCUT, highcut=LM_HIGHCUT, fs=10, output='sos')
    x1 = butter_bandpass_fs(sig=unwrap, lowcut=1.5, highcut=3, fs=50, output='sos')
    # axes[2, 1].plot(np.abs(x1[1]))
    axes[2, 1].set_ylim([-4, 4])
    axes[2, 1].plot(np.abs(x1[1]))
    # axes[2, 1] = pp
    # fig, axes = plt.subplots(1, 1)

    # plt.plot(x)
    # plt.tight_layout()
    plt.show()
    print("")
    # phase unwrapping

    # doppler processing
    # det_matrix, aoa_input = dsp.doppler_processing(range_data, num_tx_antennas=1, clutter_removal_enabled=False, interleaved=False,
    #                                                window_type_2d=Window.HAMMING)
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
    # if singFrameView:
    #     dataCube = np.zeros((1, numChirpsPerFrame, 8, numADCSamples), dtype=complex)
    #     dataCube[0, :, :, :] = adc_data[299]
    # else:
    #     dataCube = adc_data

    dataCube = adc_data
    doppler_data = np.zeros((numFrames,numADCSamples,numChirpsPerFrame))
    snr_data =  np.zeros((numFrames,numADCSamples,numChirpsPerFrame))

    for i, frame in enumerate(dataCube):
        #        print(i,end=',') # Frame tracker
        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=numTxAntennas, clutter_removal_enabled=True,
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

    print("")
    # if visTrigger and plotMakeMovie:
    #     movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)

