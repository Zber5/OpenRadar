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
from FER.utils import parseConfigFile, arange_tx

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

# figpath = "C:/Users/Zber/Desktop/mmWave_figure"
figpath = "C:/Users/Zber/Desktop/SavedFigure"


# Data and ConfigFile
# 64 chirps
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_20s_50fps_2.cfg'
adc_data_path = "C:/Users/Zber/Desktop/Subjects/Test/Chirps_0_Raw_0.bin"

# 3s
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'
adc_data_path = "D:\\Subjects\\S2\\Joy_33_Raw_0.bin"


if __name__ == '__main__':
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
    numAngleBins = 64

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

    bin_s = 7
    bin_e = 10

    det_matrix_vis_mean = np.mean(det_matrix_vis[:, bin_s:bin_e, :], axis=1)

    plt.imshow(np.abs(det_matrix_vis_mean.T), cmap=plt.cm.jet)
    plt.show()

    print("")
