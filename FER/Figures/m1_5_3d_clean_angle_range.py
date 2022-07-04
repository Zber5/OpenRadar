import os

os.chdir("C:\\Users\\Zber\\Documents\\Dev_program\\OpenRadar")
import numpy as np
import mmwave.dsp as dsp
import mmwave.dsp.music as music
from mmwave.dataloader import DCA1000
from mmwave.dsp.utils import Window
from FER.utils import parseConfigFile, arange_tx
from mmwave.dsp.cfar import ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# plotting color
import matplotlib._color_data as mcd

tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
extra_color = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04']
tab_color = tab_color + extra_color

# figpath = "C:/Users/Zber/Desktop/mmWave_figure"
figpath = "C:/Users/Zber/Desktop/SavedFigure"
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'

# additional colors
import matplotlib.colors as colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


cmap = plt.get_cmap('hot')
new_cmap = truncate_colormap(cmap, 0.0, 0.5)

if __name__ == "__main__":
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

    # aoa related
    VIRT_ELE_PAIRS = [[8, 2], [9, 3], [10, 4], [11, 5]]
    VIRT_AZI_PAIRS = [[i for i in range(0, 8)]]

    # azimuth
    ANGLE_RES_AZI = 1
    ANGLE_RANGE_AZI = 60
    ANGLE_BINS_AZI = (ANGLE_RANGE_AZI * 2) // ANGLE_RES_AZI + 1
    VIRT_ANT_AZI = 8

    # elevation
    ANGLE_RES_ELE = 1
    ANGLE_RANGE_ELE = 30
    ANGLE_BINS_ELE = (ANGLE_RANGE_ELE * 2) // ANGLE_RES_ELE + 1
    VIRT_ANT_ELE = 2

    BIN_RANG_S = 0
    BIN_RANG_E = 256
    BINS_PROCESSED = BIN_RANG_E - BIN_RANG_S
    VIRT_ANT_AZI_INDEX = [i for i in range(0, 8)]
    VIRT_ANT_ELE_INDEX = VIRT_ELE_PAIRS[2]

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
    unit = 0.042158314406249994

    # adc data path
    adc_data_path = "D:/Subjects/M2_1/Joy_0_Raw_0.bin"

    # Reading in adc data
    adc_data = np.fromfile(adc_data_path, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize_cli, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)
    print("Data Loaded!")

    # processing range data
    # window types : Bartlett, Blackman p, Hanning p and Hamming
    range_data = dsp.range_processing(adc_data)
    range_data = arange_tx(range_data, num_tx=numTxAntennas)

    # range profile
    print(range_data.shape)
    # fig5, axes5 = plt.subplots(1, 1, figsize=(16, 9))
    range_plot_data = np.mean(range_data[:, 15, :, :50], axis=(1))
    # range_plot_data = np.mean(range_data,axis=(1,2))
    # range_plot_data = range_data[:,15,2,:50]
    # axes5.imshow(np.abs(range_plot_data.T), cmap=plt.get_cmap('jet'))

    # %matplotlib widget
    s_bin = 0
    e_bin = s_bin + 100
    # %matplotlib inline
    # fig, axes = plt.subplots(1, 1, figsize=(16, 9))
    det_matrix, aoa_input = dsp.doppler_processing_frame(range_data, num_tx_antennas=numTxAntennas,
                                                         clutter_removal_enabled=True,
                                                         window_type_2d=Window.HAMMING,
                                                         accumulate=True)

    # det_matrix, aoa_input = doppler_processing(range_data, num_tx_antennas=numTxAntennas,
    #                                                     clutter_removal_enabled=True,
    #                                                     window_type_2d=Window.HAMMING,
    #                                                     accumulate=True)

    det_matrix_vis = np.fft.fftshift(det_matrix, axes=2)
    det_matrix_vis_mean = np.mean(det_matrix_vis[:, :, :], axis=0)
    # det_matrix_vis_mean = np.mean(det_matrix_vis[50:100, :, :], axis=0)
    bin_data = det_matrix_vis_mean[:, 17] + det_matrix_vis_mean[:, 15]

    peak_data = ca(bin_data, guard_len=2, noise_len=4, l_bound=8)[s_bin:e_bin]

    # axes.plot(peak_data)
    # axes.axvline(x=20, color='r', linestyle='-')
    # axes.axvline(x=24, color='r', linestyle='-')

    # axes.imshow(np.abs(det_matrix_vis_mean.T[:, s_bin:e_bin]), cmap=plt.get_cmap('jet'))

    offset = 5
    peak_data = peak_data[offset:offset + 200]
    detect_pos = np.where(peak_data == True)[0]
    detect_pos += offset
    print(detect_pos)

    # bins
    bin_start = 21
    bin_end = 26

    print("start {}, end {}".format(bin_start, bin_end))
    num_vec_azi, steering_vec_azi = dsp.gen_steering_vec(ANGLE_RANGE_AZI, ANGLE_RES_AZI, VIRT_ANT_AZI)
    num_vec_ele, steering_vec_ele = dsp.gen_steering_vec(ANGLE_RANGE_ELE, ANGLE_RES_ELE, VIRT_ANT_ELE)

    # aoa shape
    print(aoa_input.shape)
    aoa_input = np.transpose(aoa_input, (0, 3, 2, 1))
    print(aoa_input.shape)

    # micro doppler plot
    d_start = bin_start
    d_end = bin_end + 1

    # fig3, axes3 = plt.subplots(1, 1, figsize=(16, 9))
    det_matrix_vis_mean = np.mean(det_matrix_vis[:, d_start:d_end, :], axis=1)
    # axes3.imshow(np.abs(det_matrix_vis_mean.T), cmap=plt.cm.jet)

    # capon processing
    num_bins = bin_end - bin_start
    npy_azi = np.zeros((numFrames, ANGLE_BINS_AZI, num_bins))
    npy_ele = np.zeros((numFrames, ANGLE_BINS_ELE, num_bins))
    for i in range(0, 300):
        rb = 0
        for r in range(bin_start, bin_end):
            chirp_data_azi = aoa_input[i, :, VIRT_ANT_AZI_INDEX, r]
            # capon beamformer
            capon_angle_azi, beamWeights_azi = dsp.aoa_capon(chirp_data_azi, steering_vec_azi, magnitude=True)
            npy_azi[i, :, rb] = capon_angle_azi

            chirp_data_ele = aoa_input[i, :, VIRT_ANT_ELE_INDEX, r]
            # capon beamformer
            capon_angle_ele, beamWeights_ele = dsp.aoa_capon(chirp_data_ele, steering_vec_ele, magnitude=True)
            npy_ele[i, :, rb] = capon_angle_ele

            rb += 1

    # music processing
    ar_sb = 0
    ar_eb = 50

    num_bins = ar_eb - ar_sb
    ar_npy_azi = np.zeros((numFrames, ANGLE_BINS_AZI, num_bins))
    ar_npy_ele = np.zeros((numFrames, ANGLE_BINS_ELE, num_bins))
    for i in range(0, 300):
        rb = 0
        for r in range(ar_sb, ar_eb):
            chirp_data_azi = range_data[i, :, VIRT_ANT_AZI_INDEX, r]
            # capon beamformer
            # capon_angle_azi, beamWeights_azi = dsp.aoa_capon(chirp_data_azi, steering_vec_azi, magnitude=True)
            capon_angle_azi = music.aoa_music_1D(steering_vec_azi, chirp_data_azi, 1)
            ar_npy_azi[i, :, rb] = capon_angle_azi

            chirp_data_ele = range_data[i, :, VIRT_ANT_ELE_INDEX, r]
            # capon beamformer
            # capon_angle_ele, beamWeights_ele = dsp.aoa_capon(chirp_data_ele, steering_vec_ele, magnitude=True)
            capon_angle_ele = music.aoa_music_1D(steering_vec_ele, chirp_data_ele, 1)

            ar_npy_ele[i, :, rb] = capon_angle_ele
            rb += 1

    # plt.rcParams["axes.grid"] = False
    # fig1, axes1 = plt.subplots(1, 1, figsize=(8, 6))
    #
    # txt = np.arange(-60, 61, step=20)
    # pos = np.linspace(1, 119, 7)
    # axes1.set_yticks(pos)
    # axes1.set_yticklabels(txt, fontsize=20)
    # xpos_label = np.arange(0, 4, step=1)
    # xpos = xpos_label * 99.5
    # axes1.set_xticks(xpos)
    # axes1.set_xticklabels(xpos_label, fontsize=25)
    # axes1.set_ylabel("Angle (degree)", fontsize=35)
    # axes1.set_xlabel("Time (s)", fontsize=35)
    # axes1.imshow(np.mean(npy_azi, 2).T, cmap=plt.cm.jet, aspect='auto')


    # heatmap
    # fig2, axes2 = plt.subplots(1, 1, figsize=(8, 5))
    # sub 1
    ee = 50
    # axes2.imshow(ar_npy_azi[ee], cmap=plt.cm.jet, aspect='auto')
    aas = 52
    aae = 66



    bin_start = 20
    bin_end = 25


    plot_data = ar_npy_azi[ee]
    plot_data[:aas, :] = 0
    plot_data[aae:, :] = 0
    plot_data[:, :bin_start] = 0
    plot_data[:, bin_end:] = 0
    # plot_data  = plot_data - 0.2620
    # 3D surface
    X = np.arange(-60, 61, step=1)
    Y = np.arange(0, (50 - 0.01) * unit, step=unit)
    X, Y = np.meshgrid(X, Y)
    Z = plot_data.T - 0.023

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                           edgecolor='darkred', linewidth=0.1)

    plt.show()
