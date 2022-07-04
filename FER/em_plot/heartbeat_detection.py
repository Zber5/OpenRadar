
# adc data path
adc_data_path = "C:/Users/Zber/Desktop/Subjects/Distance_1m/Surprise_1_Raw_0.bin"

def butter_bandpass_fs(sig, lowcut, highcut, fs, order=5, output='sos'):
    sos = signal.butter(order, [lowcut, highcut], btype='bandpass', output=output, fs=fs)
    filtered = signal.sosfilt(sos, sig)
    return filtered

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


if __name__ == '__main__':
    # file path
    adc_data_path = "C:/Users/Zber/Desktop/Subjects/Distance_1m/Surprise_1_Raw_0.bin"

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
    adc_data = np.fromfile(adc_data_path, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    dataCube = np.apply_along_axis(DCA1000.organize_cli, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)
    print("Data Loaded!")

    # range processing
    range_data = dsp.range_processing(adc_data)
    # range_data = dsp.range_processing(adc_data, window_type_1d=Window.HANNING)
    range_data = arange_tx(range_data, num_tx=numTxAntennas)

    bin_index = 7



