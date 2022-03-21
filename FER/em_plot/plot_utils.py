from scipy import signal
import matplotlib._color_data as mcd

tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
extra_color = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04']
tab_color = tab_color + extra_color







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