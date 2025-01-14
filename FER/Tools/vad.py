import numpy as np
import scipy as sp
import wave

WINSIZE = 1024
sound = 'C:/Users/Zber/Downloads/VAD-master/data/example/SNR103F3MIC021002_ch01.wav'


def read_signal(filename, winsize):
    wf = wave.open(filename, 'rb')
    n = wf.getnframes()
    str = wf.readframes(n)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen = ((int)(len(str) / 2 / winsize) + 1) * winsize
    signal = np.zeros(siglen, np.int16)
    signal[0:len(str) // 2] = np.fromstring(str, np.int16)
    return [signal, params]


def get_frame(signal, winsize, no):
    shift = winsize // 2
    start = no * shift
    end = start + winsize
    return signal[start:end]


class LTSD():
    def __init__(self, winsize, window, order):
        self.winsize = winsize
        self.window = window
        self.order = order
        self.amplitude = {}

    def get_amplitude(self, signal, l):
        if l in self.amplitude.keys():
            return self.amplitude[l]
        else:
            amp = np.abs(np.fft.fft(get_frame(signal, self.winsize, l) * self.window))
            self.amplitude[l] = amp
            return amp

    def compute_noise_avg_spectrum(self, nsignal):
        windownum = len(nsignal) // (self.winsize // 2) - 1
        avgamp = np.zeros(self.winsize)
        for l in range(windownum):
            avgamp += np.abs(np.fft.fft(get_frame(nsignal, self.winsize, l) * self.window))
        return avgamp / float(windownum)

    def compute(self, signal):
        self.windownum = len(signal) // (self.winsize // 2) - 1
        ltsds = np.zeros(self.windownum)
        # Calculate the average noise spectrum amplitude based　on 20 frames in the head parts of input signal.
        self.avgnoise = self.compute_noise_avg_spectrum(signal[0:self.winsize * 20]) ** 2
        for l in range(self.windownum):
            ltsds[l] = self.ltsd(signal, l, 5)
        return ltsds

    def ltse(self, signal, l, order):
        maxmag = np.zeros(self.winsize)
        for idx in range(l - order, l + order + 1):
            amp = self.get_amplitude(signal, idx)
            maxmag = np.maximum(maxmag, amp)
        return maxmag

    def ltsd(self, signal, l, order):
        if l < order or l + order >= self.windownum:
            return 0
        return 10.0 * np.log10(np.sum(self.ltse(signal, l, order) ** 2 / self.avgnoise) / float(len(self.avgnoise)))


if __name__ == "__main__":
    signal, params = read_signal(sound, WINSIZE)
    window = sp.hanning(WINSIZE)
    ltsd = LTSD(WINSIZE, window, 5)
    res = ltsd.compute(signal)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(res)
    plt.show()
