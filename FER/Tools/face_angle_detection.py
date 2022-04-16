import numpy as np
import scipy as sp
import wave


def get_frame(signal, a):
    return signal[a]


class LTSD():
    def __init__(self, x, winsize, order, noise_size):
        self.signal = x
        self.winsize = winsize
        self.order = order
        self.amplitude = {}
        self.noise_size = noise_size
        self.windownum = len(x)
        self.bins = np.linspace(np.min(x), np.max(x), winsize + 1)

    def get_amplitude(self, l):
        if l in self.amplitude.keys():
            return self.amplitude[l]
        else:
            # amp = np.abs(np.fft.fft(get_frame(signal, self.winsize, l) * self.window))
            amp = np.histogram(get_frame(self.signal, l), self.bins)[0]
            self.amplitude[l] = amp
            return amp

    def compute_noise_avg_spectrum(self):
        avgamp = np.zeros(self.winsize)
        for l in range(self.noise_size):
            avgamp += np.histogram(get_frame(self.signal, l), self.bins)[0]
        return avgamp / float(self.noise_size)

    def compute(self):
        ltsds = np.zeros(self.windownum)
        # Calculate the average noise spectrum amplitude based　on 20 frames in the head parts of input signal.
        self.avgnoise = self.compute_noise_avg_spectrum()
        for l in range(self.windownum):
            ltsds[l] = self.ltsd(l)
        return ltsds

    def ltse(self, l):
        maxmag = np.zeros(self.winsize)
        for idx in range(l - self.order, l + self.order + 1):
            amp = self.get_amplitude(idx)
            maxmag = np.maximum(maxmag, amp)
        return maxmag

    def ltsd(self, l):
        if l < self.order or l + self.order >= self.windownum:
            return 0

        self.avgnoise[np.where(self.avgnoise == 0)] = 1
        return np.sum(self.ltse(l) / self.avgnoise) / len(self.avgnoise)


if __name__ == "__main__":
    # signal = np.load("C:/Users/Zber/Desktop/Subjects/Test/music_time_angle.npy")

    # standing gt: 31, ours: 34 3 degree
    # ground: gt:52, ours: 50
    # sit: gt: 42, ours: 40

    # signal = np.load("C:/Users/Zber/Desktop/Subjects/Test/music_time_angle_ground.npy")
    signal = np.load("C:/Users/Zber/Desktop/Subjects/Test/music_time_angle_sit_3.npy")

    noise_window = 30
    min = np.min(signal)
    max = np.max(signal)
    bins = np.linspace(min, max, 30)
    bin_size = len(bins)
    n_order = 5

    ltsd = LTSD(signal, bin_size, 5, noise_window)
    res = ltsd.compute()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    detect_face_point = np.where(res > 5)[0][0]
    print(detect_face_point)
    ax.plot(res)
    ax.axvline(detect_face_point)
    plt.show()

    # hanning_window = np.hanning(91) * np.ones((91)) * 5

    # np.where(res > hanning_window)

    print("")
