import numpy as np
import math
import os
import matplotlib._color_data as mcd

tab_color = [mcd.TABLEAU_COLORS[name] for name in mcd.TABLEAU_COLORS]
extra_color = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04']
colors = tab_color + extra_color
ROOT_PATH = "C:/Users/Zber/Documents/Dev_program/OpenRadar"


def get_label(name):
    labels = {'Joy': 1, 'Surprise': 2, 'Anger': 3, 'Sadness': 4, 'Fear': 5, 'Disgust': 6, 'Neutral': 0}
    for key in labels.keys():
        if key in name:
            return labels[key]


def save_to_json(data, path):
    import json
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4)


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


def correction_factor(win_type='Hanning', numADCSamples=256):
    WIN_EC = {"Hanning": 1.63, "Flattop": 2.26, "Blackman": 1.97, "Hamming": 1.59}

    numBits = 16

    cf = 20 * math.log10(2 ** (numBits - 1)) + 20 * math.log10(numADCSamples * WIN_EC[win_type]) - 20 * math.log10(
        math.sqrt(2))
    return cf


class MapRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])


    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def relative_path(self):
        return self._data[0]

    @property
    def num_frames(self):
        if self.offset == -1:
            return self.peak - self.onset + 1  # +1 because end frame is inclusive
        else:
            return self.offset - self.onset + 1  # +1 because end frame is inclusive
    @property
    def onset(self):
        return int(self._data[2])

    @onset.setter
    def onset(self, value):
        self._data[2] = value

    @property
    def peak(self):
        return int(self._data[3])

    @peak.setter
    def peak(self, value):
        self._data[3] = value

    @property
    def offset(self):
        return int(self._data[4])

    @offset.setter
    def offset(self, value):
        self._data[4] = value

    @property
    def width_err(self):
        return int(self._data[5])

    @property
    def height_err(self):
        return int(self._data[6])

    @property
    def index_err(self):
        return int(self._data[7])

    @property
    def label(self):
        return int(self._data[1])