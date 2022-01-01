import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import os
from FER.utils import parseConfigFile, arange_tx, get_label
from queue import Queue
import threading
from mmwave.dsp.utils import Window

# configure file
configFileName = 'C:/Users/Zber/Desktop/mmWave Configuration/profile_3d_aop_3s.cfg'


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


def save_phase_data(bin_path, save_path, start_bin, end_bin, num_frames, chirp_index=5, is_diff=True):
    if is_diff:
        num_frames -= 1

    # load Numpy Data
    adc_data = np.fromfile(bin_path, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)
    range_data = dsp.range_processing(adc_data, window_type_1d=Window.BLACKMAN)
    range_data = arange_tx(range_data, num_tx=numTxAntennas)
    range_data = range_data[:, chirp_index, :, start_bin:end_bin]

    range_data = range_data.transpose((1, 2, 0))

    # angle and unwrap
    sig_phase = np.angle(range_data)
    sig_phase = np.unwrap(sig_phase)

    # save file
    np.save(save_path, sig_phase)
    print("{} npy file saved!".format(save_path))


def thread_job(queue, bin_path, out_path):
    while not queue.empty():
        q = queue.get()
        bpath = os.path.join(bin_path, q)
        hpath = os.path.join(out_path, q.replace("_Raw_0.bin", ""))
        save_phase_data(bpath, hpath, start_bin=bin_start, end_bin=bin_end, num_frames=numFrames,
                        chirp_index=chirp_index, is_diff=is_diff)
        queue.task_done()


if __name__ == '__main__':

    root_path = "D:\\Subjects"
    data_path = "{}_{}_Raw_0.bin"
    output_data_path = "C:\\Users\\Zber\\Desktop\\Subjects_Phase"

    # load radar configuration
    numTxAntennas = 3
    numRxAntennas = 4
    configParameters = parseConfigFile(configFileName, numTxAnt=numTxAntennas)
    numFrames = configParameters['numFrames']
    numADCSamples = configParameters['numAdcSamples']
    numLoopsPerFrame = configParameters['numLoops']
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
    numRangeBins = numADCSamples
    numDopplerBins = numLoopsPerFrame

    # data settings
    is_diff = True
    save_config = True
    bin_start = 5
    bin_end = 15
    chirp_index = 5

    # start index
    subs = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    # subs = ['S0']
    emotion_list = ['Joy', 'Surprise', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Neutral']

    queue = Queue()

    start_index = 0
    end_index = 30

    for sub in subs:
        for l, e in enumerate(emotion_list):
            for i in range(start_index, end_index):
                bin_path = os.path.join(root_path, sub, data_path.format(e, i))
                relative_path = os.path.join(sub, data_path.format(e, i))
                queue.put(relative_path)

    if save_config:
        import json

        config = {
            "Different": str(is_diff),
            "Bin Start": bin_start,
            "Bin End": bin_end,
            "Data Start Index": start_index,
            "Data End Index": end_index,
        }

        with open(os.path.join(output_data_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    # q = queue.get()
    # bpath = os.path.join(root_path, q)
    # hpath = os.path.join(output_data_path, q.replace("_Raw_0.bin", ""))
    # save_phase_data(bpath, hpath, start_bin=bin_start, end_bin=bin_end, num_frames=numFrames, chirp_index=chirp_index,
    #                 is_diff=is_diff)

    NUM_THREADS = 16
    for i in range(NUM_THREADS):
        worker = threading.Thread(target=thread_job, args=(queue, root_path, output_data_path))
        worker.start()

    print('waiting for all tasks to be completed.', queue.qsize(), 'tasks')
    print('This can take an hour or two depending on dataset size')
    queue.join()
    print('all done')
